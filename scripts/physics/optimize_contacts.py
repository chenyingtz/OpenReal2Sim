#!/usr/bin/env python3
"""Step 5.2 — Contact and Penetration Optimization (SDF solver).

Seat the LEGO robot dog on the terrain mesh: lift / drop the robot along
the gravity axis so the deepest foot is exactly `clearance` above the
floor SDF's zero level-set, with no inter-penetration.

Matches the CLI from the implementation steps PDF:

    python scripts/physics/optimize_contacts.py \
        --robot_mesh    assets/lego_dog.obj \
        --terrain_mesh  assets/terrain.obj \
        --output_dir    assets/refined/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh

from gravity_align import align_to_gravity, load_mesh


class TerrainSDF:
    """Height-field signed distance for a gravity-aligned terrain.

    Sign convention: positive above the floor, negative below.
    Implemented as a ray-cast straight down (-Z) onto the terrain mesh.
    This is robust for open / non-watertight terrain crops, whereas a
    classical inside/outside SDF needs a watertight surface.
    Requires the terrain to already be roughly +Z aligned (Step 5.1).
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self._ri = mesh.ray
        # Cast from well above the terrain to guarantee a hit.
        self._z_cast = float(mesh.bounds[1, 2]) + 1.0

    def signed_distance(self, points: np.ndarray):
        """Return (signed_distance, hit_mask).

        For points whose XY falls inside the terrain's footprint the ray
        hits the surface and the signed distance is `z - z_terrain`.
        Missed rays get sd = +1.0 (a large positive constant) so they
        contribute no penetration and no gradient — they're effectively
        ignored. The caller should consult `hit_mask` to decide whether
        the terrain crop is too small.
        """
        points = np.asarray(points, dtype=np.float64)
        origins = np.column_stack([points[:, 0], points[:, 1],
                                   np.full(len(points), self._z_cast)])
        directions = np.tile([0.0, 0.0, -1.0], (len(points), 1))
        locations, ray_idx, _ = self._ri.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False
        )
        hit_mask = np.zeros(len(points), dtype=bool)
        sd = np.ones(len(points), dtype=np.float64)  # missed → +1.0 m
        if len(ray_idx):
            hit_mask[ray_idx] = True
            sd[ray_idx] = points[ray_idx, 2] - locations[:, 2]
        return sd, hit_mask


def _foot_samples(robot: trimesh.Trimesh, n_sample: int, foot_band: float) -> np.ndarray:
    """Surface samples concentrated near the lowest part of the robot.

    `foot_band` is the height (in meters) above the minimum-Z point that
    counts as "foot region" — only those samples drive the contact loss,
    which keeps the optimizer from gluing the torso to the floor.
    """
    n = min(n_sample, max(2000, 5 * len(robot.vertices)))
    samples, _ = trimesh.sample.sample_surface(robot, n)
    samples = np.asarray(samples)
    z_min = float(samples[:, 2].min())
    feet = samples[samples[:, 2] <= z_min + foot_band]
    if len(feet) < 50:
        # Robot is small / nearly flat — just use everything.
        feet = samples
    return feet


def optimize_z_offset(
    robot: trimesh.Trimesh,
    sdf: TerrainSDF,
    clearance: float = 0.002,
    foot_band: float = 0.015,
    n_sample: int = 8000,
    n_iter: int = 200,
    lr: float = 5e-3,
    max_step: float = 2e-3,
) -> Tuple[trimesh.Trimesh, dict]:
    """Translate the robot along +Z to remove penetration with the terrain.

    Solves   min_tz  mean( ReLU(-sdf(p + tz e_z))^2 )
    via projected gradient on a finite-difference estimate of d sdf / d z.
    After convergence we add an explicit clearance lift so the final
    minimum signed distance is exactly `clearance`.
    """
    feet = _foot_samples(robot, n_sample, foot_band)
    initial_sd, initial_hits = sdf.signed_distance(feet)
    coverage = float(initial_hits.mean())
    if coverage < 0.5:
        print(f"[contacts] WARNING: only {coverage*100:.1f}% of foot samples "
              f"sit above the terrain mesh — terrain crop is likely smaller "
              f"than the robot's XY footprint. Optimization will use only "
              f"covered samples; results may be incorrect.")
    if coverage == 0.0:
        raise RuntimeError(
            "No foot samples hit the terrain mesh. The robot and terrain "
            "footprints do not overlap in XY — re-run Step 4 producing a "
            "terrain mesh that covers under the robot, or pass --skip_gravity "
            "if the meshes are not in the same frame."
        )

    tz = 0.0
    eps = 1e-3
    history = []

    for it in range(n_iter):
        pts = feet + np.array([0.0, 0.0, tz])
        sdf_vals, hits = sdf.signed_distance(pts)
        if not hits.any():
            break
        pen_full = np.clip(-sdf_vals, 0.0, None)
        loss = float((pen_full[hits] ** 2).mean())
        history.append(loss)
        if loss < 1e-10:
            break
        pts_up = pts + np.array([0.0, 0.0, eps])
        sdf_up, _ = sdf.signed_distance(pts_up)
        d_pen = -(sdf_up - sdf_vals) / eps  # d(penetration)/dz, with sign mask
        grad_full = 2.0 * pen_full * d_pen * (pen_full > 0)
        grad = float(grad_full[hits].mean())
        step = float(np.clip(-lr * grad, -max_step, max_step))
        if not np.isfinite(step):
            break
        tz += step

    # Final clearance bump: lift so deepest covered foot is `clearance` above.
    pts = feet + np.array([0.0, 0.0, tz])
    sd_final, hits_final = sdf.signed_distance(pts)
    if hits_final.any():
        min_dist = float(sd_final[hits_final].min())
        if min_dist < clearance:
            tz += (clearance - min_dist)

    if not np.isfinite(tz):
        raise RuntimeError(
            f"Contact optimizer produced non-finite delta_z (={tz}). "
            "Check that the robot and terrain meshes are in the same metric "
            "frame and that the terrain covers the robot's XY footprint."
        )

    refined = robot.copy()
    T = np.eye(4)
    T[2, 3] = tz
    refined.apply_transform(T)

    sd_done, hits_done = sdf.signed_distance(feet + np.array([0.0, 0.0, tz]))
    final_min = float(sd_done[hits_done].min()) if hits_done.any() else float("nan")
    info = {
        "delta_z": float(tz),
        "final_min_signed_distance": final_min,
        "foot_terrain_coverage": coverage,
        "iterations": len(history),
        "loss_history": history[::max(1, len(history) // 32)] if history else [],
    }
    return refined, info


def run_step5_alignment(
    robot_path: str | Path,
    terrain_path: str | Path,
    output_dir: str | Path,
    clearance: float = 0.002,
    do_gravity_align: bool = True,
    seed: int = 0,
) -> dict:
    """End-to-end Step 5.1 + 5.2 pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    terrain = load_mesh(terrain_path)
    robot = load_mesh(robot_path)

    if do_gravity_align:
        terrain, robot, grav_info = align_to_gravity(terrain, robot, seed=seed)
    else:
        grav_info = {"skipped": True}

    sdf = TerrainSDF(terrain)
    refined_robot, contact_info = optimize_z_offset(robot, sdf, clearance=clearance)

    terrain_out = output_dir / "terrain.obj"
    robot_out = output_dir / "lego_dog.obj"
    terrain.export(terrain_out)
    refined_robot.export(robot_out)

    report = {
        "robot_mesh_input": str(robot_path),
        "terrain_mesh_input": str(terrain_path),
        "robot_mesh_refined": str(robot_out),
        "terrain_mesh_refined": str(terrain_out),
        "clearance": clearance,
        "gravity": grav_info,
        "contacts": contact_info,
    }
    with open(output_dir / "alignment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[contacts] delta_z      = {contact_info['delta_z']:+.5f} m")
    print(f"[contacts] min sdf      = {contact_info['final_min_signed_distance']:+.5f} m")
    print(f"[contacts] refined dog  -> {robot_out}")
    print(f"[contacts] refined floor-> {terrain_out}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Step 5 — Physical grounding & alignment")
    ap.add_argument("--robot_mesh", required=True)
    ap.add_argument("--terrain_mesh", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clearance", type=float, default=0.002,
                    help="Min signed distance between robot feet and terrain (meters)")
    ap.add_argument("--skip_gravity", action="store_true",
                    help="Assume terrain is already +Z aligned; only run contact optimization")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_step5_alignment(
        robot_path=args.robot_mesh,
        terrain_path=args.terrain_mesh,
        output_dir=args.output_dir,
        clearance=args.clearance,
        do_gravity_align=not args.skip_gravity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
