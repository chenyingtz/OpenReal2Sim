#!/usr/bin/env python3
"""Step 5.1 — Gravity Plane Alignment.

Fit a ground plane to the terrain mesh M^b via RANSAC, build a rotation
matrix R_grav that maps the plane normal onto +Z (the simulator's gravity
axis e_z), and apply R_grav to both terrain and robot meshes so the world
sits "flat" with gravity pointing down -Z.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


E_Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _fit_plane_lstsq(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    return centroid, normal / (np.linalg.norm(normal) + 1e-12)


def ransac_plane(
    points: np.ndarray,
    distance_threshold: float = 0.01,
    n_iterations: int = 1000,
    min_inlier_ratio: float = 0.3,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (point_on_plane, unit_normal, inlier_mask).

    Standard 3-point RANSAC followed by a least-squares refit on inliers.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_pts = len(points)
    if n_pts < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    best_inliers: np.ndarray | None = None
    best_count = -1

    for _ in range(n_iterations):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        dists = np.abs((points - p0) @ normal)
        inliers = dists < distance_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < max(3, int(min_inlier_ratio * n_pts)):
        # Fallback: trust a global PCA fit.
        centroid, normal = _fit_plane_lstsq(points)
        return centroid, normal, np.ones(n_pts, dtype=bool)

    centroid, normal = _fit_plane_lstsq(points[best_inliers])
    return centroid, normal, best_inliers


def rotation_to_z(normal: np.ndarray) -> np.ndarray:
    """Rotation matrix R that maps the unit vector `normal` onto +Z."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    # The plane normal could point either way; pick the orientation that
    # has positive z so the floor ends up below the scene rather than above.
    if n[2] < 0:
        n = -n
    v = np.cross(n, E_Z)
    s = np.linalg.norm(v)
    c = float(np.dot(n, E_Z))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def apply_rotation(mesh: trimesh.Trimesh, R: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    out.apply_transform(T)
    return out


def align_to_gravity(
    terrain_mesh: trimesh.Trimesh,
    robot_mesh: trimesh.Trimesh | None = None,
    n_samples: int = 20000,
    distance_threshold: float = 0.01,
    n_iterations: int = 1000,
    seed: int = 0,
):
    """Compute R_grav from the terrain and apply it to terrain (+robot)."""
    sample_count = min(n_samples, max(1000, len(terrain_mesh.vertices)))
    samples, _ = trimesh.sample.sample_surface(terrain_mesh, sample_count)
    rng = np.random.default_rng(seed)
    p0, normal, inliers = ransac_plane(
        np.asarray(samples),
        distance_threshold=distance_threshold,
        n_iterations=n_iterations,
        rng=rng,
    )
    R_grav = rotation_to_z(normal)
    aligned_terrain = apply_rotation(terrain_mesh, R_grav)

    # After rotation, the floor's signed offset along Z is constant for
    # inlier points; push it to z = 0 so the simulator's ground sits at 0.
    floor_pts = (R_grav @ np.asarray(samples)[inliers].T).T
    z_floor = float(np.median(floor_pts[:, 2]))
    T = np.eye(4)
    T[2, 3] = -z_floor
    aligned_terrain.apply_transform(T)

    aligned_robot = None
    if robot_mesh is not None:
        aligned_robot = apply_rotation(robot_mesh, R_grav)
        aligned_robot.apply_transform(T)

    info = {
        "plane_point_world": p0.tolist(),
        "plane_normal_world": normal.tolist(),
        "inlier_ratio": float(inliers.mean()),
        "R_grav": R_grav.tolist(),
        "z_floor_offset": z_floor,
    }
    return aligned_terrain, aligned_robot, info


def main():
    ap = argparse.ArgumentParser(description="Step 5.1 — Gravity plane alignment")
    ap.add_argument("--robot_mesh", required=True)
    ap.add_argument("--terrain_mesh", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--distance_threshold", type=float, default=0.01)
    ap.add_argument("--n_iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    terrain = load_mesh(args.terrain_mesh)
    robot = load_mesh(args.robot_mesh)

    aligned_terrain, aligned_robot, info = align_to_gravity(
        terrain, robot,
        distance_threshold=args.distance_threshold,
        n_iterations=args.n_iterations,
        seed=args.seed,
    )

    terrain_out = out / "terrain_gravity_aligned.obj"
    robot_out = out / "lego_dog_gravity_aligned.obj"
    aligned_terrain.export(terrain_out)
    aligned_robot.export(robot_out)
    with open(out / "gravity_alignment.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[gravity] inlier ratio = {info['inlier_ratio']:.3f}")
    print(f"[gravity] terrain  -> {terrain_out}")
    print(f"[gravity] robot    -> {robot_out}")


if __name__ == "__main__":
    main()
