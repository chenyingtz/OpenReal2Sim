#!/usr/bin/env python3
"""Step 4.3 — Assemble the Digital Twin Scene.

Reads the YAML config (configs/lego_dog_locomotion.yaml), picks up the
robot mesh from Step 4.2a and the terrain mesh from Step 4.2b, and:

  1. Scales the robot to its physical length (default 0.20 m along its
     longest principal axis; override with --robot_length_m or in the
     config under `assets.robot_length_m`).
  2. Drops the terrain so its median height sits at z = 0 (gravity-align
     comes later in Step 5, this is just to bring meshes into the same
     ballpark for inspection).
  3. Positions the robot above the terrain centroid, with its lowest
     vertex `clearance_m` above the floor.
  4. Writes a combined scene.glb (both meshes), saves them individually
     into `<asset_dir>/<robot_mesh_name>` and `<asset_dir>/<terrain_mesh_name>`
     so Step 5's `run_step5.py` can ingest them as-is.

Matches the PDF CLI:

    python scripts/assets/assemble_digital_twin.py \\
        --config configs/lego_dog_locomotion.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
import yaml


def _load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def _principal_length(mesh: trimesh.Trimesh) -> float:
    """Longest of the AABB extents — a stable proxy for 'robot length'."""
    return float(np.max(mesh.extents))


def _resize_to_length(mesh: trimesh.Trimesh, target_length_m: float) -> trimesh.Trimesh:
    """Uniform scale so the longest AABB axis equals target_length_m."""
    cur = _principal_length(mesh)
    if cur <= 0:
        raise RuntimeError("Robot mesh has degenerate extents; cannot scale.")
    s = target_length_m / cur
    out = mesh.copy()
    out.apply_scale(s)
    return out


def _level_terrain_to_zero(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop the terrain so its median Z lands on 0.

    The depth-derived terrain is in camera frame where +Z is the depth
    axis. Step 5's gravity-alignment will rotate it properly later; here
    we just shift so the trainer's MuJoCo viewer has a sensible camera.
    """
    out = mesh.copy()
    if len(out.vertices) == 0:
        return out
    z_median = float(np.median(out.vertices[:, 2]))
    T = np.eye(4)
    T[2, 3] = -z_median
    out.apply_transform(T)
    return out


def _seat_robot_on_terrain(
    robot: trimesh.Trimesh, terrain: trimesh.Trimesh, clearance_m: float
) -> Tuple[trimesh.Trimesh, dict]:
    """Position the robot above the centroid of the terrain footprint, with
    its lowest vertex `clearance_m` above the terrain's surface at that
    point (using the median terrain Z as a robust floor proxy)."""
    out = robot.copy()
    if len(terrain.vertices) == 0:
        terrain_xy = np.zeros(2)
        terrain_z = 0.0
    else:
        terrain_xy = terrain.vertices[:, :2].mean(axis=0)
        terrain_z = float(np.median(terrain.vertices[:, 2]))

    # Move the robot so its lowest vertex sits at terrain_z + clearance.
    robot_min = float(out.vertices[:, 2].min())
    dz = (terrain_z + clearance_m) - robot_min
    # Move XY so the robot is centered on the terrain's XY centroid.
    centroid_xy = out.vertices[:, :2].mean(axis=0)
    dxy = terrain_xy - centroid_xy
    T = np.eye(4)
    T[0, 3] = dxy[0]
    T[1, 3] = dxy[1]
    T[2, 3] = dz
    out.apply_transform(T)
    return out, {
        "robot_translation": [float(dxy[0]), float(dxy[1]), float(dz)],
        "terrain_z_median": terrain_z,
        "terrain_xy_centroid": [float(terrain_xy[0]), float(terrain_xy[1])],
        "clearance_m": clearance_m,
    }


def assemble(
    config_path: Path,
    asset_dir: Optional[Path] = None,
    robot_mesh_path: Optional[Path] = None,
    terrain_mesh_path: Optional[Path] = None,
    robot_mesh_name: str = "lego_dog.obj",
    terrain_mesh_name: str = "terrain.obj",
    robot_length_m: Optional[float] = None,
    clearance_m: float = 0.005,
) -> Path:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    repo_root = config_path.parent.parent
    assets_cfg = (cfg.get("assets") or {})
    if robot_length_m is None:
        robot_length_m = float(assets_cfg.get("robot_length_m", 0.20))

    if asset_dir is None:
        # Default: write next to the URDF location declared in the config.
        urdf_field = (cfg.get("scene") or {}).get("urdf")
        if urdf_field:
            asset_dir = (Path(urdf_field) if Path(urdf_field).is_absolute()
                         else repo_root / urdf_field).parent
        else:
            asset_dir = repo_root / "assets" / "refined"
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Locate the input meshes. Default search paths follow the conventions
    # from generate_robot_mesh.py and generate_terrain_mesh.py.
    if robot_mesh_path is None:
        for cand in (
            assets_cfg.get("robot_mesh"),
            asset_dir / robot_mesh_name,
            repo_root / "data/lego_dog_walk/meshes" / robot_mesh_name,
        ):
            if cand and Path(cand).exists():
                robot_mesh_path = Path(cand)
                break
    if terrain_mesh_path is None:
        for cand in (
            assets_cfg.get("terrain_mesh"),
            asset_dir / terrain_mesh_name,
            repo_root / "data/lego_dog_walk/meshes" / terrain_mesh_name,
        ):
            if cand and Path(cand).exists():
                terrain_mesh_path = Path(cand)
                break
    if robot_mesh_path is None:
        raise FileNotFoundError(
            "Robot mesh not found. Run scripts/assets/generate_robot_mesh.py "
            "first, or pass --robot_mesh explicitly."
        )
    if terrain_mesh_path is None:
        raise FileNotFoundError(
            "Terrain mesh not found. Run scripts/assets/generate_terrain_mesh.py "
            "first, or pass --terrain_mesh explicitly."
        )

    robot = _load_mesh(Path(robot_mesh_path))
    terrain = _load_mesh(Path(terrain_mesh_path))
    print(f"[assemble] robot   : {robot_mesh_path}  "
          f"verts={len(robot.vertices)}  length={_principal_length(robot):.3f}")
    print(f"[assemble] terrain : {terrain_mesh_path}  "
          f"verts={len(terrain.vertices)}  extents={terrain.extents.tolist()}")

    # 1. Scale robot
    robot_scaled = _resize_to_length(robot, target_length_m=robot_length_m)
    print(f"[assemble] robot scaled to length {robot_length_m:.3f} m "
          f"(scale = {robot_length_m / _principal_length(robot):.4f})")

    # 2. Level terrain
    terrain_leveled = _level_terrain_to_zero(terrain)

    # 3. Seat robot on terrain
    robot_seated, seat_info = _seat_robot_on_terrain(
        robot_scaled, terrain_leveled, clearance_m=clearance_m,
    )
    print(f"[assemble] seat info: {seat_info}")

    # 4. Save individual meshes (Step 5's run_step5.py reads these names).
    robot_out = asset_dir / robot_mesh_name
    terrain_out = asset_dir / terrain_mesh_name
    robot_seated.export(str(robot_out))
    terrain_leveled.export(str(terrain_out))

    # 5. Save combined GLB for inspection.
    scene = trimesh.Scene()
    scene.add_geometry(terrain_leveled, "terrain")
    scene.add_geometry(robot_seated, "lego_dog")
    combined_out = asset_dir / "scene_assembled.glb"
    scene.export(str(combined_out))

    manifest = {
        "config": str(config_path),
        "robot_mesh_input": str(robot_mesh_path),
        "terrain_mesh_input": str(terrain_mesh_path),
        "robot_mesh_output": str(robot_out),
        "terrain_mesh_output": str(terrain_out),
        "scene_glb": str(combined_out),
        "robot_length_m": robot_length_m,
        "seat_info": seat_info,
    }
    with open(asset_dir / "_assemble_meta.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[assemble] robot     -> {robot_out}")
    print(f"[assemble] terrain   -> {terrain_out}")
    print(f"[assemble] scene glb -> {combined_out}")
    print(f"[assemble] manifest  -> {asset_dir / '_assemble_meta.json'}")
    return combined_out


def main():
    ap = argparse.ArgumentParser(description="Step 4.3 — Assemble the digital twin scene")
    ap.add_argument("--config", required=True,
                    help="Path to lego_dog_locomotion.yaml")
    ap.add_argument("--asset_dir", default=None,
                    help="Output directory (default: parent of config.scene.urdf)")
    ap.add_argument("--robot_mesh", default=None,
                    help="Input robot mesh (default: <asset_dir>/lego_dog.obj)")
    ap.add_argument("--terrain_mesh", default=None,
                    help="Input terrain mesh (default: <asset_dir>/terrain.obj)")
    ap.add_argument("--robot_mesh_name", default="lego_dog.obj")
    ap.add_argument("--terrain_mesh_name", default="terrain.obj")
    ap.add_argument("--robot_length_m", type=float, default=None,
                    help="Real-world length of the LEGO dog along its long axis "
                         "(default: assets.robot_length_m in config, else 0.20 m).")
    ap.add_argument("--clearance_m", type=float, default=0.005,
                    help="Spawn the robot's lowest vertex this far above the floor.")
    args = ap.parse_args()

    assemble(
        config_path=Path(args.config),
        asset_dir=Path(args.asset_dir) if args.asset_dir else None,
        robot_mesh_path=Path(args.robot_mesh) if args.robot_mesh else None,
        terrain_mesh_path=Path(args.terrain_mesh) if args.terrain_mesh else None,
        robot_mesh_name=args.robot_mesh_name,
        terrain_mesh_name=args.terrain_mesh_name,
        robot_length_m=args.robot_length_m,
        clearance_m=args.clearance_m,
    )


if __name__ == "__main__":
    main()
