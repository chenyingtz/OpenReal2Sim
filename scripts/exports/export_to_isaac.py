#!/usr/bin/env python3
"""Step 7.1 — Export Workspace Assets.

Package the refined assets from Step 5 + the torso reference trajectory from
Step 6 into a self-contained directory the residual-RL trainer can consume.

Matches the PDF CLI:

    python scripts/exports/export_to_isaac.py \
        --asset_dir assets/refined/ \
        --trajectory data/lego_dog_walk/torso_trajectory.json

The exporter is sim-backend-aware:
  - Always: copies URDF + meshes, normalizes the trajectory, writes
    `scene.yaml` + `trajectory.npz` + an MJCF wrapper so MuJoCo can load
    everything via `mujoco.MjModel.from_xml_path()`. The MuJoCo backend is
    the primary supported target because it runs anywhere.
  - Isaac Lab (optional): if `omni.isaac.lab` is importable, generates a
    `.usd` scene as well. Skipped with an informative message otherwise.

The exported directory structure under `<asset_dir>/isaac_export/`:

    isaac_export/
    ├── lego_dog.urdf              (copy)
    ├── lego_dog.obj               (copy, robot mesh)
    ├── terrain.obj                (copy, ground mesh; optional)
    ├── scene.mjcf                 (MJCF wrapper: includes URDF + ground)
    ├── trajectory.npz             (times[T], positions[T,3], quaternions[T,4])
    ├── trajectory.json            (copy of input)
    └── scene.yaml                 (paths + joint names + frame_0 offset)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def _resolve(asset_dir: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (asset_dir / p)


def _normalize_trajectory(trajectory_json: dict) -> dict:
    """Pull out (times, positions, quaternions) and the frame-0 offset.

    The PDF + Step 6 emit `{p_t, q_t}` per-frame in xyzw quaternion order
    and meters. Step 7's reward compares simulated torso pose to this
    sequence; for sim stability we re-center the trajectory so frame 0
    sits at (x=0, y=0, q=identity). The frame-0 pose is recorded in
    `frame_zero_pose` so any downstream code that needs world-frame
    poses can undo the recentering.
    """
    traj = trajectory_json["trajectory"]
    T = len(traj)
    times = np.array([e["time"] for e in traj], dtype=np.float64)
    positions = np.array([e["position"] for e in traj], dtype=np.float64)
    quaternions = np.array([e["quaternion"] for e in traj], dtype=np.float64)

    p0 = positions[0].copy()
    q0 = quaternions[0].copy()
    centered = positions - p0
    centered[:, 2] = positions[:, 2]  # keep height absolute (gravity-aligned)

    return {
        "times": times,
        "positions": centered,
        "quaternions": quaternions,
        "frame_zero_pose": {
            "position": p0.tolist(),
            "quaternion": q0.tolist(),
        },
        "frame_count": T,
        "fps": float(trajectory_json.get("fps", 30.0)),
    }


def _write_mjcf_wrapper(out_dir: Path, urdf_name: str, terrain_name: Optional[str],
                       ground_friction: float):
    """Write a small MJCF file that includes the URDF and adds a ground plane."""
    ground_geom = ""
    if terrain_name is not None:
        ground_geom = f"""
    <asset>
      <mesh name="terrain" file="{terrain_name}"/>
    </asset>
    <worldbody>
      <geom name="terrain" type="mesh" mesh="terrain" friction="{ground_friction} 0.005 0.0001"/>
    </worldbody>
"""
    else:
        ground_geom = f"""
    <worldbody>
      <geom name="ground" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1.0"
            friction="{ground_friction} 0.005 0.0001"/>
    </worldbody>
"""

    mjcf = f"""<?xml version="1.0"?>
<mujoco model="lego_dog_scene">
  <compiler meshdir="." angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <include file="{urdf_name}"/>
{ground_geom}
</mujoco>
"""
    (out_dir / "scene.mjcf").write_text(mjcf)


def _try_export_usd(out_dir: Path, urdf_path: Path) -> bool:
    """Attempt to generate Isaac Lab USD assets. Returns True on success."""
    try:
        from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg  # type: ignore
    except ImportError:
        return False
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(out_dir),
        usd_file_name="lego_dog.usd",
    )
    UrdfConverter(cfg)
    return True


def export(
    asset_dir: str | Path,
    trajectory_path: str | Path,
    config_path: Optional[str | Path] = None,
    robot_mesh_name: str = "lego_dog.obj",
    terrain_mesh_name: str = "terrain.obj",
    urdf_name: str = "lego_dog.urdf",
    ground_friction: float = 0.9,
) -> Path:
    asset_dir = Path(asset_dir).resolve()
    trajectory_path = Path(trajectory_path).resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"asset_dir not found: {asset_dir}")

    out_dir = asset_dir / "isaac_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy URDF + meshes (URDF references meshes by relative path).
    urdf_src = asset_dir / urdf_name
    if not urdf_src.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_src}")
    shutil.copy2(urdf_src, out_dir / urdf_name)

    robot_mesh_src = asset_dir / robot_mesh_name
    if robot_mesh_src.exists():
        shutil.copy2(robot_mesh_src, out_dir / robot_mesh_name)

    terrain_src = asset_dir / terrain_mesh_name
    terrain_exported = terrain_src.exists()
    if terrain_exported:
        shutil.copy2(terrain_src, out_dir / terrain_mesh_name)

    # Trajectory normalization.
    with open(trajectory_path) as f:
        traj_json = json.load(f)
    norm = _normalize_trajectory(traj_json)
    np.savez(
        out_dir / "trajectory.npz",
        times=norm["times"],
        positions=norm["positions"],
        quaternions=norm["quaternions"],
    )
    shutil.copy2(trajectory_path, out_dir / "trajectory.json")

    # MJCF wrapper so MuJoCo can load the scene out of the box.
    _write_mjcf_wrapper(
        out_dir,
        urdf_name=urdf_name,
        terrain_name=terrain_mesh_name if terrain_exported else None,
        ground_friction=ground_friction,
    )

    # Optional Isaac Lab USD export.
    usd_written = _try_export_usd(out_dir, out_dir / urdf_name)

    # scene.yaml — single source of truth the trainer reads.
    config_passthrough = {}
    if config_path is not None:
        config_passthrough = {"upstream_config": str(Path(config_path).resolve())}

    scene = {
        "schema_version": "1.0",
        "urdf": str(out_dir / urdf_name),
        "mjcf": str(out_dir / "scene.mjcf"),
        "usd": str(out_dir / "lego_dog.usd") if usd_written else None,
        "robot_mesh": str(out_dir / robot_mesh_name) if robot_mesh_src.exists() else None,
        "terrain_mesh": str(out_dir / terrain_mesh_name) if terrain_exported else None,
        "trajectory_json": str(out_dir / "trajectory.json"),
        "trajectory_npz": str(out_dir / "trajectory.npz"),
        "frame_zero_pose": norm["frame_zero_pose"],
        "frame_count": norm["frame_count"],
        "fps": norm["fps"],
        "ground_friction": ground_friction,
        **config_passthrough,
    }
    with open(out_dir / "scene.yaml", "w") as f:
        yaml.safe_dump(scene, f, sort_keys=False)

    print(f"[export] URDF             : {out_dir / urdf_name}")
    print(f"[export] MJCF wrapper     : {out_dir / 'scene.mjcf'}")
    print(f"[export] trajectory.npz   : {norm['frame_count']} frames, "
          f"fps={norm['fps']}")
    print(f"[export] frame-0 offset   : pos={norm['frame_zero_pose']['position']}")
    if usd_written:
        print(f"[export] Isaac Lab USD    : {out_dir / 'lego_dog.usd'}")
    else:
        print(f"[export] Isaac Lab USD    : skipped (omni.isaac.lab not importable). "
              f"MuJoCo backend is fully exported.")
    print(f"[export] scene manifest   : {out_dir / 'scene.yaml'}")
    return out_dir / "scene.yaml"


def main():
    ap = argparse.ArgumentParser(description="Step 7.1 — Export refined assets + trajectory")
    ap.add_argument("--asset_dir", required=True,
                    help="Refined-asset directory from Step 5 (contains the URDF)")
    ap.add_argument("--trajectory", required=True,
                    help="Torso trajectory JSON from Step 6")
    ap.add_argument("--config", default=None,
                    help="Optional path to lego_dog_locomotion.yaml (recorded in scene.yaml)")
    ap.add_argument("--robot_mesh_name", default="lego_dog.obj")
    ap.add_argument("--terrain_mesh_name", default="terrain.obj")
    ap.add_argument("--urdf_name", default="lego_dog.urdf")
    ap.add_argument("--ground_friction", type=float, default=0.9)
    args = ap.parse_args()

    export(
        asset_dir=args.asset_dir,
        trajectory_path=args.trajectory,
        config_path=args.config,
        robot_mesh_name=args.robot_mesh_name,
        terrain_mesh_name=args.terrain_mesh_name,
        urdf_name=args.urdf_name,
        ground_friction=args.ground_friction,
    )


if __name__ == "__main__":
    main()
