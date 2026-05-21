#!/usr/bin/env python3
"""Step 6 — Torso Reference Trajectory Extraction.

Track the 6D rigid-body transform of the LEGO dog's torso link across every
frame of the generated video using the gravity-aligned mesh from Step 5 and
the metric-accurate dynamic point clouds {P_t} from Step 3.

Matches the CLI from the implementation-steps PDF:

    python scripts/tracking/run_foundationpose.py \
        --mesh             assets/refined/lego_dog.obj \
        --point_clouds     data/lego_dog_walk/metric_point_clouds/ \
        --output_trajectory data/lego_dog_walk/torso_trajectory.json

Outputs:
- `--output_trajectory` JSON with per-frame {position, quaternion} matching
  x_t^o = [p_t^o, q_t^o] in the project spec (consumed by Step 7's reward).
- A sibling `<basename>.npy` containing [T+1, 4, 4] SE3 matrices for the
  repo's existing motion-stage consumers (openreal2sim/motion/...).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from pose_tracker import (
    _initial_pose_search,
    _load_point_cloud,
    discover_frames,
    enforce_quat_continuity,
    point_to_mesh_icp,
    quat_xyzw_from_R,
)


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def extract_trajectory(
    mesh_path: str | Path,
    point_cloud_dir: str | Path,
    output_path: str | Path,
    fps: float = 30.0,
    init_yaw_samples: int = 8,
) -> dict:
    mesh_path = Path(mesh_path)
    point_cloud_dir = Path(point_cloud_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = _load_mesh(mesh_path)
    frame_paths = discover_frames(point_cloud_dir)
    print(f"[track] mesh   : {mesh_path}  ({len(mesh.vertices)} verts)")
    print(f"[track] frames : {len(frame_paths)} in {point_cloud_dir}")

    trajectory = []
    se3_stack = np.zeros((len(frame_paths), 4, 4), dtype=np.float64)
    R_prev = t_prev = q_prev = None

    for i, p in enumerate(frame_paths):
        points = _load_point_cloud(p)
        if i == 0:
            R, t, residual = _initial_pose_search(mesh, points,
                                                  n_yaw_samples=init_yaw_samples)
        else:
            R, t, residual = point_to_mesh_icp(mesh, points,
                                               R0=R_prev, t0=t_prev)
        q = quat_xyzw_from_R(R)
        if q_prev is not None:
            q = enforce_quat_continuity(q_prev, q)

        trajectory.append({
            "frame": i,
            "time": i / float(fps),
            "source": p.name,
            "position": [float(x) for x in t],
            "quaternion": [float(x) for x in q],
            "rotation_matrix": R.tolist(),
            "icp_residual": float(residual),
        })
        se3_stack[i] = _make_se3(R, t)

        R_prev, t_prev, q_prev = R, t, q
        if (i + 1) % 10 == 0 or i in (0, len(frame_paths) - 1):
            print(f"[track] frame {i+1}/{len(frame_paths)}   "
                  f"pos=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]  "
                  f"residual={residual*1000:.2f} mm")

    out = {
        "schema_version": "1.0",
        "frame_count": len(frame_paths),
        "fps": fps,
        "mesh": str(mesh_path),
        "point_cloud_dir": str(point_cloud_dir),
        "backend": "icp",
        "convention": {
            "quaternion": "xyzw",
            "position_units": "meters",
            "frame": "world (gravity-aligned, +Z up)",
        },
        "trajectory": trajectory,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, se3_stack)

    print(f"[track] wrote {output_path}")
    print(f"[track] wrote {npy_path}  (shape={se3_stack.shape})")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Step 6 — Torso Reference Trajectory Extraction (ICP backend)"
    )
    ap.add_argument("--mesh", required=True,
                    help="Robot mesh (refined output from Step 5)")
    ap.add_argument("--point_clouds", required=True,
                    help="Directory of per-frame point clouds {P_t} from Step 3")
    ap.add_argument("--output_trajectory", required=True,
                    help="Output JSON path for the torso trajectory")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Source video frame rate; only used for the 'time' field")
    ap.add_argument("--init_yaw_samples", type=int, default=8,
                    help="Number of yaw seeds for the frame-0 multi-start ICP")
    ap.add_argument("--no_foundationpose", action="store_true",
                    help="Reserved for symmetry; the ICP backend is the only "
                         "supported backend for point-cloud-directory inputs.")
    args = ap.parse_args()

    extract_trajectory(
        mesh_path=args.mesh,
        point_cloud_dir=args.point_clouds,
        output_path=args.output_trajectory,
        fps=args.fps,
        init_yaw_samples=args.init_yaw_samples,
    )


if __name__ == "__main__":
    main()
