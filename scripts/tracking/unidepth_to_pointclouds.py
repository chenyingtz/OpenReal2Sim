#!/usr/bin/env python3
"""Convert UniDepth per-frame .npz depth maps + segmentation masks into the
per-frame .npy point-cloud directory that scripts/tracking/run_foundationpose.py
consumes.

UniDepth's mega-sam wrapper saves each frame as:
    np.savez(out_path,
             depth=np.float32(depth),   # (H, W) metric depth in meters
             fov=fov_deg)               # scalar horizontal FOV (degrees)

That is *not* directly a point cloud, so this script:
  1. Resizes the per-frame dog mask to the depth resolution (nearest-neighbor).
  2. Optionally applies a linear depth correction d' = scale * d + shift
     (i.e. the α, β recovered by Step 3's calibrate_metric_scale.py).
  3. Unprojects only the masked pixels using the FOV-derived intrinsics:
        fx = (W/2) / tan(fov_h / 2),  fy = fx,  cx = W/2,  cy = H/2
        x = (u - cx) * d / fx,  y = (v - cy) * d / fy,  z = d
  4. Saves each frame as frame_NNNN.npy of shape (N, 3) in camera frame.

The output is suitable as `--point_clouds` for run_foundationpose.py — the
tracker handles the camera-vs-world frame difference implicitly by absorbing
it into the recovered pose at frame 0.

Usage:
    python scripts/tracking/unidepth_to_pointclouds.py \
        --depth_dir  third_party/mega-sam/UniDepth/UniDepth_outputs/<scene>/ \
        --mask_dir   data/lego_dog_walk/masks/ \
        --output_dir data/lego_dog_walk/metric_point_clouds/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def _natural_sort(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=lambda p: (len(p.stem), p.stem))


def _pair_files(depth_files: List[Path], mask_files: List[Path]
                ) -> List[Tuple[Path, Path]]:
    """Match depth and mask files by stem when possible, else by sorted order."""
    depth_files = _natural_sort(depth_files)
    mask_files = _natural_sort(mask_files)
    by_stem = {m.stem: m for m in mask_files}
    pairs: List[Tuple[Path, Path]] = []
    for d in depth_files:
        m = by_stem.get(d.stem)
        if m is None:
            break
        pairs.append((d, m))
    if len(pairs) == len(depth_files):
        return pairs
    if len(depth_files) != len(mask_files):
        raise ValueError(
            f"Cannot pair files by stem and counts differ: "
            f"{len(depth_files)} depth vs {len(mask_files)} mask. "
            f"Either rename so stems match or trim to equal counts."
        )
    return list(zip(depth_files, mask_files))


def _load_mask(path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"))
    H, W = target_hw
    if img.shape != (H, W):
        img = np.array(Image.fromarray(img).resize((W, H), Image.NEAREST))
    return img > 127


def _unproject(depth: np.ndarray, fov_deg: float, mask: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    fov_rad = float(np.deg2rad(fov_deg))
    fx = (W / 2.0) / float(np.tan(fov_rad / 2.0))
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    vs, us = np.where(mask)
    if vs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    d = depth[vs, us]
    valid = d > 0
    vs, us, d = vs[valid], us[valid], d[valid]
    x = (us - cx) * d / fx
    y = (vs - cy) * d / fy
    return np.stack([x, y, d], axis=1).astype(np.float32)


def convert(
    depth_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    scale: float = 1.0,
    shift: float = 0.0,
    min_points: int = 200,
) -> dict:
    depth_dir = Path(depth_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_files = list(depth_dir.glob("*.npz"))
    mask_files = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
    if not depth_files:
        raise FileNotFoundError(f"No .npz files in {depth_dir}")
    if not mask_files:
        raise FileNotFoundError(f"No .png/.jpg files in {mask_dir}")

    pairs = _pair_files(depth_files, mask_files)
    print(f"[convert] {len(pairs)} (depth, mask) pairs")
    print(f"[convert] depth correction: d' = {scale} * d + {shift}")

    written = 0
    skipped: List[int] = []
    per_frame_counts: List[int] = []
    depth_min, depth_max = np.inf, -np.inf

    for i, (df, mf) in enumerate(pairs):
        z = np.load(df)
        if "depth" not in z or "fov" not in z:
            raise KeyError(
                f"{df}: expected keys 'depth' and 'fov', found {list(z.keys())}. "
                "Was this produced by third_party/mega-sam/UniDepth/scripts/demo_mega-sam.py?"
            )
        depth = z["depth"].astype(np.float32)
        depth = scale * depth + shift
        fov_deg = float(np.asarray(z["fov"]).reshape(-1)[0])

        mask = _load_mask(mf, depth.shape)
        pts = _unproject(depth, fov_deg, mask)
        per_frame_counts.append(len(pts))

        if len(pts) < min_points:
            skipped.append(i)
            print(f"[convert] WARNING frame {i} ({df.name}): only "
                  f"{len(pts)} points after masking; skipping")
            continue

        # Track depth range for diagnostics.
        d_in_mask = depth[mask & (depth > 0)]
        if d_in_mask.size:
            depth_min = min(depth_min, float(d_in_mask.min()))
            depth_max = max(depth_max, float(d_in_mask.max()))

        out_path = output_dir / f"frame_{i:04d}.npy"
        np.save(out_path, pts)
        written += 1
        if i == 0 or (i + 1) % 25 == 0 or i == len(pairs) - 1:
            print(f"[convert] frame {i+1}/{len(pairs)}  pts={len(pts):5d}  "
                  f"depth_range=[{float(d_in_mask.min()):.3f}, {float(d_in_mask.max()):.3f}] m")

    report = {
        "depth_dir": str(depth_dir),
        "mask_dir": str(mask_dir),
        "output_dir": str(output_dir),
        "scale": scale,
        "shift": shift,
        "frames_written": written,
        "frames_skipped": skipped,
        "median_points_per_frame": int(np.median(per_frame_counts)) if per_frame_counts else 0,
        "min_points_per_frame": int(min(per_frame_counts)) if per_frame_counts else 0,
        "max_points_per_frame": int(max(per_frame_counts)) if per_frame_counts else 0,
        "depth_range_meters": [depth_min, depth_max] if np.isfinite(depth_min) else None,
    }
    with open(output_dir / "_conversion_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[convert] wrote {written} / {len(pairs)} frames to {output_dir}")
    return report


def main():
    ap = argparse.ArgumentParser(description="UniDepth .npz + masks -> point-cloud .npy directory")
    ap.add_argument("--depth_dir", required=True,
                    help="Directory of UniDepth .npz files (depth, fov per frame)")
    ap.add_argument("--mask_dir", required=True,
                    help="Directory of per-frame dog masks (PNG/JPG, white=dog)")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write per-frame .npy point clouds")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Linear depth scale α from Step 3 calibration (d' = α*d + β)")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="Linear depth shift β from Step 3 calibration (d' = α*d + β)")
    ap.add_argument("--min_points", type=int, default=200,
                    help="Skip frames whose dog mask yields fewer than this many points")
    args = ap.parse_args()

    convert(
        depth_dir=args.depth_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        shift=args.shift,
        min_points=args.min_points,
    )


if __name__ == "__main__":
    main()
