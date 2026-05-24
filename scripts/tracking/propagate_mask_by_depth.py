#!/usr/bin/env python3
"""Propagate a single foreground mask to all UniDepth frames via depth banding.

When you have one anchor-frame dog mask (typical Step-4 output) and 286
UniDepth depth maps from the generated video, this script auto-produces a
per-frame mask directory that `scripts/tracking/unidepth_to_pointclouds.py`
can consume.

Algorithm:
  1. Load the seed mask and the depth at the seed frame.
  2. Compute the dog's depth band [d_min, d_max] from the masked pixels.
  3. For every UniDepth frame: threshold = (d_min - tol) < depth < (d_max + tol),
     morphological clean-up, keep only the largest connected component.

Works when the dog is depth-separable from the floor and walls (the typical
robot-on-tile case). Falls apart when another object shares the dog's depth
band — in that case run SAM2 video segmentation instead.

Usage:
    python scripts/tracking/propagate_mask_by_depth.py \
        --seed_mask  data/lego_dog_walk/robot_mask.png \
        --depth_dir  data/lego_dog_walk/UniDepth_outputs/ \
        --output_dir data/lego_dog_walk/masks_propagated/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening, label


def _natural_sort(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=lambda p: (len(p.stem), p.stem))


def _load_mask(path: Path, target_hw):
    img = np.array(Image.open(path).convert("L"))
    H, W = target_hw
    if img.shape != (H, W):
        img = np.array(Image.fromarray(img).resize((W, H), Image.NEAREST))
    return img > 127


def _seed_depth_path(depth_files: List[Path], seed_frame: Optional[str]) -> Path:
    """Pick the depth file matching the user's --seed_frame, else use frame 0."""
    if seed_frame is None:
        return depth_files[0]
    p = Path(seed_frame)
    # If user gave just a filename like "frame_000.npz", look it up in the dir.
    if not p.is_absolute() and not p.exists():
        for d in depth_files:
            if d.name == p.name or d.stem == p.stem:
                return d
    return p


def propagate(
    seed_mask_path: Path,
    depth_dir: Path,
    output_dir: Path,
    seed_frame: Optional[str] = None,
    depth_tolerance: float = 0.05,
    min_pixels: int = 100,
    open_iter: int = 1,
    close_iter: int = 2,
    invert_seed_mask: bool = False,
) -> dict:
    seed_mask_path = Path(seed_mask_path)
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_files = _natural_sort(list(depth_dir.glob("*.npz")))
    if not depth_files:
        raise FileNotFoundError(f"No .npz files in {depth_dir}")

    seed_depth_path = _seed_depth_path(depth_files, seed_frame)
    with np.load(seed_depth_path) as z:
        seed_depth = np.asarray(z["depth"], dtype=np.float32)
    seed_mask = _load_mask(seed_mask_path, seed_depth.shape)
    if invert_seed_mask:
        seed_mask = ~seed_mask
        print("[propagate] --invert_seed_mask: flipped seed mask polarity")

    seed_coverage = float(seed_mask.mean())
    print(f"[propagate] seed mask coverage: {seed_coverage*100:.1f}% of image")
    # A typical foreground mask covers <30% of the frame. Above ~40% it is
    # almost certainly the background mask (the user's robot_mask.png had
    # white = floor / wall). Warn loudly so the user re-runs with
    # --invert_seed_mask instead of debugging the propagated masks.
    if seed_coverage > 0.4 and not invert_seed_mask:
        print(f"[propagate] WARNING: the seed mask covers {seed_coverage*100:.1f}% "
              f"of the image — this is unusual for a foreground (dog) mask and "
              f"strongly suggests the polarity is inverted (white = background). "
              f"If the propagated frames look inverted, re-run with "
              f"--invert_seed_mask.")
    elif seed_coverage < 0.005:
        print(f"[propagate] WARNING: the seed mask covers only "
              f"{seed_coverage*100:.2f}% of the image — too small to compute a "
              f"reliable depth band. Use a more generous seed mask.")

    valid = (seed_depth > 0) & seed_mask
    if int(valid.sum()) == 0:
        raise ValueError(
            f"Seed mask + seed depth ({seed_depth_path.name}) overlap is empty. "
            "Either the mask is the wrong frame or the mask polarity is "
            "inverted (white must mark the dog; my loader uses pixel > 127). "
            "Try re-running with --invert_seed_mask."
        )
    d_values = seed_depth[valid]
    d_min = float(np.percentile(d_values, 2))   # robust min, ignores depth-edge bleed
    d_max = float(np.percentile(d_values, 98))  # robust max
    band = (d_min - depth_tolerance, d_max + depth_tolerance)

    print(f"[propagate] seed depth file : {seed_depth_path.name}")
    print(f"[propagate] dog depth (P2-P98): [{d_min:.3f}, {d_max:.3f}] m")
    print(f"[propagate] band w/ tol      : [{band[0]:.3f}, {band[1]:.3f}] m")
    print(f"[propagate] frames           : {len(depth_files)} in {depth_dir}")

    sizes: List[int] = []
    warn_empty = 0
    warn_small = 0

    for i, df in enumerate(depth_files):
        with np.load(df) as z:
            depth = np.asarray(z["depth"], dtype=np.float32)
        candidate = (depth > band[0]) & (depth < band[1])
        if open_iter > 0:
            candidate = binary_opening(candidate, iterations=open_iter)
        if close_iter > 0:
            candidate = binary_closing(candidate, iterations=close_iter)

        labels, n = label(candidate)
        if n == 0:
            warn_empty += 1
            mask = np.zeros_like(candidate, dtype=bool)
        else:
            comp_sizes = np.bincount(labels.ravel())
            comp_sizes[0] = 0  # background label
            largest = int(comp_sizes.argmax())
            mask = labels == largest

        size = int(mask.sum())
        sizes.append(size)
        if size < min_pixels:
            warn_small += 1

        Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / f"{df.stem}.png")
        if i == 0 or (i + 1) % 25 == 0 or i == len(depth_files) - 1:
            print(f"[propagate] frame {i+1}/{len(depth_files)}  size={size:6d}  "
                  f"depth_range=[{float(depth[mask].min()) if size else 0:.3f}, "
                  f"{float(depth[mask].max()) if size else 0:.3f}] m")

    report = {
        "seed_mask": str(seed_mask_path),
        "seed_depth": str(seed_depth_path),
        "depth_band_meters": list(band),
        "p2_p98_meters": [d_min, d_max],
        "depth_tolerance": depth_tolerance,
        "frames": len(depth_files),
        "frames_with_empty_mask": warn_empty,
        "frames_below_min_pixels": warn_small,
        "size_median": int(np.median(sizes)) if sizes else 0,
        "size_min": int(min(sizes)) if sizes else 0,
        "size_max": int(max(sizes)) if sizes else 0,
    }
    with open(output_dir / "_propagation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[propagate] median dog pixels per frame: {report['size_median']}")
    print(f"[propagate] empty masks                : {warn_empty}")
    print(f"[propagate] masks below min_pixels     : {warn_small}")
    print(f"[propagate] wrote {len(depth_files)} masks to {output_dir}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Propagate a single foreground mask "
                                             "to a video via UniDepth depth banding.")
    ap.add_argument("--seed_mask", required=True,
                    help="Path to the single anchor-frame dog mask (PNG, white=dog)")
    ap.add_argument("--depth_dir", required=True,
                    help="Directory of UniDepth .npz files")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write per-frame masks (PNG, white=dog, stems "
                         "matching the depth filenames)")
    ap.add_argument("--seed_frame", default=None,
                    help="Filename of the depth that corresponds to the seed mask "
                         "(e.g. frame_000.npz). Defaults to the first depth file.")
    ap.add_argument("--depth_tolerance", type=float, default=0.05,
                    help="Meters of slack around the seed depth band (default 5 cm)")
    ap.add_argument("--min_pixels", type=int, default=100,
                    help="Warn if a frame's mask is below this many pixels")
    ap.add_argument("--open_iter", type=int, default=1,
                    help="Binary-opening iterations (removes salt noise)")
    ap.add_argument("--close_iter", type=int, default=2,
                    help="Binary-closing iterations (fills small holes)")
    ap.add_argument("--invert_seed_mask", action="store_true",
                    help="Flip the seed mask before computing the depth band. "
                         "Use when your seed mask has white=background, "
                         "black=dog (the opposite of the script's convention).")
    args = ap.parse_args()

    propagate(
        seed_mask_path=args.seed_mask,
        depth_dir=args.depth_dir,
        output_dir=args.output_dir,
        seed_frame=args.seed_frame,
        depth_tolerance=args.depth_tolerance,
        min_pixels=args.min_pixels,
        open_iter=args.open_iter,
        close_iter=args.close_iter,
        invert_seed_mask=args.invert_seed_mask,
    )


if __name__ == "__main__":
    main()
