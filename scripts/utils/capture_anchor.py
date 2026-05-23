#!/usr/bin/env python3
"""Step 2.1 — Capture the Physical Anchor (RGB + metric depth).

Captures a synchronized RGB / depth pair from an Intel RealSense camera
(D415 / D435 / D455 / L515) and writes the two files Step 2 of the PDF
expects:

    <output_dir>/input_anchor.png    8-bit BGR PNG (the RGB anchor I_0)
    <output_dir>/input_depth.png     16-bit PNG, units = millimeters (D_0)

The depth stream is aligned to the color stream so the two PNGs are pixel-
aligned. A short warm-up burst lets auto-exposure stabilize before the
saved frame is grabbed.

Usage:
    python scripts/utils/capture_anchor.py --output_dir data/lego_dog_walk/

Requirements:
    pip install pyrealsense2 opencv-python numpy

For non-RealSense hardware:
- iPhone Pro / iPad Pro with LiDAR: use Record3D, export the depth-aligned
  PNG, then rename to input_depth.png.
- Azure Kinect: use the Azure Kinect SDK's k4aviewer to capture, then
  convert the depth .mkv / .raw to a 16-bit PNG in millimeters.
- Any RGB-only camera: skip this step. Step 3.2 supports a calibration-free
  path (just trust UniDepth's metric output); see calibrate_metric_scale.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def _import_realsense():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError as e:
        raise RuntimeError(
            "pyrealsense2 not installed. Install with:\n"
            "    pip install pyrealsense2\n"
            "If you don't have a RealSense camera, see the script docstring "
            "for alternatives (iPhone LiDAR, Azure Kinect, RGB-only)."
        ) from e


def capture(
    output_dir: str | Path,
    color_resolution=(1280, 720),
    depth_resolution=(640, 480),
    warmup_frames: int = 30,
    fps: int = 30,
    align_depth_to_color: bool = True,
) -> dict:
    rs = _import_realsense()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1],
                         rs.format.z16, fps)
    config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1],
                         rs.format.bgr8, fps)
    profile = pipeline.start(config)

    try:
        # Auto-exposure + IR-emitter warm-up. The first frames after `start()`
        # are typically over-exposed and noisy, so we burn a few before
        # grabbing the keeper frame.
        print(f"[anchor] warming up for {warmup_frames} frames "
              f"(~{warmup_frames / fps:.1f}s) ...")
        for _ in range(warmup_frames):
            pipeline.wait_for_frames()

        align = rs.align(rs.stream.color) if align_depth_to_color else None
        frames = pipeline.wait_for_frames()
        if align is not None:
            frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to grab synchronized color+depth frame.")

        # Recover the depth-to-meters scale from the device. For most
        # RealSense models depth_scale = 0.001 (i.e. raw uint16 is mm), but
        # the L515 reports a different scale, so we compute mm exactly.
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_units_m = float(depth_sensor.get_depth_scale())  # meters per unit
        depth_mm_scale = depth_units_m * 1000.0  # mm per unit

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16

        # Convert raw depth -> millimeters, encoded as uint16 (matches the
        # input_depth.png format that calibrate_metric_scale.py expects).
        depth_mm = (depth_raw.astype(np.float32) * depth_mm_scale).astype(np.uint16)

    finally:
        pipeline.stop()

    anchor_path = output_dir / "input_anchor.png"
    depth_path = output_dir / "input_depth.png"
    cv2.imwrite(str(anchor_path), color)
    cv2.imwrite(str(depth_path), depth_mm)

    H, W = depth_mm.shape
    valid = depth_mm > 0
    median_mm = float(np.median(depth_mm[valid])) if valid.any() else 0.0
    p2_mm = float(np.percentile(depth_mm[valid], 2)) if valid.any() else 0.0
    p98_mm = float(np.percentile(depth_mm[valid], 98)) if valid.any() else 0.0

    meta = {
        "anchor_image": str(anchor_path),
        "depth_image": str(depth_path),
        "color_resolution": list(color_resolution),
        "depth_resolution": list(depth_resolution),
        "depth_aligned_to_color": align_depth_to_color,
        "device_depth_unit_meters": depth_units_m,
        "depth_encoding": "uint16 PNG, units = millimeters",
        "valid_depth_pixels_pct": float(100.0 * valid.mean()),
        "depth_median_mm": median_mm,
        "depth_p2_mm": p2_mm,
        "depth_p98_mm": p98_mm,
        "captured_at_unix": time.time(),
    }
    with open(output_dir / "_anchor_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[anchor] color -> {anchor_path}  ({color.shape[1]}x{color.shape[0]})")
    print(f"[anchor] depth -> {depth_path}  ({W}x{H}, 16-bit mm)")
    print(f"[anchor] valid depth pixels: {meta['valid_depth_pixels_pct']:.1f}%  "
          f"median={median_mm/1000:.3f} m  P2-P98=[{p2_mm/1000:.3f}, {p98_mm/1000:.3f}] m")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Step 2.1 — RealSense RGB+depth anchor capture")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write input_anchor.png and input_depth.png "
                         "(typically data/lego_dog_walk/)")
    ap.add_argument("--color_width", type=int, default=1280)
    ap.add_argument("--color_height", type=int, default=720)
    ap.add_argument("--depth_width", type=int, default=640)
    ap.add_argument("--depth_height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup_frames", type=int, default=30,
                    help="Frames to discard before grabbing the keeper (lets "
                         "auto-exposure settle)")
    ap.add_argument("--no_align", action="store_true",
                    help="Skip depth-to-color alignment (advanced; the saved "
                         "depth PNG will not be pixel-registered to the RGB)")
    args = ap.parse_args()

    capture(
        output_dir=args.output_dir,
        color_resolution=(args.color_width, args.color_height),
        depth_resolution=(args.depth_width, args.depth_height),
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        align_depth_to_color=not args.no_align,
    )


if __name__ == "__main__":
    main()
