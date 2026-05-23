#!/usr/bin/env python3
"""Step 2.1 — Capture the Physical Anchor.

Two input paths are supported, picked at runtime:

A) RealSense capture (default).
   Captures a synchronized RGB / depth pair from an Intel RealSense camera
   (D415 / D435 / D455 / L515) and writes both files Step 2 of the PDF
   expects:
       <output_dir>/input_anchor.png    8-bit BGR PNG (the RGB anchor I_0)
       <output_dir>/input_depth.png     16-bit PNG, mm units            (D_0)
   The depth stream is aligned to the color stream so the two PNGs are
   pixel-aligned. A short warm-up burst lets auto-exposure stabilize.

       python scripts/utils/capture_anchor.py --output_dir data/lego_dog_walk/

B) Pre-existing RGB photo (no depth sensor).
   Pass `--rgb_source path/to/your_photo.{jpg,png,...}` and the script
   normalizes that file (strips alpha, drops to 8-bit, re-encodes as PNG)
   and writes it to <output_dir>/input_anchor.png. No depth file is
   produced; Step 3.2 has a calibration-free fallback for this case (see
   `scripts/reconstruction/calibrate_metric_scale.py`).

       python scripts/utils/capture_anchor.py \\
           --rgb_source path/to/lego_dog.jpg \\
           --output_dir data/lego_dog_walk/

Requirements:
    pip install opencv-python numpy             # both modes
    pip install pyrealsense2                    # only for RealSense mode

For non-RealSense depth hardware:
- iPhone Pro / iPad Pro with LiDAR: use Record3D, export the depth-aligned
  PNG, then rename to input_depth.png.
- Azure Kinect: use the Azure Kinect SDK's k4aviewer to capture, then
  convert the depth .mkv / .raw to a 16-bit PNG in millimeters.
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
            "If you don't have a RealSense camera, pass --rgb_source "
            "path/to/your_photo.jpg to import an existing RGB image instead, "
            "or see the script docstring for other depth-sensor options."
        ) from e


def import_rgb_anchor(
    rgb_source: str | Path,
    output_dir: str | Path,
    max_long_edge: int | None = None,
) -> dict:
    """Normalize an existing RGB photo into <output_dir>/input_anchor.png.

    Useful when you have a phone / DSLR photo of the LEGO dog but no depth
    sensor — the file becomes the I_0 anchor that Step 2.2 (video diffusion)
    and Step 4 (segmentation / inpainting) ingest. No input_depth.png is
    written; Step 3.2's calibration-free path handles that case.
    """
    src = Path(rgb_source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"--rgb_source not found: {src}")

    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(
            f"OpenCV could not decode {src}. Supported formats include "
            f"PNG, JPG, BMP, TIFF, WEBP."
        )

    src_shape = list(img.shape)

    # Strip alpha channel if present.
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 2:
        # Single-channel grayscale -> replicate to 3 channels so downstream
        # tooling (Veo, UniDepth, ObjectClear) sees a BGR image.
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Down-cast >8-bit photos (some smartphones export 16-bit HEIC->PNG).
    if img.dtype != np.uint8:
        max_val = float(img.max()) if img.size else 0.0
        scale = 255.0 / max_val if max_val > 0 else 1.0
        img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    if max_long_edge is not None:
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > max_long_edge:
            new_w = int(round(w * max_long_edge / long_edge))
            new_h = int(round(h * max_long_edge / long_edge))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    anchor_path = output_dir / "input_anchor.png"
    if not cv2.imwrite(str(anchor_path), img):
        raise RuntimeError(f"Failed to write {anchor_path}")

    meta = {
        "mode": "rgb_only",
        "anchor_image": str(anchor_path),
        "depth_image": None,
        "source_path": str(src),
        "source_shape": src_shape,
        "saved_shape": [int(img.shape[0]), int(img.shape[1]), int(img.shape[2])],
        "max_long_edge": max_long_edge,
        "depth_image_note": (
            "No depth sensor; Step 3.2 should be run with the "
            "calibration-free path (trust UniDepth's metric output)."
        ),
        "captured_at_unix": time.time(),
    }
    with open(output_dir / "_anchor_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[anchor] mode  : rgb_only")
    print(f"[anchor] source: {src}  ({src_shape[1]}x{src_shape[0]})")
    print(f"[anchor] color : {anchor_path}  "
          f"({img.shape[1]}x{img.shape[0]}, 8-bit BGR PNG)")
    print(f"[anchor] depth : not written (no sensor). Step 3.2 will run "
          f"without --real_depth; see calibrate_metric_scale.py.")
    return meta


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
    ap = argparse.ArgumentParser(description="Step 2.1 — Physical anchor capture")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write input_anchor.png (and input_depth.png in "
                         "RealSense mode). Typically data/lego_dog_walk/.")
    ap.add_argument("--rgb_source", default=None,
                    help="Path to an existing RGB photo to import as input_anchor.png. "
                         "When provided, RealSense capture is skipped (rgb-only mode).")
    ap.add_argument("--max_long_edge", type=int, default=None,
                    help="(rgb-only mode) Resize so the longer edge is at most this "
                         "many pixels (default: keep native resolution).")
    ap.add_argument("--color_width", type=int, default=1280)
    ap.add_argument("--color_height", type=int, default=720)
    ap.add_argument("--depth_width", type=int, default=640)
    ap.add_argument("--depth_height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup_frames", type=int, default=30,
                    help="Frames to discard before grabbing the keeper (lets "
                         "auto-exposure settle).")
    ap.add_argument("--no_align", action="store_true",
                    help="(RealSense mode) Skip depth-to-color alignment.")
    args = ap.parse_args()

    if args.rgb_source is not None:
        import_rgb_anchor(
            rgb_source=args.rgb_source,
            output_dir=args.output_dir,
            max_long_edge=args.max_long_edge,
        )
        return

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
