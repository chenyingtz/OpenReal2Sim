#!/usr/bin/env python3
"""Step 2.3 — Slice a generated video into per-frame image files.

Matches the PDF CLI:

    python scripts/utils/video_to_frames.py \
        --video_path path/to/generated.mp4 \
        --output_dir data/lego_dog_walk/generated_frames/

Frames are written as zero-padded PNGs (frame_0000.png, frame_0001.png, ...)
so the natural-sort logic in scripts/tracking/run_foundationpose.py and
scripts/reconstruction/run_megasam.py pairs them correctly.

Useful secondary flags:
    --stride 2          emit every 2nd frame (halve fps)
    --max_frames 300    cap the number of frames written
    --start_frame 30    skip the first N source frames
    --target_height 720 resize so the short edge is 720, preserving aspect ratio
    --output_format jpg JPEG instead of PNG (smaller, lossy)
    --naming "f_{:05d}" custom filename pattern (default: "frame_{:04d}")
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    output_format: str = "png",
    stride: int = 1,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    target_height: Optional[int] = None,
    naming: str = "frame_{:04d}",
) -> dict:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if output_format.lower() not in ("png", "jpg", "jpeg"):
        raise ValueError(f"Unsupported output_format: {output_format}")
    ext = "jpg" if output_format.lower() in ("jpg", "jpeg") else "png"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open {video_path}. The file may be corrupt, in "
            f"an unsupported container, or missing a codec. Try re-encoding "
            f"with ffmpeg first:  ffmpeg -i {video_path.name} -c:v libx264 "
            f"-pix_fmt yuv420p re_encoded.mp4"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"[v2f] video : {video_path}")
    print(f"[v2f] source: {src_w}x{src_h}  fps={fps:.2f}  frames={total}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written = 0
    read_idx = start_frame
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (read_idx - start_frame) % stride == 0:
            if target_height is not None and frame.shape[0] != target_height:
                h, w = frame.shape[:2]
                new_w = int(round(w * target_height / h))
                interp = cv2.INTER_AREA if h > target_height else cv2.INTER_CUBIC
                frame = cv2.resize(frame, (new_w, target_height), interpolation=interp)

            out_name = naming.format(written) + "." + ext
            cv2.imwrite(str(output_dir / out_name), frame)
            written += 1
            if max_frames is not None and written >= max_frames:
                break
            if written == 1 or written % 50 == 0:
                print(f"[v2f] wrote frame {written}  (source idx {read_idx})")
        read_idx += 1
    cap.release()

    elapsed = time.time() - t0
    print(f"[v2f] wrote {written} frames in {elapsed:.2f}s -> {output_dir}")

    meta = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "source_fps": fps,
        "source_frame_count": total,
        "source_resolution": [src_w, src_h],
        "stride": stride,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "target_height": target_height,
        "output_format": ext,
        "frames_written": written,
        "effective_fps": (fps / stride) if fps > 0 else None,
    }
    with open(output_dir / "_extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Step 2.3 — Extract frames from a video")
    ap.add_argument("--video_path", required=True, help="Input video file (.mp4 / .mov / .avi / ...)")
    ap.add_argument("--output_dir", required=True, help="Directory to write per-frame images")
    ap.add_argument("--output_format", default="png", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--stride", type=int, default=1,
                    help="Emit every Nth source frame (default 1 = all frames)")
    ap.add_argument("--start_frame", type=int, default=0,
                    help="Skip the first N source frames before extracting")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Maximum frames to write (default: all)")
    ap.add_argument("--target_height", type=int, default=None,
                    help="Resize each frame to this height (preserves aspect ratio)")
    ap.add_argument("--naming", default="frame_{:04d}",
                    help="Python format string for output stem (default 'frame_{:04d}')")
    args = ap.parse_args()

    extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        output_format=args.output_format,
        stride=args.stride,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        target_height=args.target_height,
        naming=args.naming,
    )


if __name__ == "__main__":
    main()
