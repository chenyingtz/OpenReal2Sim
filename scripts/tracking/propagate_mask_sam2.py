#!/usr/bin/env python3
"""Propagate a single seed mask across all video frames via SAM2 video.

When propagate_mask_by_depth.py fails — because the dog shares a depth
band with another object, or it walks in/out of the seed depth window —
switch to this script. It uses Meta's SAM2 video object segmenter, the
same submodule the repo's hand_extraction module already imports
(third_party/Grounded-SAM-2). SAM2 tracks the dog by its visual
appearance + memory of past frames, not just depth, so it is robust to
occlusion, motion, and inter-frame drift.

Outputs match the schema that
`scripts/tracking/unidepth_to_pointclouds.py` expects:
    <output_dir>/<stem>.png      one mask per frame, white = dog
    <output_dir>/_propagation_report.json  per-frame size stats

Usage:
    python scripts/tracking/propagate_mask_sam2.py \
        --seed_mask  data/lego_dog_walk/robot_mask.png \
        --frame_dir  data/lego_dog_walk/generated_frames/ \
        --output_dir data/lego_dog_walk/masks_sam2/

First-run setup:
    cd third_party/Grounded-SAM-2 && pip install -e .
    cd checkpoints && bash download_ckpts.sh   # ~2 GB for hiera_large
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SAM2_SRC = REPO_ROOT / "third_party" / "Grounded-SAM-2"


def _ensure_sam2_on_path():
    if not SAM2_SRC.exists():
        raise RuntimeError(
            f"Grounded-SAM-2 not found at {SAM2_SRC}. Run:\n"
            f"    git submodule update --init --recursive\n"
            f"then install:\n"
            f"    cd {SAM2_SRC} && pip install -e ."
        )
    if str(SAM2_SRC) not in sys.path:
        sys.path.insert(0, str(SAM2_SRC))


def _default_checkpoint() -> Path:
    return SAM2_SRC / "checkpoints" / "sam2.1_hiera_large.pt"


def _convert_frames_to_jpeg(frame_dir: Path, jpeg_dir: Path,
                            extensions=(".png", ".jpg", ".jpeg")
                            ) -> List[Tuple[int, str]]:
    """SAM2's init_state requires JPEG files named 0.jpg / 1.jpg / 5-digit-padded
    integers — basically `int(os.path.splitext(name)[0])` must succeed.
    The user's frames are typically `frame_0000.png`, so we re-encode them
    to `00000.jpg` here and remember the original stem so we can write
    output masks back with the original naming.
    """
    frame_dir = Path(frame_dir)
    jpeg_dir = Path(jpeg_dir)
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    src_files = sorted(
        [p for p in frame_dir.iterdir() if p.suffix.lower() in extensions],
        key=lambda p: (len(p.stem), p.stem),
    )
    if not src_files:
        raise FileNotFoundError(f"No frame files in {frame_dir}")

    mapping: List[Tuple[int, str]] = []
    for i, src in enumerate(src_files):
        img = Image.open(src).convert("RGB")
        img.save(jpeg_dir / f"{i:05d}.jpg", "JPEG", quality=95)
        mapping.append((i, src.stem))
    return mapping


def _load_seed_mask(seed_mask_path: Path, target_hw: Tuple[int, int],
                    invert: bool = False) -> np.ndarray:
    img = np.array(Image.open(seed_mask_path).convert("L"))
    H, W = target_hw
    if img.shape != (H, W):
        img = np.array(Image.fromarray(img).resize((W, H), Image.NEAREST))
    mask = img > 127
    if invert:
        mask = ~mask
    if int(mask.sum()) < 50:
        raise ValueError(
            f"Seed mask is essentially empty after thresholding "
            f"({int(mask.sum())} px). Check the file, and if your mask "
            f"has white=background pass --invert_seed_mask."
        )
    return mask


def propagate(
    seed_mask_path: Path,
    frame_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    seed_frame_idx: int = 0,
    invert_seed_mask: bool = False,
    device: Optional[str] = None,
    offload_state_to_cpu: bool = False,
    offload_video_to_cpu: bool = False,
    mask_threshold: float = 0.0,
) -> dict:
    _ensure_sam2_on_path()
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_mask_path = Path(seed_mask_path)
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint}\n"
            f"Download with:\n"
            f"    cd {SAM2_SRC}/checkpoints && bash download_ckpts.sh"
        )

    print(f"[sam2] device       : {device}")
    print(f"[sam2] checkpoint   : {checkpoint}")
    print(f"[sam2] model_cfg    : {model_cfg}")

    with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmpdir:
        jpeg_dir = Path(tmpdir)
        print(f"[sam2] converting frames to JPEG -> {jpeg_dir} ...")
        t0 = time.time()
        mapping = _convert_frames_to_jpeg(frame_dir, jpeg_dir)
        print(f"[sam2] {len(mapping)} frames converted in {time.time()-t0:.1f}s")

        first = np.array(Image.open(jpeg_dir / "00000.jpg"))
        H, W = first.shape[:2]
        seed_mask = _load_seed_mask(seed_mask_path, (H, W),
                                    invert=invert_seed_mask)
        print(f"[sam2] seed mask    : {int(seed_mask.sum())} px "
              f"({100*seed_mask.mean():.1f}% of {W}x{H})")

        print(f"[sam2] loading SAM2 video predictor ...")
        t0 = time.time()
        predictor = build_sam2_video_predictor(
            config_file=model_cfg,
            ckpt_path=str(checkpoint),
            device=device,
        )
        inference_state = predictor.init_state(
            video_path=str(jpeg_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        print(f"[sam2] init_state   : {time.time()-t0:.1f}s")

        # Seed at the user-specified frame (default 0).
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=seed_frame_idx,
            obj_id=1,
            mask=seed_mask,
        )

        print(f"[sam2] propagating  ...")
        per_frame_size = {}
        t0 = time.time()

        # Forward pass: seed_frame_idx -> end.
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            mask = (mask_logits[0, 0] > mask_threshold).cpu().numpy()
            stem = mapping[frame_idx][1]
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                output_dir / f"{stem}.png"
            )
            per_frame_size[frame_idx] = int(mask.sum())
            if frame_idx == seed_frame_idx or (frame_idx + 1) % 25 == 0 \
                    or frame_idx == len(mapping) - 1:
                print(f"[sam2] fwd frame {frame_idx+1}/{len(mapping)}  "
                      f"size={int(mask.sum()):6d}")

        # Reverse pass: seed_frame_idx -> 0 (only if seed isn't frame 0).
        if seed_frame_idx > 0:
            try:
                for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
                    inference_state, reverse=True
                ):
                    mask = (mask_logits[0, 0] > mask_threshold).cpu().numpy()
                    stem = mapping[frame_idx][1]
                    Image.fromarray((mask * 255).astype(np.uint8)).save(
                        output_dir / f"{stem}.png"
                    )
                    per_frame_size[frame_idx] = int(mask.sum())
            except TypeError:
                print(f"[sam2] WARN: this SAM2 build does not accept "
                      f"reverse=True; frames before seed_frame_idx="
                      f"{seed_frame_idx} will have no mask written.")

        print(f"[sam2] propagate    : {time.time()-t0:.1f}s "
              f"({len(per_frame_size)/(time.time()-t0):.1f} fps)")

    sizes = sorted(per_frame_size.values())
    report = {
        "seed_mask": str(seed_mask_path),
        "frame_dir": str(frame_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "model_cfg": model_cfg,
        "seed_frame_idx": seed_frame_idx,
        "invert_seed_mask": invert_seed_mask,
        "frames": len(sizes),
        "size_median": int(np.median(sizes)) if sizes else 0,
        "size_min": int(sizes[0]) if sizes else 0,
        "size_max": int(sizes[-1]) if sizes else 0,
        "frames_with_empty_mask": sum(1 for s in sizes if s == 0),
    }
    with open(output_dir / "_propagation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[sam2] median dog px / frame : {report['size_median']}")
    print(f"[sam2] empty masks           : {report['frames_with_empty_mask']}")
    print(f"[sam2] wrote {len(sizes)} masks to {output_dir}")
    return report


def main():
    ap = argparse.ArgumentParser(description="SAM2-based video mask propagation")
    ap.add_argument("--seed_mask", required=True,
                    help="Path to the single seed mask (white=dog)")
    ap.add_argument("--frame_dir", required=True,
                    help="Directory of RGB video frames (PNG / JPG)")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write per-frame masks (named to match frame stems)")
    ap.add_argument("--checkpoint", default=None,
                    help=f"SAM2 checkpoint path. Default: {_default_checkpoint()}")
    ap.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                    help="SAM2 config file (relative to sam2/configs)")
    ap.add_argument("--seed_frame_idx", type=int, default=0,
                    help="Index of the video frame the seed mask corresponds to "
                         "(0-indexed; default 0 = first frame)")
    ap.add_argument("--invert_seed_mask", action="store_true",
                    help="Flip the seed mask (white=background -> white=dog)")
    ap.add_argument("--device", default=None,
                    help="'cuda' / 'cpu' (auto-detects if omitted)")
    ap.add_argument("--offload_state_to_cpu", action="store_true",
                    help="Reduce VRAM at ~10%% speed cost (useful for long videos)")
    ap.add_argument("--offload_video_to_cpu", action="store_true",
                    help="Reduce VRAM further by keeping raw video frames on CPU")
    ap.add_argument("--mask_threshold", type=float, default=0.0,
                    help="Mask logit threshold (default 0; raise for stricter masks)")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else _default_checkpoint()

    propagate(
        seed_mask_path=args.seed_mask,
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        checkpoint=checkpoint,
        model_cfg=args.model_cfg,
        seed_frame_idx=args.seed_frame_idx,
        invert_seed_mask=args.invert_seed_mask,
        device=args.device,
        offload_state_to_cpu=args.offload_state_to_cpu,
        offload_video_to_cpu=args.offload_video_to_cpu,
        mask_threshold=args.mask_threshold,
    )


if __name__ == "__main__":
    main()
