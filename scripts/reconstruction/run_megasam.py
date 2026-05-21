#!/usr/bin/env python3
"""Step 3.1 — Generate Temporal Relative Depths.

Run UniDepth (the depth backbone bundled inside MegaSaM at
`third_party/mega-sam/UniDepth`) on every frame of the generated video and
save per-frame depth + FOV as a .npz, matching the format expected by
`scripts/reconstruction/calibrate_metric_scale.py` and
`scripts/tracking/unidepth_to_pointclouds.py`.

Matches the PDF CLI:

    python scripts/reconstruction/run_megasam.py \
        --frame_dir  data/lego_dog_walk/generated_frames/ \
        --output_dir data/lego_dog_walk/relative_depths/

Output: <output_dir>/<frame_stem>.npz with keys
    depth : (H, W) float32 meters (UniDepth's raw metric prediction)
    fov   : float, horizontal field of view in degrees
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIDEPTH_SRC = REPO_ROOT / "third_party" / "mega-sam" / "UniDepth"


def _ensure_unidepth_on_path():
    if not UNIDEPTH_SRC.exists():
        raise RuntimeError(
            f"UniDepth source not found at {UNIDEPTH_SRC}. Did you `git "
            f"submodule update --init --recursive` to pull mega-sam?"
        )
    if str(UNIDEPTH_SRC) not in sys.path:
        sys.path.insert(0, str(UNIDEPTH_SRC))


def _load_model(device: str):
    try:
        import torch  # noqa: F401
        from unidepth.models import UniDepthV2
    except ImportError as e:
        raise RuntimeError(
            "UniDepth dependencies missing. Activate the openreal2sim conda env "
            "and ensure torch + unidepth are installed."
        ) from e
    import torch
    model = UniDepthV2.from_pretrained(
        "lpiccinelli/unidepth-v2-vitl14",
        revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4",
    )
    model = model.to(torch.device(device))
    model.eval()
    return model, torch


def _resize_long_dim(rgb: np.ndarray, long_dim: int):
    import cv2
    H, W = rgb.shape[:2]
    if W >= H:
        new_w, new_h = long_dim, int(round(long_dim * H / W))
    else:
        new_w, new_h = int(round(long_dim * W / H)), long_dim
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _fov_from_intrinsics(K: np.ndarray, width: int) -> float:
    fx = float(K[0, 0])
    return float(np.rad2deg(2.0 * np.arctan(width / (2.0 * fx))))


def run(
    frame_dir: Path,
    output_dir: Path,
    long_dim: int = 518,
    device: str | None = None,
    extensions=(".png", ".jpg", ".jpeg"),
):
    _ensure_unidepth_on_path()
    import cv2  # noqa: F401  (imported via _resize_long_dim too; keep here for the
                # early-failure error message)
    model, torch_mod = _load_model(device or ("cuda" if _have_cuda() else "cpu"))

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in extensions])
    if not frames:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")

    print(f"[megasam] frames : {len(frames)} in {frame_dir}")
    print(f"[megasam] device : {next(model.parameters()).device}")

    for i, frame_path in enumerate(frames):
        rgb = cv2.imread(str(frame_path))
        if rgb is None:
            print(f"[megasam] WARN: failed to read {frame_path}, skipping")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = _resize_long_dim(rgb, long_dim)
        rgb_torch = torch_mod.from_numpy(rgb).permute(2, 0, 1)

        with torch_mod.no_grad():
            preds = model.infer(rgb_torch)

        depth = preds["depth"][0, 0].cpu().numpy().astype(np.float32)
        K = preds["intrinsics"][0].cpu().numpy()
        fov = _fov_from_intrinsics(K, depth.shape[1])

        out_path = output_dir / f"{frame_path.stem}.npz"
        np.savez(out_path, depth=depth, fov=fov)

        if i == 0 or (i + 1) % 25 == 0 or i == len(frames) - 1:
            print(f"[megasam] frame {i+1}/{len(frames)}  "
                  f"shape={depth.shape}  fov={fov:.2f}°  "
                  f"depth_range=[{float(depth.min()):.3f}, {float(depth.max()):.3f}] m  -> {out_path.name}")

    print(f"[megasam] done — wrote {len(frames)} depth files to {output_dir}")


def _have_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser(description="Step 3.1 — UniDepth-based relative depth extraction")
    ap.add_argument("--frame_dir", required=True,
                    help="Directory of RGB frames from Step 2 (PNG/JPG)")
    ap.add_argument("--output_dir", required=True,
                    help="Where to save per-frame .npz depth files")
    ap.add_argument("--long_dim", type=int, default=518,
                    help="Long-edge resize before UniDepth inference (matches "
                         "mega-sam's default of 518)")
    ap.add_argument("--device", default=None,
                    help="'cuda' / 'cpu'. Auto-detects if omitted.")
    args = ap.parse_args()

    run(
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        long_dim=args.long_dim,
        device=args.device,
    )


if __name__ == "__main__":
    main()
