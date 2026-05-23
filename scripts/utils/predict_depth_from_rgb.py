#!/usr/bin/env python3
"""Predict a single-frame depth map from an RGB anchor image.

Produces `input_depth.png` (16-bit, units = mm) for users who don't have a
hardware depth sensor but want to run Step 3.2's
`calibrate_metric_scale.py` end-to-end. Two backends:

  --method unidepth                   (default; uses third_party/mega-sam/UniDepth)
  --method depth_anything_v2          (Hugging Face: requires `pip install transformers`)

CIRCULARITY WARNING ─ if you pass UniDepth's output here as --real_depth and
also pass UniDepth's per-frame outputs as --gen_depth to
calibrate_metric_scale.py, the Huber fit will trivially recover α=1, β=0
because both inputs come from the same predictor. The 'calibration' is a
no-op in that case — equivalent to skipping Step 3.2.

For non-circular calibration:
  - Use a DIFFERENT predictor here than the one that produced --gen_depth
    (mix UniDepth ↔ DepthAnything-V2-Metric).
  - Or measure one physical dimension of the robot and constrain α from
    that single observation (planned: --known_dimension_meters flag on
    calibrate_metric_scale.py).

Usage:
    python3 scripts/utils/predict_depth_from_rgb.py \\
        --rgb_source   data/lego_dog_walk/input_anchor.png \\
        --output_depth data/lego_dog_walk/input_depth.png \\
        --method       unidepth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIDEPTH_SRC = REPO_ROOT / "third_party" / "mega-sam" / "UniDepth"


# ───────────────────────── Common helpers ─────────────────────────

def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(
            f"OpenCV could not decode {path}. Supported formats include "
            f"PNG, JPG, BMP, TIFF, WEBP."
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_long_edge(rgb: np.ndarray, long_dim: int) -> tuple:
    H, W = rgb.shape[:2]
    if W >= H:
        new_w, new_h = long_dim, int(round(long_dim * H / W))
    else:
        new_w, new_h = int(round(long_dim * W / H)), long_dim
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), (new_h, new_w)


def _save_outputs(depth_m: np.ndarray, fov_deg: float,
                  output_depth: Path, source: Path,
                  source_shape, model_input_shape, method: str) -> dict:
    """Write the 16-bit mm PNG (for calibrate_metric_scale.py) and a sidecar
    .npz (for any tool that wants the float-metric array + FOV)."""
    output_depth.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(output_depth), depth_mm):
        raise RuntimeError(f"Failed to write {output_depth}")

    npz_path = output_depth.with_suffix(".npz")
    np.savez(str(npz_path), depth=depth_m.astype(np.float32), fov=np.float32(fov_deg))

    valid = depth_m > 0
    stats = {
        "method": method,
        "depth_path": str(output_depth),
        "npz_path": str(npz_path),
        "source": str(source),
        "source_shape": list(source_shape),
        "model_input_shape": list(model_input_shape),
        "fov_degrees": float(fov_deg),
        "depth_units": "meters (float32 npz) / millimeters (uint16 PNG)",
        "valid_pct": float(100.0 * valid.mean()),
        "depth_median_m": float(np.median(depth_m[valid])) if valid.any() else 0.0,
        "depth_p2_m": float(np.percentile(depth_m[valid], 2)) if valid.any() else 0.0,
        "depth_p98_m": float(np.percentile(depth_m[valid], 98)) if valid.any() else 0.0,
        "circularity_warning": (
            "If --gen_depth in calibrate_metric_scale.py was produced by the "
            "same predictor, the Huber fit will recover α=1, β=0 (no-op). "
            "Use a different predictor here for meaningful calibration."
        ),
        "predicted_at_unix": time.time(),
    }
    meta_path = output_depth.parent / "_anchor_depth_meta.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _print_report(stats: dict, source_resized_to):
    H, W = stats["source_shape"]
    nH, nW = stats["model_input_shape"]
    print(f"[depth] method  : {stats['method']}")
    print(f"[depth] source  : {stats['source']}  ({W}x{H} -> {nW}x{nH})")
    print(f"[depth] fov     : {stats['fov_degrees']:.2f}°")
    print(f"[depth] PNG     : {stats['depth_path']}  (16-bit mm)")
    print(f"[depth] NPZ     : {stats['npz_path']}")
    print(f"[depth] stats   : valid={stats['valid_pct']:.1f}%  "
          f"median={stats['depth_median_m']:.3f}m  "
          f"P2-P98=[{stats['depth_p2_m']:.3f}, {stats['depth_p98_m']:.3f}]m")
    print()
    print("WARNING - circularity:")
    print("  If --gen_depth in calibrate_metric_scale.py was produced by the")
    print("  SAME predictor that produced this PNG, the calibration is a no-op")
    print("  (alpha=1, beta=0). For a meaningful fit, use a DIFFERENT predictor")
    print("  for one of the two sides — e.g. UniDepth here + DepthAnything-V2")
    print("  metric for the per-frame --gen_depth, or vice versa.")


# ───────────────────────── UniDepth backend ─────────────────────────

def _ensure_unidepth_on_path():
    if not UNIDEPTH_SRC.exists():
        raise RuntimeError(
            f"UniDepth source not found at {UNIDEPTH_SRC}. Run "
            f"`git submodule update --init --recursive` first."
        )
    if str(UNIDEPTH_SRC) not in sys.path:
        sys.path.insert(0, str(UNIDEPTH_SRC))


def predict_unidepth(rgb_source: Path, output_depth: Path,
                     long_dim: int, device: str) -> dict:
    _ensure_unidepth_on_path()
    try:
        import torch
        from unidepth.models import UniDepthV2
    except ImportError as e:
        raise RuntimeError(
            "UniDepth dependencies missing. Activate the openreal2sim conda "
            "env (which has torch + unidepth installed for run_megasam.py)."
        ) from e

    print(f"[depth] loading UniDepthV2 on {device} ...")
    model = UniDepthV2.from_pretrained(
        "lpiccinelli/unidepth-v2-vitl14",
        revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4",
    )
    model = model.to(torch.device(device)).eval()

    rgb = _load_rgb(rgb_source)
    H, W = rgb.shape[:2]
    rgb_resized, model_input_hw = _resize_long_edge(rgb, long_dim)
    rgb_torch = torch.from_numpy(rgb_resized).permute(2, 0, 1)

    with torch.no_grad():
        preds = model.infer(rgb_torch)

    depth_m = preds["depth"][0, 0].cpu().numpy().astype(np.float32)  # meters
    K = preds["intrinsics"][0].cpu().numpy()
    fx = float(K[0, 0])
    fov_deg = float(np.rad2deg(2.0 * np.arctan(depth_m.shape[1] / (2.0 * fx))))
    return _save_outputs(depth_m, fov_deg, output_depth, rgb_source,
                         (H, W), model_input_hw, method="unidepth")


# ───────────────────────── DepthAnything-V2 backend ─────────────────────────

def predict_depth_anything_v2(rgb_source: Path, output_depth: Path,
                              long_dim: int, device: str,
                              hf_model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
                              ) -> dict:
    """Use Hugging Face's DepthAnything-V2 Metric checkpoint. Truly different
    model family from UniDepth, so it's the right choice for cross-predictor
    calibration."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as e:
        raise RuntimeError(
            "transformers not installed. Install with:\n"
            "    pip install transformers accelerate safetensors"
        ) from e

    print(f"[depth] loading {hf_model} on {device} ...")
    processor = AutoImageProcessor.from_pretrained(hf_model)
    model = AutoModelForDepthEstimation.from_pretrained(hf_model).to(device).eval()

    rgb = _load_rgb(rgb_source)
    H, W = rgb.shape[:2]
    rgb_resized, model_input_hw = _resize_long_edge(rgb, long_dim)
    inputs = processor(images=rgb_resized, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # `predicted_depth` shape: (1, H, W). DepthAnything-V2-Metric outputs
    # metric depth in meters for the indoor model.
    depth_m = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
    # DepthAnything doesn't return intrinsics; use a 60-deg horizontal FOV
    # default (typical phone camera) so the NPZ has a usable fov field.
    fov_deg = 60.0
    return _save_outputs(depth_m, fov_deg, output_depth, rgb_source,
                         (H, W), model_input_hw, method="depth_anything_v2")


# ───────────────────────── Main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Predict input_depth.png from an RGB anchor")
    ap.add_argument("--rgb_source", required=True,
                    help="Path to the RGB anchor image (e.g. input_anchor.png)")
    ap.add_argument("--output_depth", required=True,
                    help="Where to write the 16-bit mm depth PNG")
    ap.add_argument("--method", default="unidepth",
                    choices=["unidepth", "depth_anything_v2"],
                    help="Depth predictor backend. Use a different backend "
                         "than the one that produced your per-frame --gen_depth "
                         "to avoid circular calibration.")
    ap.add_argument("--long_dim", type=int, default=518,
                    help="Resize the long edge before inference (default 518, "
                         "matches mega-sam's UniDepth invocation)")
    ap.add_argument("--device", default=None,
                    help="'cuda' / 'cpu'. Auto-detects if omitted.")
    ap.add_argument("--hf_model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                    help="(DepthAnything-V2 only) Hugging Face model id. The "
                         "Metric-Outdoor variant exists too: "
                         "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    args = ap.parse_args()

    # Auto-detect device only when one of the backends is actually invoked.
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    rgb_source = Path(args.rgb_source).resolve()
    output_depth = Path(args.output_depth).resolve()

    if args.method == "unidepth":
        stats = predict_unidepth(rgb_source, output_depth, args.long_dim, device)
    else:
        stats = predict_depth_anything_v2(
            rgb_source, output_depth, args.long_dim, device, hf_model=args.hf_model,
        )

    _print_report(stats, source_resized_to=stats["model_input_shape"])


if __name__ == "__main__":
    main()
