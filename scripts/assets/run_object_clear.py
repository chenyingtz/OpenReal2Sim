#!/usr/bin/env python3
"""Step 4.1 — Segment and Inpaint the Floor Terrain.

Given the anchor RGB and the dog mask, produce two anchor-frame artifacts
the rest of Step 4 needs:

    <output_dir>/foreground.png         dog pixels with alpha (I^o, full size)
    <output_dir>/foreground_tight.png   dog cropped to its bbox + padding
    <output_dir>/background.png         the anchor with the dog erased (I^b)
    <output_dir>/dog_mask_dilated.png   the mask actually used for inpaint
    <output_dir>/_inpaint_meta.json     parameters + statistics

Two inpainting backends, picked at runtime:

  --backend objectclear   (default if jixin0101/ObjectClear weights are loadable)
                          Uses third_party/ObjectClear. Best quality but
                          requires a ~5 GB model download + GPU.
  --backend opencv        Always available. OpenCV's TELEA-method inpaint.
                          Fast, fine for plain floors; not great for textured
                          backgrounds, but good enough that Step 4.2's
                          terrain mesh comes out usable.

Matches the PDF CLI:

    python scripts/assets/run_object_clear.py \\
        --image      data/lego_dog_walk/input_anchor.png \\
        --mask       data/lego_dog_walk/robot_mask.png \\
        --output_dir data/lego_dog_walk/inpainted/
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
OBJECTCLEAR_SRC = REPO_ROOT / "third_party" / "ObjectClear"


# ───────────────────────── Helpers ─────────────────────────

def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"OpenCV could not decode {path}")
    return img


def _load_mask(path: Path, target_hw, invert: bool) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"OpenCV could not decode {path}")
    H, W = target_hw
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    bool_mask = m > 127
    if invert:
        bool_mask = ~bool_mask
    return bool_mask.astype(np.uint8) * 255


# ───────────────────────── OpenCV backend ─────────────────────────

def inpaint_opencv(image_bgr: np.ndarray, mask_u8: np.ndarray,
                   inpaint_radius: int = 5,
                   method: str = "telea") -> np.ndarray:
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image_bgr, mask_u8, inpaint_radius, flag)


# ───────────────────────── ObjectClear backend ─────────────────────────

def inpaint_objectclear(image_bgr: np.ndarray, mask_u8: np.ndarray,
                        steps: int, strength: float, guidance_scale: float,
                        seed: int, device: Optional[str]) -> np.ndarray:
    """Run third_party/ObjectClear. Heavy: ~5 GB model + GPU."""
    if not OBJECTCLEAR_SRC.exists():
        raise RuntimeError(
            f"ObjectClear not found at {OBJECTCLEAR_SRC}. Run:\n"
            f"    git submodule update --init --recursive third_party/ObjectClear"
        )
    if str(OBJECTCLEAR_SRC) not in sys.path:
        sys.path.insert(0, str(OBJECTCLEAR_SRC))

    try:
        import torch
        from PIL import Image
        from objectclear.pipelines.pipeline_objectclear import ObjectClearPipeline
    except ImportError as e:
        raise RuntimeError(
            "ObjectClear dependencies missing. Install with:\n"
            f"    pip install -r {OBJECTCLEAR_SRC}/requirements.txt"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=dtype,
        apply_attention_guided_fusion=True,
        variant=("fp16" if device == "cuda" else None),
    )
    pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Resize so the SHORT side is 512 (the size ObjectClear was trained at).
    H, W = image_bgr.shape[:2]
    short = min(H, W)
    scale = 512.0 / short
    new_w, new_h = int(round(W * scale)), int(round(H * scale))
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.BICUBIC)
    mask_pil = Image.fromarray(mask_u8).resize((new_w, new_h), Image.NEAREST)

    result = pipe(
        prompt="remove the instance of object",
        image=img_pil,
        mask_image=mask_pil,
        generator=generator,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=guidance_scale,
        height=new_h,
        width=new_w,
        return_attn_map=False,
    )
    fused_pil = result.images[0].resize((W, H), Image.BICUBIC)
    return cv2.cvtColor(np.asarray(fused_pil), cv2.COLOR_RGB2BGR)


# ───────────────────────── Driver ─────────────────────────

def process(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    backend: str = "opencv",
    invert_mask: bool = False,
    dilation_iters: int = 2,
    bbox_pad: int = 20,
    inpaint_radius: int = 5,
    opencv_method: str = "telea",
    oc_steps: int = 25,
    oc_strength: float = 1.0,
    oc_guidance_scale: float = 1.5,
    oc_seed: int = 0,
    device: Optional[str] = None,
) -> dict:
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = _load_image(image_path)
    H, W = image_bgr.shape[:2]
    mask_u8 = _load_mask(mask_path, (H, W), invert=invert_mask)

    coverage = float((mask_u8 > 127).mean())
    print(f"[inpaint] image    : {image_path}  ({W}x{H})")
    print(f"[inpaint] mask     : {mask_path}  coverage={coverage*100:.1f}%")
    if coverage > 0.5 and not invert_mask:
        print(f"[inpaint] WARNING: mask covers {coverage*100:.1f}% of the image "
              f"(unusual for a dog mask). Consider --invert_mask.")
    if coverage < 0.005:
        print(f"[inpaint] WARNING: mask covers only {coverage*100:.2f}% of the "
              f"image — likely too small to identify the dog correctly.")

    # Dilate so the inpaint covers the dog's anti-aliased halo and motion blur
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_u8, kernel, iterations=dilation_iters)
    cv2.imwrite(str(output_dir / "dog_mask_dilated.png"), mask_dilated)

    # ── Run the inpaint backend ──
    t0 = time.time()
    print(f"[inpaint] backend  : {backend}")
    if backend == "objectclear":
        try:
            background = inpaint_objectclear(
                image_bgr, mask_dilated,
                steps=oc_steps, strength=oc_strength,
                guidance_scale=oc_guidance_scale, seed=oc_seed, device=device,
            )
        except Exception as e:
            print(f"[inpaint] ObjectClear failed ({e}); falling back to OpenCV.")
            background = inpaint_opencv(image_bgr, mask_dilated,
                                        inpaint_radius=inpaint_radius,
                                        method=opencv_method)
            backend = "opencv (fallback)"
    elif backend == "opencv":
        background = inpaint_opencv(image_bgr, mask_dilated,
                                    inpaint_radius=inpaint_radius,
                                    method=opencv_method)
    else:
        raise ValueError(f"Unknown --backend: {backend}")
    inpaint_dt = time.time() - t0

    bg_path = output_dir / "background.png"
    cv2.imwrite(str(bg_path), background)

    # ── Foreground extraction ──
    # Keep the dog pixels on a transparent background as a 4-channel PNG; the
    # image-to-3D pipeline in generate_robot_mesh.py expects this format.
    alpha = mask_dilated  # use the same mask we inpainted
    rgba = np.dstack([image_bgr, alpha])
    fg_path = output_dir / "foreground.png"
    cv2.imwrite(str(fg_path), rgba)

    # Tight crop to mask bbox + padding (image-to-3D models work best on
    # square-ish crops at ~512 px).
    ys, xs = np.where(mask_dilated > 127)
    tight_path = None
    bbox = None
    if len(ys) > 0:
        y0 = max(0, int(ys.min()) - bbox_pad)
        y1 = min(H, int(ys.max()) + bbox_pad)
        x0 = max(0, int(xs.min()) - bbox_pad)
        x1 = min(W, int(xs.max()) + bbox_pad)
        tight = rgba[y0:y1, x0:x1]
        tight_path = output_dir / "foreground_tight.png"
        cv2.imwrite(str(tight_path), tight)
        bbox = [int(x0), int(y0), int(x1), int(y1)]

    meta = {
        "image": str(image_path),
        "mask": str(mask_path),
        "image_shape": [H, W],
        "mask_coverage_pct": float(coverage * 100),
        "backend": backend,
        "dilation_iters": dilation_iters,
        "bbox_pad": bbox_pad,
        "foreground": str(fg_path),
        "foreground_tight": str(tight_path) if tight_path else None,
        "background": str(bg_path),
        "bbox": bbox,
        "inpaint_seconds": float(inpaint_dt),
    }
    with open(output_dir / "_inpaint_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[inpaint] background -> {bg_path}")
    print(f"[inpaint] foreground -> {fg_path}")
    if tight_path:
        print(f"[inpaint] tight crop -> {tight_path}  bbox={bbox}")
    print(f"[inpaint] done in {inpaint_dt:.1f}s")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Step 4.1 — Segment and inpaint the floor")
    ap.add_argument("--image", required=True, help="Anchor RGB image (I_0)")
    ap.add_argument("--mask", required=True, help="Dog mask PNG (white = dog)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--backend", choices=["opencv", "objectclear"], default="opencv",
                    help="Inpaint backend. 'objectclear' is higher quality but "
                         "needs the ~5GB jixin0101/ObjectClear model + GPU.")
    ap.add_argument("--invert_mask", action="store_true",
                    help="Flip the mask (white=background -> white=dog)")
    ap.add_argument("--dilation_iters", type=int, default=2,
                    help="3x3 dilation iterations on the mask before inpainting.")
    ap.add_argument("--bbox_pad", type=int, default=20,
                    help="Pixels of padding around the mask bbox for the tight crop.")
    ap.add_argument("--inpaint_radius", type=int, default=5,
                    help="(OpenCV) Inpaint radius in pixels.")
    ap.add_argument("--opencv_method", choices=["telea", "ns"], default="telea")
    ap.add_argument("--oc_steps", type=int, default=25,
                    help="(ObjectClear) Diffusion steps.")
    ap.add_argument("--oc_strength", type=float, default=1.0)
    ap.add_argument("--oc_guidance_scale", type=float, default=1.5)
    ap.add_argument("--oc_seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    process(
        image_path=args.image,
        mask_path=args.mask,
        output_dir=args.output_dir,
        backend=args.backend,
        invert_mask=args.invert_mask,
        dilation_iters=args.dilation_iters,
        bbox_pad=args.bbox_pad,
        inpaint_radius=args.inpaint_radius,
        opencv_method=args.opencv_method,
        oc_steps=args.oc_steps,
        oc_strength=args.oc_strength,
        oc_guidance_scale=args.oc_guidance_scale,
        oc_seed=args.oc_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
