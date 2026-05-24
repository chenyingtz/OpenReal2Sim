#!/usr/bin/env python3
"""Step 4.2a — Image-to-3D for the robot mesh (M^o).

Takes the foreground crop produced by `run_object_clear.py` and produces a
3D mesh asset for the LEGO dog. Two backends:

  --backend hunyuan3d    (default if third_party/Hunyuan3D-2 is set up)
                         Tencent Hunyuan3D-2 via tencent/Hunyuan3D-2 weights.
                         Photo-realistic geometry; needs GPU + ~10 GB model.
  --backend box          Always works. Builds a coarse box mesh with the same
                         aspect ratio as the mask bbox + an estimated depth.
                         Useful only for pipeline plumbing / smoke testing —
                         the result will not look like a dog.

Output (default name matches what scripts/physics/run_step5.py reads):
    <output_dir>/lego_dog.obj
    <output_dir>/_robot_mesh_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
HY3D_SRC = REPO_ROOT / "third_party" / "Hunyuan3D-2"


# ───────────────────────── Hunyuan3D-2 backend ─────────────────────────

def generate_with_hunyuan3d(
    foreground_path: Path,
    output_path: Path,
    model_id: str = "tencent/Hunyuan3D-2",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.5,
    octree_resolution: int = 256,
    device: Optional[str] = None,
    seed: int = 0,
) -> dict:
    if not HY3D_SRC.exists():
        raise RuntimeError(
            f"Hunyuan3D-2 not found at {HY3D_SRC}. Run:\n"
            f"    git submodule update --init --recursive third_party/Hunyuan3D-2"
        )
    if str(HY3D_SRC) not in sys.path:
        sys.path.insert(0, str(HY3D_SRC))

    try:
        import torch
        from PIL import Image
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as e:
        raise RuntimeError(
            "Hunyuan3D dependencies missing. Install with:\n"
            f"    cd {HY3D_SRC} && pip install -r requirements.txt"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[robot_mesh] device       : {device}")
    print(f"[robot_mesh] model        : {model_id}")

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_id, device=device,
    )

    img = Image.open(foreground_path).convert("RGBA")
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"[robot_mesh] running image-to-3D (steps={num_inference_steps}, "
          f"guidance={guidance_scale}, octree={octree_resolution}) ...")
    mesh_list = pipeline(
        image=img,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        octree_resolution=octree_resolution,
    )
    # Hunyuan3D returns a list (one per image); take the first.
    mesh = mesh_list[0]
    if not hasattr(mesh, "export"):
        # Some Hunyuan3D versions return a custom Mesh object; convert via trimesh.
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.export(str(output_path))

    return {
        "backend": "hunyuan3d",
        "model_id": model_id,
        "vertices": int(len(mesh.vertices)) if hasattr(mesh, "vertices") else None,
        "faces": int(len(mesh.faces)) if hasattr(mesh, "faces") else None,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "octree_resolution": octree_resolution,
    }


# ───────────────────────── Box fallback backend ─────────────────────────

def _bbox_from_alpha(rgba: np.ndarray):
    """Return (x0, y0, x1, y1) from the alpha channel of an RGBA image."""
    alpha = rgba[..., 3] > 127
    ys, xs = np.where(alpha)
    if len(ys) == 0:
        raise RuntimeError("Foreground PNG has empty alpha channel; nothing to bbox.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def generate_box_proxy(
    foreground_path: Path,
    output_path: Path,
    estimated_length_meters: float = 0.20,
    aspect_yz: float = 0.5,
    aspect_xz: float = 0.35,
) -> dict:
    """Build a quick box mesh whose XY aspect matches the mask bbox.

    Lets Step 5 onward run for plumbing tests even when Hunyuan3D weights are
    unavailable. NOT a substitute for a real mesh — geometry is just a box.
    Estimated dimensions:
        length (X) = estimated_length_meters
        width  (Y) = length * (bbox_h / bbox_w) * aspect_yz
        height (Z) = length * aspect_xz
    """
    import cv2
    rgba = cv2.imread(str(foreground_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        raise RuntimeError(
            f"Box fallback needs an RGBA foreground; got {foreground_path}. "
            f"Run scripts/assets/run_object_clear.py first."
        )
    x0, y0, x1, y1 = _bbox_from_alpha(rgba)
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)

    L = float(estimated_length_meters)
    W = L * (bbox_h / bbox_w) * aspect_yz
    H = L * aspect_xz
    box = trimesh.creation.box(extents=(L, W, H))
    box.apply_translation([0.0, 0.0, H / 2])  # rest on z=0
    box.export(str(output_path))

    return {
        "backend": "box",
        "bbox_pixels": [x0, y0, x1, y1],
        "estimated_length_m": L,
        "estimated_width_m": W,
        "estimated_height_m": H,
        "note": "Geometric proxy only; replace with hunyuan3d for a real mesh.",
    }


# ───────────────────────── Main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Step 4.2a — Image-to-3D for the robot mesh")
    ap.add_argument("--foreground", required=True,
                    help="Path to foreground RGBA PNG from run_object_clear.py "
                         "(typically foreground_tight.png)")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write lego_dog.obj")
    ap.add_argument("--output_name", default="lego_dog.obj",
                    help="Filename of the output mesh (default lego_dog.obj)")
    ap.add_argument("--backend", choices=["hunyuan3d", "box"], default="hunyuan3d")
    ap.add_argument("--model_id", default="tencent/Hunyuan3D-2",
                    help="(hunyuan3d) HF model id")
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=5.5)
    ap.add_argument("--octree_resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--estimated_length_meters", type=float, default=0.20,
                    help="(box) Real-world length of the dog along its long axis.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    foreground = Path(args.foreground)

    if args.backend == "hunyuan3d":
        try:
            meta = generate_with_hunyuan3d(
                foreground_path=foreground,
                output_path=output_path,
                model_id=args.model_id,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                octree_resolution=args.octree_resolution,
                device=args.device,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[robot_mesh] Hunyuan3D failed ({e}). Falling back to box proxy.")
            meta = generate_box_proxy(
                foreground_path=foreground,
                output_path=output_path,
                estimated_length_meters=args.estimated_length_meters,
            )
    else:
        meta = generate_box_proxy(
            foreground_path=foreground,
            output_path=output_path,
            estimated_length_meters=args.estimated_length_meters,
        )

    meta["foreground"] = str(foreground)
    meta["output"] = str(output_path)
    with open(output_dir / "_robot_mesh_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[robot_mesh] wrote {output_path}  (backend={meta['backend']})")


if __name__ == "__main__":
    main()
