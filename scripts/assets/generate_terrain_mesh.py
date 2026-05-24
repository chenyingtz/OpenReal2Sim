#!/usr/bin/env python3
"""Step 4.2b — Generate the background terrain mesh M^b.

Given the inpainted background image (from `run_object_clear.py`) + the
matching depth map (e.g. UniDepth's frame_0000.npz), unproject every
non-dog pixel into a 3D point cloud and triangulate it as a height field.
Edge triangles spanning a depth discontinuity are dropped so the floor
mesh has clean borders rather than long depth-edge spikes.

This is a pure-numpy implementation — no open3d / Poisson dependency —
because the depth map is already a regular grid, which makes a direct
quad → 2-triangle mesh both faster and topologically cleaner than
Poisson surface reconstruction for floor-like scenes.

Output:
    <output_dir>/terrain.obj            triangle mesh in camera frame, meters
    <output_dir>/_terrain_meta.json     vertex count, bbox, dropped triangles

Inputs:
    --background_depth  data/lego_dog_walk/UniDepth_outputs/frame_0000.npz
       (or .npy / 16-bit-mm .png)
    --background_image  (optional) for vertex colors
    --mask              data/lego_dog_walk/robot_mask.png — dog region to skip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import trimesh


# ───────────────────────── Depth I/O ─────────────────────────

def _load_depth_and_fov(path: Path, fallback_fov_deg: Optional[float]
                       ) -> Tuple[np.ndarray, float]:
    ext = path.suffix.lower()
    if ext == ".npz":
        with np.load(path) as z:
            depth = z["depth"].astype(np.float32)
            fov_deg = float(np.asarray(z["fov"]).reshape(-1)[0]) if "fov" in z else None
        if fov_deg is None:
            if fallback_fov_deg is None:
                raise ValueError(f"{path} has no 'fov'; pass --fov_degrees.")
            fov_deg = fallback_fov_deg
        return depth, fov_deg
    if ext == ".npy":
        depth = np.load(path).astype(np.float32)
        if fallback_fov_deg is None:
            raise ValueError("NPY input has no fov; pass --fov_degrees.")
        return depth, fallback_fov_deg
    if ext == ".png":
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None or d.dtype != np.uint16:
            raise ValueError(f"{path}: expected 16-bit mm PNG")
        if fallback_fov_deg is None:
            raise ValueError("PNG input has no fov; pass --fov_degrees.")
        return (d.astype(np.float32) / 1000.0), fallback_fov_deg
    raise ValueError(f"Unrecognized depth extension: {ext}")


def _load_mask(path: Path, target_hw: Tuple[int, int], invert: bool) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Could not read mask: {path}")
    H, W = target_hw
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    b = m > 127
    return ~b if invert else b


# ───────────────────────── Height-field mesh ─────────────────────────

def depth_to_mesh(
    depth: np.ndarray,
    fov_deg: float,
    foreground_mask: Optional[np.ndarray] = None,
    background_color: Optional[np.ndarray] = None,
    mask_dilation_iters: int = 3,
    edge_threshold_m: float = 0.05,
    stride: int = 1,
) -> Tuple[trimesh.Trimesh, dict]:
    """Unproject `depth` to a triangle mesh on a regular pixel grid.

    For each 2x2 pixel block we emit two triangles, but only if:
      - every corner has a valid (positive, finite) depth,
      - no corner is inside the dilated foreground mask (= dog), and
      - the max-min depth across the block is < edge_threshold_m (avoids
        long spikes along the dog silhouette and along walls).
    """
    H, W = depth.shape
    fov_rad = float(np.deg2rad(fov_deg))
    fx = (W / 2.0) / float(np.tan(fov_rad / 2.0))
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    # Dilate dog mask so anti-aliased silhouette pixels are also excluded.
    if foreground_mask is not None and mask_dilation_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        fg_dilated = cv2.dilate(
            (foreground_mask.astype(np.uint8) * 255), kernel,
            iterations=mask_dilation_iters,
        ) > 127
    else:
        fg_dilated = (foreground_mask if foreground_mask is not None
                      else np.zeros_like(depth, dtype=bool))

    # Subsample if requested (large depth maps).
    if stride > 1:
        depth = depth[::stride, ::stride]
        fg_dilated = fg_dilated[::stride, ::stride]
        H, W = depth.shape
        fx = fx / stride
        fy = fy / stride
        cx = cx / stride
        cy = cy / stride
        if background_color is not None:
            background_color = background_color[::stride, ::stride]

    # Build per-pixel 3D points (camera frame: +z forward).
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    X = (us - cx) * depth / fx
    Y = (vs - cy) * depth / fy
    Z = depth

    valid = np.isfinite(depth) & (depth > 0) & ~fg_dilated
    # Flatten to vertex list; remember a (H,W)->vertex_idx LUT for triangles.
    idx_lut = -np.ones((H, W), dtype=np.int64)
    vidx = np.where(valid.ravel())[0]
    idx_lut.ravel()[vidx] = np.arange(len(vidx))

    verts = np.stack([X.ravel()[vidx], Y.ravel()[vidx], Z.ravel()[vidx]], axis=1)

    colors = None
    if background_color is not None:
        colors = background_color.reshape(-1, background_color.shape[-1])[vidx]

    # For each 2x2 block (i, j), emit two triangles if all 4 corners are valid
    # and the depth span across the block is small enough.
    faces = []
    dropped_invalid = 0
    dropped_jump = 0

    # Use vectorized slicing — much faster than Python loops over pixels.
    a = idx_lut[:-1, :-1]    # (H-1, W-1)
    b = idx_lut[:-1, 1:]
    c = idx_lut[1:, :-1]
    d = idx_lut[1:, 1:]
    Za = Z[:-1, :-1]
    Zb = Z[:-1, 1:]
    Zc = Z[1:, :-1]
    Zd = Z[1:, 1:]

    all_valid = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    dropped_invalid = int(((~all_valid)).sum())

    Zstack = np.stack([Za, Zb, Zc, Zd])
    Zmax = Zstack.max(axis=0)
    Zmin = Zstack.min(axis=0)
    smooth = (Zmax - Zmin) < edge_threshold_m

    ok = all_valid & smooth
    dropped_jump = int((all_valid & ~smooth).sum())

    ii, jj = np.where(ok)
    if len(ii):
        tri_lo = np.stack([a[ii, jj], c[ii, jj], b[ii, jj]], axis=1)
        tri_hi = np.stack([b[ii, jj], c[ii, jj], d[ii, jj]], axis=1)
        faces = np.concatenate([tri_lo, tri_hi], axis=0)
    else:
        faces = np.zeros((0, 3), dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if colors is not None and len(mesh.vertices) == len(colors):
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh, vertex_colors=colors[:len(mesh.vertices)].astype(np.uint8),
        )

    stats = {
        "input_shape": [H, W],
        "fov_degrees": fov_deg,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "dropped_blocks_invalid_corner": dropped_invalid,
        "dropped_blocks_depth_jump": dropped_jump,
        "edge_threshold_m": edge_threshold_m,
        "stride": stride,
    }
    if len(mesh.vertices):
        stats["bbox_min"] = mesh.bounds[0].tolist()
        stats["bbox_max"] = mesh.bounds[1].tolist()
        stats["depth_range_m"] = [
            float(verts[:, 2].min()), float(verts[:, 2].max()),
        ]
    return mesh, stats


# ───────────────────────── Main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Step 4.2b — depth-to-mesh terrain")
    ap.add_argument("--background_depth", required=True,
                    help="Depth map for the anchor / background frame (NPZ / NPY / 16-bit PNG)")
    ap.add_argument("--mask", default=None,
                    help="Dog mask PNG — these pixels are excluded from the terrain "
                         "(use the original robot_mask.png, not the inpaint output)")
    ap.add_argument("--background_image", default=None,
                    help="Optional inpainted background.png for vertex colors")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--output_name", default="terrain.obj")
    ap.add_argument("--fov_degrees", type=float, default=None,
                    help="Required when the depth file has no embedded FOV (NPY / PNG).")
    ap.add_argument("--invert_mask", action="store_true",
                    help="Flip the mask (white=background -> white=dog)")
    ap.add_argument("--mask_dilation_iters", type=int, default=3,
                    help="Dilation iterations on the dog mask before excluding it.")
    ap.add_argument("--edge_threshold_m", type=float, default=0.05,
                    help="Drop triangles whose 4 corners span more than this depth.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Subsample the depth grid by this factor (1 = no skip).")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    depth, fov_deg = _load_depth_and_fov(Path(args.background_depth), args.fov_degrees)
    print(f"[terrain] depth shape: {depth.shape}  fov={fov_deg:.2f}°")
    H, W = depth.shape

    fg = None
    if args.mask is not None:
        fg = _load_mask(Path(args.mask), (H, W), invert=args.invert_mask)
        print(f"[terrain] mask coverage: {fg.mean()*100:.1f}% (this fraction is excluded)")

    bg_rgb = None
    if args.background_image is not None:
        bg = cv2.imread(args.background_image, cv2.IMREAD_COLOR)
        if bg is not None:
            if bg.shape[:2] != (H, W):
                bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    mesh, stats = depth_to_mesh(
        depth=depth, fov_deg=fov_deg,
        foreground_mask=fg,
        background_color=bg_rgb,
        mask_dilation_iters=args.mask_dilation_iters,
        edge_threshold_m=args.edge_threshold_m,
        stride=args.stride,
    )
    mesh.export(str(output_path))

    stats["background_depth"] = str(Path(args.background_depth))
    stats["mask"] = str(Path(args.mask)) if args.mask else None
    stats["background_image"] = str(Path(args.background_image)) if args.background_image else None
    stats["output"] = str(output_path)
    with open(output_dir / "_terrain_meta.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[terrain] vertices : {stats['vertices']}")
    print(f"[terrain] faces    : {stats['faces']}")
    print(f"[terrain] dropped  : {stats['dropped_blocks_invalid_corner']} invalid + "
          f"{stats['dropped_blocks_depth_jump']} depth-jump")
    print(f"[terrain] wrote    : {output_path}")


if __name__ == "__main__":
    main()
