#!/usr/bin/env python3
"""Step 3.2 — Metric Depth Calibration.

Fit a robust linear correction `d_metric = alpha * d_gen + beta` between
the generated (UniDepth / MegaSaM) anchor-frame depth and the real sensor
depth at the same frame, then — optionally in the same run — apply that
correction to every per-frame generated depth and unproject the masked
foreground into per-frame point clouds suitable for Step 6 tracking.

Matches the PDF CLI:

    python scripts/reconstruction/calibrate_metric_scale.py \
        --real_depth data/lego_dog_walk/input_depth.png \
        --gen_depth  data/lego_dog_walk/relative_depths/frame_000.npz \
        --config     configs/lego_dog_locomotion.yaml

The robust fit is `scipy.optimize.least_squares(..., loss='huber')` with a
default outlier threshold of 5 cm. By default the fit uses every pixel where
*both* the real sensor and the generated depth report finite, positive depth
— giving the regression as wide a depth range as possible and a well-
conditioned (alpha, beta). `--mask` is available for users who want to
restrict the fit to a specific region, but tight masks (e.g. only the dog)
can collapse the depth range and make alpha unidentifiable; the script
warns when this happens.

Outputs (all under `--output_dir`, default `<gen_depth>/../metric/`):
    calibration.json            alpha, beta, fit residual stats, pixel count
    metric_depths/<stem>.npy    (H, W) float32 metric depth per frame
    metric_point_clouds/<stem>.npy  (N, 3) float32 dog points per frame
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import least_squares


# ───────────────────────── Depth I/O ─────────────────────────

def _load_real_depth(path: Path, units: str) -> np.ndarray:
    """Load a real-sensor depth map and return (H, W) float64 in meters.

    Supports 16-bit PNG (mm encoding, the RealSense / Kinect default),
    32-bit float PNG/TIFF/NPY, and NPZ with a `depth` key. Invalid pixels
    encoded as 0 become NaN; downstream masking respects this.
    """
    ext = path.suffix.lower()
    if ext == ".npz":
        with np.load(path) as z:
            d = np.asarray(z["depth"], dtype=np.float64)
    elif ext == ".npy":
        d = np.load(path).astype(np.float64)
    elif ext in (".png", ".tif", ".tiff"):
        img = np.asarray(Image.open(path))
        if img.dtype == np.uint16:
            d = img.astype(np.float64)
            if units == "mm":
                d /= 1000.0
        elif img.dtype == np.uint8:
            raise ValueError(
                f"{path}: 8-bit PNG detected — that is almost certainly a "
                "depth *visualization* (colormap), not a raw sensor map. "
                "Re-export the RealSense / Kinect frame as a 16-bit PNG "
                "(values in millimeters) and re-run."
            )
        else:
            d = img.astype(np.float64)
    else:
        raise ValueError(f"Unrecognized real-depth extension: {ext}")

    if units == "meters":
        pass  # already in meters
    elif units == "mm" and ext not in (".png",):
        d = d / 1000.0

    # Sensor returns 0 (and sometimes NaN) for invalid pixels.
    d = np.where(d > 0, d, np.nan)
    return d


def _load_gen_depth(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    """Load a generated-frame depth map. Returns (depth_meters, fov_deg_or_None).

    UniDepth .npz: keys `depth` (H, W float32, meters) and `fov` (scalar deg).
    .npy: assumed already-metric float32 (no FOV info — caller must supply).
    .png: 16-bit mm encoding (matches the PDF's frame_000.png convention).
    """
    ext = path.suffix.lower()
    if ext == ".npz":
        with np.load(path) as z:
            d = np.asarray(z["depth"], dtype=np.float64)
            fov = float(np.asarray(z["fov"]).reshape(-1)[0]) if "fov" in z else None
        return d, fov
    if ext == ".npy":
        return np.load(path).astype(np.float64), None
    if ext == ".png":
        img = np.asarray(Image.open(path))
        if img.dtype == np.uint16:
            return img.astype(np.float64) / 1000.0, None
        raise ValueError(f"{path}: expected 16-bit PNG for generated depth, got {img.dtype}")
    raise ValueError(f"Unrecognized generated-depth extension: {ext}")


def _load_mask(path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"))
    H, W = target_hw
    if img.shape != (H, W):
        img = np.array(Image.fromarray(img).resize((W, H), Image.NEAREST))
    return img > 127


def _resize_to(arr: np.ndarray, target_hw: Tuple[int, int], nearest: bool = False
               ) -> np.ndarray:
    """Resize a depth map to (H, W). NaNs are preserved by masking after resize."""
    H, W = target_hw
    if arr.shape == (H, W):
        return arr
    finite = np.isfinite(arr) & (arr > 0)
    arr_filled = np.where(finite, arr, 0.0).astype(np.float32)
    method = Image.NEAREST if nearest else Image.BILINEAR
    out = np.asarray(Image.fromarray(arr_filled).resize((W, H), method)).astype(np.float64)
    valid = np.asarray(Image.fromarray(finite.astype(np.uint8) * 255)
                       .resize((W, H), Image.NEAREST)) > 127
    out = np.where(valid & (out > 0), out, np.nan)
    return out


# ───────────────────────── Huber regression ─────────────────────────

def huber_alpha_beta(
    gen: np.ndarray,
    real: np.ndarray,
    f_scale: float = 0.05,
    max_pixels: int = 50000,
    seed: int = 0,
) -> Tuple[float, float, dict]:
    """Robust linear fit:  alpha * gen + beta ≈ real  (Huber loss).

    `gen` and `real` are equal-length 1D arrays of paired metric depths
    (NaNs already filtered). `f_scale` (in meters) controls the Huber knee;
    residuals smaller than f_scale are quadratic, larger ones linear, so the
    fit is insensitive to outliers like depth-edge bleed, dynamic-object
    pixels, or sensor dropouts.
    """
    n = len(gen)
    if n < 50:
        raise ValueError(
            f"Only {n} co-valid pixels between real and generated depth — "
            "calibration needs more overlap. Check that the mask is loose "
            "enough and the depth maps are aligned."
        )
    if n > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_pixels, replace=False)
        gen = gen[idx]
        real = real[idx]
        n = max_pixels

    # Conditioning check: if the gen depth has very little variation over the
    # fit region, alpha can absorb arbitrarily into beta and the fit becomes
    # degenerate. Warn the user before letting Huber pick the trivial solution.
    gen_std = float(gen.std())
    real_std = float(real.std())
    if gen_std < 0.02:  # less than 2 cm of generated-depth variation
        print(f"[calib] WARNING: gen depth std = {gen_std*100:.1f} cm over the "
              f"fit region — too narrow to identify alpha reliably. The fit "
              f"may collapse to alpha≈0 (beta absorbs the mean). Drop --mask "
              f"or widen it to include floor/background depth variation.")

    # Step 1: ratio-MAD pre-filter. For the linear model real = alpha*gen + beta
    # the ratio real/gen ≈ alpha + beta/gen is tight for inliers and wild for
    # leverage outliers (extreme gen values with arbitrary real values). Reject
    # pixels whose ratio sits >5 MAD from the median — this is a 1D test that
    # ignores high-leverage outliers from gen-side noise (which would dominate
    # any 2D loss including Huber).
    ratios = real / np.maximum(gen, 1e-6)
    med = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - med)))
    ratio_threshold = 5.0 * max(mad, 1e-3)
    ratio_inliers = np.abs(ratios - med) < ratio_threshold
    n_inliers = int(ratio_inliers.sum())
    if n_inliers < 50:
        # Pre-filter rejected too much — fall back to keeping the inner 50%.
        sorted_idx = np.argsort(np.abs(ratios - med))
        ratio_inliers = np.zeros(len(gen), dtype=bool)
        ratio_inliers[sorted_idx[: max(50, len(gen) // 2)]] = True
        n_inliers = int(ratio_inliers.sum())

    gen_clean = gen[ratio_inliers]
    real_clean = real[ratio_inliers]

    # Step 2: OLS on the cleaned set seeds the Huber refinement.
    A = np.column_stack([gen_clean, np.ones_like(gen_clean)])
    sol, *_ = np.linalg.lstsq(A, real_clean, rcond=None)
    alpha0, beta0 = float(sol[0]), float(sol[1])

    # Step 3: Huber refinement on the cleaned set. With most extreme outliers
    # already removed, Huber locks the fit onto the inlier population.
    def residual(params):
        return params[0] * gen_clean + params[1] - real_clean

    res = least_squares(residual, x0=[alpha0, beta0], loss="huber", f_scale=f_scale)
    alpha, beta = float(res.x[0]), float(res.x[1])

    fitted_residuals = np.abs(alpha * gen_clean + beta - real_clean)
    stats = {
        "n_pixels_total": int(n),
        "n_pixels_after_ratio_filter": n_inliers,
        "ratio_filter_rejected_pct": 100.0 * (n - n_inliers) / max(n, 1),
        "f_scale_meters": float(f_scale),
        "gen_std_meters": gen_std,
        "real_std_meters": real_std,
        "median_ratio": med,
        "mad_ratio": mad,
        "alpha_initial_ols": alpha0,
        "beta_initial_ols": beta0,
        "alpha_huber": alpha,
        "beta_huber": beta,
        "residual_median_mm": float(np.median(fitted_residuals) * 1000.0),
        "residual_p95_mm": float(np.percentile(fitted_residuals, 95) * 1000.0),
        "residual_max_mm": float(fitted_residuals.max() * 1000.0),
    }
    return alpha, beta, stats


def _pair_valid_pixels(real: np.ndarray, gen: np.ndarray,
                       mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gen_flat, real_flat) for pixels where both depths are valid
    and the mask (if provided) selects."""
    valid = np.isfinite(real) & (real > 0) & np.isfinite(gen) & (gen > 0)
    if mask is not None:
        valid &= mask
    return gen[valid], real[valid]


# ───────────────────────── Unprojection ─────────────────────────

def _fov_to_focal(fov_deg: float, width: int) -> Tuple[float, float]:
    fx = (width / 2.0) / float(np.tan(np.deg2rad(fov_deg) / 2.0))
    return fx, fx


def unproject(depth: np.ndarray, fov_deg: float,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
    H, W = depth.shape
    fx, fy = _fov_to_focal(fov_deg, W)
    cx, cy = W / 2.0, H / 2.0
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask
    vs, us = np.where(valid)
    if vs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    d = depth[vs, us]
    x = (us - cx) * d / fx
    y = (vs - cy) * d / fy
    return np.stack([x, y, d], axis=1).astype(np.float32)


# ───────────────────────── Pipeline driver ─────────────────────────

def _pair_directories(gen_dir: Path, mask_dir: Optional[Path]):
    gen_files = sorted([p for p in gen_dir.iterdir()
                        if p.suffix.lower() in (".npz", ".npy", ".png")])
    if not gen_files:
        raise FileNotFoundError(f"No gen depth files in {gen_dir}")
    if mask_dir is None:
        return [(g, None) for g in gen_files]
    masks = {p.stem: p for p in mask_dir.iterdir()
             if p.suffix.lower() in (".png", ".jpg")}
    pairs = []
    for g in gen_files:
        pairs.append((g, masks.get(g.stem)))
    return pairs


def calibrate(
    real_depth_path: Path,
    gen_depth_path: Path,
    output_dir: Path,
    mask_path: Optional[Path] = None,
    real_units: str = "mm",
    huber_f_scale: float = 0.05,
    gen_depth_dir: Optional[Path] = None,
    mask_dir: Optional[Path] = None,
    config: Optional[dict] = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real = _load_real_depth(Path(real_depth_path), units=real_units)
    gen_anchor, fov_anchor = _load_gen_depth(Path(gen_depth_path))

    # Co-register resolutions: bring real onto gen's grid (the smaller of the two).
    if real.shape != gen_anchor.shape:
        real_for_fit = _resize_to(real, gen_anchor.shape, nearest=False)
    else:
        real_for_fit = real

    mask = None
    if mask_path is not None:
        mask = _load_mask(Path(mask_path), gen_anchor.shape)

    g_flat, r_flat = _pair_valid_pixels(real_for_fit, gen_anchor, mask)
    print(f"[calib] anchor pixels paired: {len(g_flat)} "
          f"(mask: {'yes' if mask is not None else 'no'})")

    alpha, beta, stats = huber_alpha_beta(
        g_flat, r_flat, f_scale=huber_f_scale,
    )
    print(f"[calib] alpha = {alpha:+.5f}   beta = {beta:+.5f} m")
    print(f"[calib] residuals (mm): median={stats['residual_median_mm']:.2f}  "
          f"p95={stats['residual_p95_mm']:.2f}  max={stats['residual_max_mm']:.2f}")

    calibration = {
        "alpha": alpha,
        "beta": beta,
        "real_depth": str(real_depth_path),
        "gen_depth_anchor": str(gen_depth_path),
        "mask": str(mask_path) if mask_path else None,
        "real_units_input": real_units,
        "fov_anchor_degrees": fov_anchor,
        "fit_stats": stats,
        "schema_version": "1.0",
    }
    if config:
        calibration["config"] = config

    with open(output_dir / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    # Apply to all frames if requested.
    if gen_depth_dir is not None:
        metric_dir = output_dir / "metric_depths"
        metric_dir.mkdir(exist_ok=True)
        pcd_dir = output_dir / "metric_point_clouds"
        if mask_dir is not None:
            pcd_dir.mkdir(exist_ok=True)

        pairs = _pair_directories(Path(gen_depth_dir),
                                  Path(mask_dir) if mask_dir else None)
        print(f"[calib] applying (alpha,beta) to {len(pairs)} frames")
        written_metric, written_pcd = 0, 0
        for i, (gp, mp) in enumerate(pairs):
            d, fov = _load_gen_depth(gp)
            d_metric = alpha * d + beta
            np.save(metric_dir / f"{gp.stem}.npy", d_metric.astype(np.float32))
            written_metric += 1
            if mask_dir is not None and mp is not None and fov is not None:
                m = _load_mask(mp, d_metric.shape)
                pts = unproject(d_metric, fov, m)
                if len(pts) > 0:
                    np.save(pcd_dir / f"{gp.stem}.npy", pts)
                    written_pcd += 1
            if i == 0 or (i + 1) % 25 == 0 or i == len(pairs) - 1:
                print(f"[calib] frame {i+1}/{len(pairs)}  "
                      f"depth_range=[{float(np.nanmin(d_metric)):.3f}, "
                      f"{float(np.nanmax(d_metric)):.3f}] m")
        calibration["frames_metric_depth_written"] = written_metric
        calibration["frames_point_cloud_written"] = written_pcd
        with open(output_dir / "calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"[calib] wrote {written_metric} metric depth maps "
              f"and {written_pcd} point clouds under {output_dir}")

    return calibration


def main():
    ap = argparse.ArgumentParser(description="Step 3.2 — Metric depth calibration")
    ap.add_argument("--real_depth", required=True,
                    help="Real sensor depth at the anchor frame "
                         "(16-bit PNG mm, NPZ, or NPY)")
    ap.add_argument("--gen_depth", required=True,
                    help="Generated anchor-frame depth (UniDepth .npz, .npy, or .png)")
    ap.add_argument("--mask", default=None,
                    help="Optional anchor-frame mask PNG. Restricts the (alpha, "
                         "beta) fit to a region. Default is no mask, which "
                         "gives the regression the widest depth range and the "
                         "most stable fit. A tight mask (e.g. only the dog) "
                         "can collapse the depth range and make alpha "
                         "unidentifiable; the script warns when this happens.")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write calibration.json + per-frame outputs "
                         "(default: <gen_depth dir>/../metric/)")
    ap.add_argument("--config", default=None,
                    help="Optional YAML config; values here are recorded into "
                         "calibration.json but do not override CLI flags.")
    ap.add_argument("--real_units", choices=["mm", "meters"], default="mm")
    ap.add_argument("--huber_f_scale", type=float, default=0.05,
                    help="Huber outlier threshold in meters (default 5 cm).")
    ap.add_argument("--gen_depth_dir", default=None,
                    help="If set, apply (alpha,beta) to every depth in this dir.")
    ap.add_argument("--mask_dir", default=None,
                    help="Per-frame masks (paired by stem to gen_depth_dir). "
                         "Required to also emit per-frame point clouds.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.gen_depth).parent.parent / "metric"

    config_dict = None
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            print(f"[calib] WARNING: failed to load config {args.config}: {e}")

    calibrate(
        real_depth_path=Path(args.real_depth),
        gen_depth_path=Path(args.gen_depth),
        output_dir=output_dir,
        mask_path=Path(args.mask) if args.mask else None,
        real_units=args.real_units,
        huber_f_scale=args.huber_f_scale,
        gen_depth_dir=Path(args.gen_depth_dir) if args.gen_depth_dir else None,
        mask_dir=Path(args.mask_dir) if args.mask_dir else None,
        config=config_dict,
    )


if __name__ == "__main__":
    main()
