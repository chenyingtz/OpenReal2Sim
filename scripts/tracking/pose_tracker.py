#!/usr/bin/env python3
"""Algorithmic core of Step 6 — point-cloud-to-mesh 6D pose tracking.

Importable so the CLI in `run_foundationpose.py` stays a thin wrapper and the
algorithm can be unit-tested in isolation.

Convention throughout:
- Points are (N, 3) row-major numpy arrays in world coordinates.
- A rigid pose is `(R, t)` with R in SO(3) and t in R^3 such that
        world_point = R @ mesh_local_point + t.
  Equivalently for row-stacked points:
        world_row = mesh_row @ R.T + t.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


POINT_CLOUD_EXTS = (".ply", ".npy", ".npz", ".pcd", ".xyz")


# ───────────────────────── Point-cloud loading ─────────────────────────

def _load_point_cloud(path: Path) -> np.ndarray:
    """Load a single point cloud as (N, 3) float64. Auto-detect by extension."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".ply":
        obj = trimesh.load(str(path), process=False)
        if isinstance(obj, trimesh.PointCloud):
            return np.asarray(obj.vertices, dtype=np.float64)
        if isinstance(obj, trimesh.Trimesh):
            return np.asarray(obj.vertices, dtype=np.float64)
        if isinstance(obj, trimesh.Scene):
            arrs = [np.asarray(g.vertices) for g in obj.geometry.values()
                    if hasattr(g, "vertices")]
            if not arrs:
                raise ValueError(f"{path}: scene has no vertex data")
            return np.concatenate(arrs, axis=0).astype(np.float64)
        raise ValueError(f"{path}: unsupported .ply contents ({type(obj)})")

    if ext == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=np.float64).reshape(-1, 3)

    if ext == ".npz":
        z = np.load(path)
        for key in ("points", "xyz", "pcd"):
            if key in z:
                return np.asarray(z[key], dtype=np.float64).reshape(-1, 3)
        first = list(z.keys())[0]
        return np.asarray(z[first], dtype=np.float64).reshape(-1, 3)

    if ext == ".xyz":
        return np.loadtxt(path)[:, :3].astype(np.float64)

    if ext == ".pcd":
        try:
            import open3d as o3d
        except ImportError as e:
            raise RuntimeError(
                "Reading .pcd requires open3d (`pip install open3d`)."
            ) from e
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points, dtype=np.float64)

    raise ValueError(f"Unrecognized point cloud extension: {ext}")


def discover_frames(point_cloud_dir: Path) -> List[Path]:
    """Return per-frame point cloud files sorted in natural numeric order."""
    point_cloud_dir = Path(point_cloud_dir)
    paths: List[Path] = []
    for ext in POINT_CLOUD_EXTS:
        paths.extend(point_cloud_dir.glob(f"*{ext}"))
    if not paths:
        raise FileNotFoundError(
            f"No point cloud files found in {point_cloud_dir} "
            f"(supported extensions: {POINT_CLOUD_EXTS})."
        )
    # Natural sort: shorter stems first, then lexicographic. Handles both
    # frame_0.ply / frame_1.ply / frame_10.ply and zero-padded variants.
    return sorted(set(paths), key=lambda p: (len(p.stem), p.stem))


# ───────────────────────── Mesh kd-tree ─────────────────────────

class MeshKDTree:
    """Nearest-neighbor query against a uniformly surface-sampled mesh.

    Why not `trimesh.proximity.ProximityQuery`: that constructs an rtree over
    every triangle, which on hundreds-of-thousands-of-faces meshes (e.g.
    photogrammetry / image-to-3D reconstructions) routinely allocates
    gigabytes and triggers the Linux OOM killer. For ICP-style closest-mesh-
    point queries, a kd-tree over a fixed surface-sample budget is equivalent
    within a few millimeters and is memory-bounded regardless of mesh size.
    """

    def __init__(self, mesh: trimesh.Trimesh, n_samples: int = 8000, seed: int = 0):
        from scipy.spatial import cKDTree
        # Always surface-sample so the kd-tree is dense in 3D regardless of
        # how the mesh was tessellated. Vertex-only fallback is reserved for
        # the degenerate case of a face-less mesh (i.e. a point cloud).
        if len(mesh.faces) > 0:
            np.random.seed(seed)
            samples, _ = trimesh.sample.sample_surface(mesh, n_samples)
            pts = np.asarray(samples, dtype=np.float64)
        else:
            pts = np.asarray(mesh.vertices, dtype=np.float64)
        self.points = pts
        self.tree = cKDTree(pts)

    def closest_points(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (closest_mesh_points, distances) for each query point."""
        dists, idx = self.tree.query(queries, k=1)
        return self.points[idx], dists


# ───────────────────────── Rigid alignment ─────────────────────────

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form rigid alignment.

    Finds (R, t) minimizing sum_i || (R @ P_i + t) - Q_i ||^2 over
    rotations R in SO(3) and translations t in R^3. Returns the solution.

    Equivalently for row-stacked inputs the predicted target is
        Q_pred = P @ R.T + t.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    return R, t


def point_to_mesh_icp(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    R0: Optional[np.ndarray] = None,
    t0: Optional[np.ndarray] = None,
    max_iter: int = 40,
    tol: float = 1e-7,
    outlier_quantile: float = 0.85,
    max_points: int = 4000,
    mesh_kdtree: Optional[MeshKDTree] = None,
    mesh_samples: int = 8000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Align `mesh` to the observed `points` via point-to-point ICP.

    Each iteration:
      1. Map `points` into mesh-local frame: p_local = (points - t) @ R
         (since the forward transform is world = R @ mesh_local + t).
      2. Query each p_local for its closest mesh surface sample (kd-tree).
      3. Drop the worst `1 - outlier_quantile` of correspondences by distance.
      4. Solve `kabsch(closest_pts, points)` on the inlier set.

    The mesh kd-tree is built once and can be passed in via `mesh_kdtree` so
    the multi-start search at frame 0 doesn't repeat the construction cost.

    Returns `(R, t, mean_inlier_residual_in_meters)`.
    """
    R = np.eye(3) if R0 is None else np.asarray(R0, dtype=np.float64).copy()
    t = np.zeros(3) if t0 is None else np.asarray(t0, dtype=np.float64).copy()

    points = np.asarray(points, dtype=np.float64)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), max_points, replace=False)]

    kd = mesh_kdtree if mesh_kdtree is not None else MeshKDTree(mesh, n_samples=mesh_samples)

    prev_err = np.inf
    for _ in range(max_iter):
        p_local = (points - t) @ R
        closest_pts, dists = kd.closest_points(p_local)
        thresh = float(np.quantile(dists, outlier_quantile))
        inliers = dists < thresh
        if int(inliers.sum()) < 10:
            break
        R_new, t_new = kabsch(closest_pts[inliers], points[inliers])
        err = float(dists[inliers].mean())
        R, t = R_new, t_new
        if abs(prev_err - err) < tol:
            prev_err = err
            break
        prev_err = err

    return R, t, float(prev_err)


def _initial_pose_search(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    n_yaw_samples: int = 8,
    mesh_kdtree: Optional[MeshKDTree] = None,
    mesh_samples: int = 8000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Multi-start ICP for frame 0.

    Tries n_yaw_samples evenly-spaced yaw rotations around +Z (gravity axis
    from Step 5), each with translation initialized to align centroids.
    Returns the (R, t, residual) of the seed with the lowest final residual.
    """
    kd = mesh_kdtree if mesh_kdtree is not None else MeshKDTree(mesh, n_samples=mesh_samples)
    point_centroid = points.mean(axis=0)
    mesh_centroid = np.asarray(mesh.centroid, dtype=np.float64)

    best = (np.eye(3), point_centroid - mesh_centroid, np.inf)
    for yaw in np.linspace(0.0, 2.0 * np.pi, n_yaw_samples, endpoint=False):
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        R_init = np.array([[c, -s, 0.0],
                           [s,  c, 0.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        t_init = point_centroid - R_init @ mesh_centroid
        R, t, err = point_to_mesh_icp(mesh, points, R0=R_init, t0=t_init, mesh_kdtree=kd)
        if err < best[2]:
            best = (R, t, err)
    return best


# ───────────────────────── Quaternion utilities ─────────────────────────

def quat_xyzw_from_R(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion in [x, y, z, w] order."""
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()


def enforce_quat_continuity(q_prev: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Negate `q_cur` if it lies in the opposite hemisphere from `q_prev`.

    A unit quaternion and its negation represent the same rotation, but
    numerical solvers can flip sign between adjacent frames, which would
    inject spurious 360-degree-style discontinuities into Step 7's
    quaternion-distance reward.
    """
    return -q_cur if float(np.dot(q_prev, q_cur)) < 0 else q_cur
