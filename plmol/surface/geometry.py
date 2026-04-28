"""Curvature computation for surface point clouds."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .mapping import _normalize_to_range, CURVATURE_SCALES

logger = logging.getLogger(__name__)


def _compute_pca_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    tree: Optional[cKDTree] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate mean and Gaussian curvature from a point cloud via local PCA.

    Uses a **KNN-based** approach where K is adapted to the scale *radius*.
    This is more robust than a fixed-radius ball query because it guarantees
    a minimum number of neighbours even at small scales, and avoids the
    "everything inside" degeneracy at large scales.

    Heuristic: ``K = clamp(int(radius * 8), 6, N//2)``
    – smaller radius → fewer neighbours (fine detail)
    – larger radius → more neighbours (coarse curvature)

    Eigenvalue ratios of the local 3×3 covariance matrix:
        lambda_0 <= lambda_1 <= lambda_2

    * **mean curvature proxy** – ``lambda_0 / total``
    * **Gaussian curvature proxy** – ``lambda_0 * lambda_1 / total²``

    Both are normalised to [-1, 1] before return.

    Args:
        points: (N, 3) surface positions.
        normals: (N, 3) surface normals.
        radius: curvature scale (Å) – controls K.
        tree: optional pre-built cKDTree of *points*.

    Returns:
        (mean_curv, gauss_curv) each (N,), normalised to [-1, 1].
    """
    n = len(points)
    mean_curv = np.zeros(n, dtype=np.float32)
    gauss_curv = np.zeros(n, dtype=np.float32)

    if n < 6:
        return mean_curv, gauss_curv

    if tree is None:
        tree = cKDTree(points)

    # Adaptive K based on scale
    k = max(6, min(int(radius * 8), n // 2))
    _, knn_idx = tree.query(points, k=k, workers=-1)
    if knn_idx.ndim == 1:
        knn_idx = knn_idx[:, None]

    # Vectorised PCA: batch covariance for all points
    neighbours = points[knn_idx]                     # (N, K, 3)
    centroid = neighbours.mean(axis=1, keepdims=True)  # (N, 1, 3)
    centered = neighbours - centroid                    # (N, K, 3)

    # Covariance matrices: (N, 3, 3)
    covs = np.einsum('nki,nkj->nij', centered, centered) / k

    # Batch eigenvalues: (N, 3) ascending
    eigvals = np.linalg.eigvalsh(covs)
    eigvals = np.maximum(eigvals, 0.0)

    total = eigvals.sum(axis=1)  # (N,)
    valid = total > 1e-12

    # Mean curvature proxy
    mean_curv[valid] = eigvals[valid, 0] / total[valid]

    # Gaussian curvature proxy
    gauss_curv[valid] = (
        eigvals[valid, 0] * eigvals[valid, 1]
    ) / (total[valid] ** 2)

    return _normalize_to_range(mean_curv), _normalize_to_range(gauss_curv)


def compute_pointcloud_geometry(
    points: np.ndarray,
    normals: np.ndarray,
    curvature_scales: tuple[float, ...] = CURVATURE_SCALES,
    verbose: bool = False,
) -> dict:
    """Compute geometry features from a point cloud.

    Uses PCA-based curvature estimation (adaptive KNN per scale)
    instead of mesh-based discrete curvature.

    Can be used independently for point-cloud-based surface analysis.

    Args:
        points: Point cloud positions (N, 3)
        normals: Point normals (N, 3)
        curvature_scales: Radii for multi-scale curvature computation
        verbose: Whether to print progress messages

    Returns:
        Dict with keys:
            - 'mean_curvature': (N, n_scales) normalized to [-1, 1]
            - 'gaussian_curvature': (N, n_scales) normalized to [-1, 1]
            - 'vertex_normal': (N, 3) unit vectors
    """
    n_verts = len(points)
    n_scales = len(curvature_scales)
    mean_curvatures = np.zeros((n_verts, n_scales), dtype=np.float32)
    gauss_curvatures = np.zeros((n_verts, n_scales), dtype=np.float32)
    vertex_normals = np.asarray(normals, dtype=np.float32)

    if n_verts >= 4:
        if verbose:
            logger.debug("Computing PCA curvature for point cloud")
        pc_tree = cKDTree(points)
        with ThreadPoolExecutor(max_workers=n_scales) as executor:
            futures = {
                executor.submit(
                    _compute_pca_curvature, points, normals, radius, pc_tree
                ): i
                for i, radius in enumerate(curvature_scales)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    mc, gc = future.result()
                    mean_curvatures[:, i] = mc
                    gauss_curvatures[:, i] = gc
                except Exception as e:
                    logger.warning("PCA curvature at radius=%s failed: %s", curvature_scales[i], e)

    return {
        'mean_curvature': mean_curvatures,
        'gaussian_curvature': gauss_curvatures,
        'vertex_normal': vertex_normals,
    }
