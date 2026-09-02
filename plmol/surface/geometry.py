"""Curvature computation for surface point clouds."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .mapping import _normalize_to_range, CURVATURE_SCALES

logger = logging.getLogger(__name__)


# Points per curvature block. Blocks are both the memory bound and the unit
# of parallelism, so this trades per-call overhead against how evenly the
# work spreads: on a 14.8k-point cloud, 1024 measured fastest (44 ms against
# 80 ms at 4096 and 46 ms at 256).
_CURVATURE_CHUNK = 1024


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
    k = _adaptive_k(radius, n)

    # Points are processed in chunks. Each point's neighbourhood is independent,
    # so the result is identical to one pass, but the (N, K, 3) neighbour and
    # centred arrays would otherwise reach tens of megabytes per scale.
    for start in range(0, n, _CURVATURE_CHUNK):
        stop = min(start + _CURVATURE_CHUNK, n)
        _, knn_idx = tree.query(points[start:stop], k=k, workers=-1)
        if knn_idx.ndim == 1:
            knn_idx = knn_idx[:, None]
        _pca_curvature_from_neighbours(
            points, knn_idx, mean_curv[start:stop], gauss_curv[start:stop]
        )

    # Normalisation is global, so it runs once over the assembled arrays.
    return _normalize_to_range(mean_curv), _normalize_to_range(gauss_curv)


def _adaptive_k(radius: float, n: int) -> int:
    """Neighbour count for a curvature scale: finer radius, fewer neighbours."""
    return max(6, min(int(radius * 8), n // 2))


def _pca_curvature_from_neighbours(
    points: np.ndarray,
    knn_idx: np.ndarray,
    out_mean: np.ndarray,
    out_gauss: np.ndarray,
) -> None:
    """Write raw curvature proxies for one block of points into the outputs.

    Args:
        points: Full (N, 3) cloud.
        knn_idx: (block, K) neighbour indices into ``points``.
        out_mean, out_gauss: (block,) views written in place.
    """
    k = knn_idx.shape[1]

    # Vectorised PCA: batch covariance for this block
    neighbours = points[knn_idx]                       # (block, K, 3)
    centroid = neighbours.mean(axis=1, keepdims=True)  # (block, 1, 3)
    centered = neighbours - centroid                   # (block, K, 3)

    # Covariance matrices: (block, 3, 3)
    covs = np.einsum('nki,nkj->nij', centered, centered) / k

    # Batch eigenvalues: (block, 3) ascending
    eigvals = np.linalg.eigvalsh(covs)
    eigvals = np.maximum(eigvals, 0.0)

    total = eigvals.sum(axis=1)  # (block,)
    valid = total > 1e-12

    # Mean curvature proxy
    out_mean[valid] = eigvals[valid, 0] / total[valid]
    # Gaussian curvature proxy
    out_gauss[valid] = (
        eigvals[valid, 0] * eigvals[valid, 1]
    ) / (total[valid] ** 2)


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
        # kNN results are sorted by distance, so the k nearest neighbours are a
        # prefix of the k_max nearest. One query per block therefore serves
        # every scale, instead of one query per scale.
        neighbour_counts = [_adaptive_k(radius, n_verts) for radius in curvature_scales]
        k_max = max(neighbour_counts)

        blocks = [
            (start, min(start + _CURVATURE_CHUNK, n_verts))
            for start in range(0, n_verts, _CURVATURE_CHUNK)
        ]

        def process_block(bounds: tuple[int, int]) -> None:
            start, stop = bounds
            _, knn_idx = pc_tree.query(points[start:stop], k=k_max, workers=1)
            if knn_idx.ndim == 1:
                knn_idx = knn_idx[:, None]
            for scale, k in enumerate(neighbour_counts):
                _pca_curvature_from_neighbours(
                    points,
                    knn_idx[:, :k],
                    mean_curvatures[start:stop, scale],
                    gauss_curvatures[start:stop, scale],
                )

        # Blocks, not scales, are the unit of parallelism: the largest scale
        # alone took as long as the other four together, so splitting by scale
        # left most workers idle.
        max_workers = min(len(blocks), (os.cpu_count() or 4))
        try:
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(executor.map(process_block, blocks))
            else:
                for bounds in blocks:
                    process_block(bounds)
        except Exception as e:  # pragma: no cover - defensive, matches old behaviour
            logger.warning("PCA curvature failed: %s", e)

        # Normalisation is per scale and global over the cloud.
        for scale in range(n_scales):
            mean_curvatures[:, scale] = _normalize_to_range(mean_curvatures[:, scale])
            gauss_curvatures[:, scale] = _normalize_to_range(gauss_curvatures[:, scale])

    return {
        'mean_curvature': mean_curvatures,
        'gaussian_curvature': gauss_curvatures,
        'vertex_normal': vertex_normals,
    }
