"""Curvature computation for surface point clouds."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..spatial import NeighbourIndex
from .mapping import _normalize_to_range, CURVATURE_SCALES

logger = logging.getLogger(__name__)


# Points per curvature block. Blocks are both the memory bound and the unit
# of parallelism, so this trades per-call overhead against how evenly the
# work spreads. Measured on 14.8k- and 59k-point clouds: 512 and 1024 tie,
# 2048 is ~20% worse and 4096 ~70% worse. The 3x3 eigendecomposition does
# not engage BLAS threading, so this does not depend on OPENBLAS_NUM_THREADS.
_CURVATURE_CHUNK = 1024


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
        # kNN results are sorted by distance, so the k nearest neighbours are a
        # prefix of the k_max nearest. One query therefore serves every scale,
        # instead of one query per scale. It is also one query rather than one
        # per block: a neighbour search amortises its structure over the whole
        # cloud, and splitting the queries up threw that away.
        neighbour_counts = [_adaptive_k(radius, n_verts) for radius in curvature_scales]
        k_max = max(neighbour_counts)
        _, neighbours = NeighbourIndex(points).query(points, k=k_max)
        if neighbours.ndim == 1:
            neighbours = neighbours[:, None]

        blocks = [
            (start, min(start + _CURVATURE_CHUNK, n_verts))
            for start in range(0, n_verts, _CURVATURE_CHUNK)
        ]

        def process_block(bounds: tuple[int, int]) -> None:
            start, stop = bounds
            for scale, k in enumerate(neighbour_counts):
                _pca_curvature_from_neighbours(
                    points,
                    neighbours[start:stop, :k],
                    mean_curvatures[start:stop, scale],
                    gauss_curvatures[start:stop, scale],
                )

        # Blocks, not scales, are the unit of parallelism for the PCA: the
        # largest scale alone took as long as the other four together, so
        # splitting by scale left most workers idle.
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
