"""Shared KNN utilities for atom-to-vertex feature mapping."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..constants import (
    SURFACE_DEFAULT_CURVATURE_SCALES,
    SURFACE_DEFAULT_KNN_ATOMS,
)

# Re-export for convenience
CURVATURE_SCALES = SURFACE_DEFAULT_CURVATURE_SCALES
SURFACE_KNN_ATOMS = SURFACE_DEFAULT_KNN_ATOMS


def _normalize_to_range(arr: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Normalize array to [lo, hi] using robust min/max (1st/99th percentile).

    NaN/Inf values are replaced with 0.0 (midpoint of [-1, 1]).
    Constant arrays return all zeros.
    """
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)
    finite_vals = arr[finite_mask]
    p1, p99 = np.percentile(finite_vals, [1, 99])
    if p99 - p1 < 1e-8:
        return np.zeros_like(arr)
    clipped = np.clip(arr, p1, p99)
    scaled = (clipped - p1) / (p99 - p1)  # [0, 1]
    result = scaled * (hi - lo) + lo
    result[~finite_mask] = 0.0
    return result


def _build_knn_weights(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    k: int = SURFACE_KNN_ATOMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build KNN-based distance weights for atom-to-vertex mapping.

    Uses cKDTree for O(N*K) memory instead of O(N*M) full distance matrix.

    Args:
        verts: Mesh vertices (N, 3)
        atom_positions: Atom positions (M, 3)
        k: Number of nearest atoms per vertex

    Returns:
        knn_idx: (N, K) indices of K nearest atoms per vertex
        knn_weights: (N, K) normalized inverse-distance weights (rows sum to 1)
        knn_dists: (N, K) Euclidean distances to K nearest atoms
    """
    n_atoms = len(atom_positions)
    k = min(k, n_atoms)

    tree = cKDTree(atom_positions)
    knn_dists, knn_idx = tree.query(verts, k=k, workers=-1)

    # cKDTree.query returns 1D arrays when k=1; ensure 2D
    if k == 1:
        knn_dists = knn_dists[:, None]
        knn_idx = knn_idx[:, None]

    knn_dists = knn_dists.astype(np.float32)
    knn_idx = knn_idx.astype(np.intp)

    knn_dists_clamped = np.maximum(knn_dists, 0.5)
    knn_weights = 1.0 / knn_dists_clamped
    row_sums = knn_weights.sum(axis=1, keepdims=True)
    knn_weights = knn_weights / np.maximum(row_sums, 1e-8)

    return knn_idx, knn_weights, knn_dists


def _knn_map_scalar(
    knn_idx: np.ndarray,
    knn_weights: np.ndarray,
    atom_features: np.ndarray,
) -> np.ndarray:
    """Map per-atom scalar features to vertices via KNN weights.

    Args:
        knn_idx: (N, K) KNN atom indices
        knn_weights: (N, K) normalized weights
        atom_features: (M,) per-atom scalar feature

    Returns:
        (N,) per-vertex feature
    """
    return (knn_weights * atom_features[knn_idx]).sum(axis=1)


def _knn_map_matrix(
    knn_idx: np.ndarray,
    knn_weights: np.ndarray,
    atom_features: np.ndarray,
) -> np.ndarray:
    """Map per-atom vector/matrix features to vertices via KNN weights.

    Args:
        knn_idx: (N, K) KNN atom indices
        knn_weights: (N, K) normalized weights
        atom_features: (M, D) per-atom feature matrix

    Returns:
        (N, D) per-vertex feature matrix
    """
    # atom_features[knn_idx] -> (N, K, D)
    gathered = atom_features[knn_idx]
    # knn_weights[:, :, None] -> (N, K, 1) for broadcasting
    return (knn_weights[:, :, None] * gathered).sum(axis=1)
