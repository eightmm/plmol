"""Shared utility functions for plmol."""

import numpy as np
import torch


def knn_mask_torch(dist_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """Square distance matrix -> kNN boolean mask."""
    dm = dist_matrix.clone()
    dm.fill_diagonal_(float('inf'))
    k = min(k, dm.size(0) - 1)
    _, topk_idx = torch.topk(dm, k, dim=1, largest=False)
    mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask


def knn_mask_bipartite_numpy(dm: np.ndarray, k: int) -> np.ndarray:
    """Bipartite (M, N) distance matrix -> kNN boolean mask.

    Each row's k nearest + each col's k nearest.
    """
    k_col = min(k, dm.shape[1])
    topk_col = np.argpartition(dm, k_col, axis=1)[:, :k_col]
    mask_row = np.zeros_like(dm, dtype=bool)
    np.put_along_axis(mask_row, topk_col, True, axis=1)

    k_row = min(k, dm.shape[0])
    topk_row = np.argpartition(dm.T, k_row, axis=1)[:, :k_row]
    mask_col = np.zeros_like(dm, dtype=bool)
    np.put_along_axis(mask_col, topk_row, True, axis=0)

    return mask_row | mask_col
