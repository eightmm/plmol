"""Tests for plmol/utils.py — kNN mask utilities."""

import numpy as np
import torch

from plmol.utils import knn_mask_torch, knn_mask_bipartite_numpy


class TestKnnMaskTorch:
    def test_basic_square(self):
        """k nearest neighbors selected correctly."""
        dm = torch.tensor([
            [0.0, 1.0, 3.0, 5.0],
            [1.0, 0.0, 2.0, 4.0],
            [3.0, 2.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ])
        mask = knn_mask_torch(dm, k=2)
        assert mask.shape == (4, 4)
        assert mask.dtype == torch.bool
        # Each row should have exactly 2 True values
        assert (mask.sum(dim=1) == 2).all()

    def test_k_exceeds_n(self):
        """k > n-1 is clamped."""
        dm = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        mask = knn_mask_torch(dm, k=10)
        # k clamped to 1 (n-1=1)
        assert mask.sum() == 2  # each row has 1 neighbor

    def test_single_node(self):
        dm = torch.tensor([[0.0]])
        mask = knn_mask_torch(dm, k=1)
        assert mask.shape == (1, 1)
        assert mask.sum() == 0  # no neighbors for single node

    def test_symmetric_distance(self):
        """For symmetric distance matrix, mask may not be symmetric (kNN is directional)."""
        n = 5
        coords = torch.randn(n, 3)
        dm = torch.cdist(coords, coords)
        mask = knn_mask_torch(dm, k=2)
        assert mask.shape == (n, n)
        assert (mask.sum(dim=1) == 2).all()


class TestKnnMaskBipartiteNumpy:
    def test_basic_bipartite(self):
        """Row and column nearest neighbors are combined."""
        dm = np.array([
            [1.0, 2.0, 5.0],
            [3.0, 1.0, 4.0],
        ])
        mask = knn_mask_bipartite_numpy(dm, k=1)
        assert mask.shape == (2, 3)
        assert mask.dtype == bool
        # At least k=1 per row and per column
        assert mask.sum(axis=1).min() >= 1
        assert mask.sum(axis=0).min() >= 1

    def test_row_neighbors_selected(self):
        """Verify the row-direction selects the k nearest columns."""
        dm = np.array([
            [1.0, 10.0, 20.0],
            [20.0, 1.0, 10.0],
        ])
        mask = knn_mask_bipartite_numpy(dm, k=1)
        # Row 0 nearest col is col 0, Row 1 nearest col is col 1
        assert mask[0, 0] is True or mask[0, 0] == True
        assert mask[1, 1] is True or mask[1, 1] == True

    def test_k_one(self):
        """k=1 should select nearest per row and column."""
        dm = np.array([
            [1.0, 2.0, 5.0],
            [3.0, 1.0, 4.0],
        ])
        mask = knn_mask_bipartite_numpy(dm, k=1)
        assert mask.shape == (2, 3)
        assert mask.sum(axis=1).min() >= 1
