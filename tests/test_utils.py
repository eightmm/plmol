"""Tests for plmol/utils.py — kNN mask utilities."""

import numpy as np

from plmol.utils import knn_mask, knn_mask_bipartite_numpy


class TestKnnMask:
    def test_basic_square(self):
        """k nearest neighbors selected correctly."""
        dm = np.array([
            [0.0, 1.0, 3.0, 5.0],
            [1.0, 0.0, 2.0, 4.0],
            [3.0, 2.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ])
        mask = knn_mask(dm, k=2)
        assert mask.shape == (4, 4)
        assert mask.dtype == np.bool_
        # Each row should have exactly 2 True values
        assert (mask.sum(axis=1) == 2).all()

    def test_k_exceeds_n(self):
        """k > n-1 is clamped."""
        dm = np.array([[0.0, 1.0], [1.0, 0.0]])
        mask = knn_mask(dm, k=10)
        # k clamped to 1 (n-1=1)
        assert mask.sum() == 2  # each row has 1 neighbor

    def test_single_node(self):
        dm = np.array([[0.0]])
        mask = knn_mask(dm, k=1)
        assert mask.shape == (1, 1)
        assert mask.sum() == 0  # no neighbors for single node

    def test_symmetric_distance(self):
        """For symmetric distance matrix, mask may not be symmetric (kNN is directional)."""
        n = 5
        coords = np.random.default_rng(3).standard_normal((n, 3)).astype(np.float32)
        dm = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        mask = knn_mask(dm, k=2)
        assert mask.shape == (n, n)
        assert (mask.sum(axis=1) == 2).all()


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

    def test_rectangular_many_rows(self):
        dm = np.arange(20, dtype=np.float32).reshape(5, 4)
        mask = knn_mask_bipartite_numpy(dm, k=2)
        assert mask.shape == (5, 4)
        assert mask.sum(axis=1).min() >= 2
        assert mask.sum(axis=0).min() >= 2

    def test_zero_k(self):
        dm = np.ones((3, 2), dtype=np.float32)
        mask = knn_mask_bipartite_numpy(dm, k=0)
        assert mask.shape == (3, 2)
        assert not mask.any()


class TestBurialIndex:
    """burial_index is 1 - sasa/RESIDUE_MAX_SASA, from plmol's own areas."""

    @staticmethod
    def _atoms(pdb_path):
        from plmol.parsers import PDBParser

        atoms = PDBParser(pdb_path).protein_atoms
        positions = np.array([a.coords for a in atoms], dtype=np.float64)
        return (positions, [a.res_name for a in atoms],
                [a.atom_name for a in atoms], len(atoms))

    def test_it_spans_a_real_range(self, example_pdb):
        from plmol.utils import compute_burial_index

        burial = compute_burial_index(*self._atoms(example_pdb))
        assert burial.shape[0] > 0
        assert 0.0 <= burial.min() and burial.max() <= 1.0
        assert burial.std() > 0.01, "a near-constant column is the old bug"

    def test_the_pdb_file_argument_no_longer_changes_the_answer(self, example_pdb):
        """It used to select a reuse path; the areas are cached on the
        coordinates now, so it is kept only for callers written against 0.3.x."""
        from plmol.utils import compute_burial_index

        atoms = self._atoms(example_pdb)
        assert np.array_equal(
            compute_burial_index(*atoms),
            compute_burial_index(*atoms, pdb_file=example_pdb),
        )

    def test_no_coordinates_gives_the_neutral_value(self):
        from plmol.utils import compute_burial_index

        assert np.array_equal(
            compute_burial_index(None, [], [], 4), np.full(4, 0.5, dtype=np.float32)
        )

class TestDihedralAngles:
    """One batched 4-point dihedral, shared by the protein and nucleic paths."""

    def test_matches_a_known_angle(self):
        from plmol.utils import dihedral_angles

        # Planar trans arrangement: 180 degrees.
        p0 = np.array([[1.0, 1.0, 0.0]])
        p1 = np.array([[0.0, 1.0, 0.0]])
        p2 = np.array([[0.0, 0.0, 0.0]])
        p3 = np.array([[-1.0, 0.0, 0.0]])
        assert np.isclose(abs(dihedral_angles(p0, p1, p2, p3)[0]), np.pi)

    def test_degenerate_central_bond_is_zero(self):
        from plmol.utils import dihedral_angles

        p = np.zeros((1, 3))
        assert dihedral_angles(np.ones((1, 3)), p, p, np.ones((1, 3)))[0] == 0.0

    def test_batched_matches_one_at_a_time(self):
        from plmol.utils import dihedral_angles

        rng = np.random.default_rng(0)
        pts = rng.normal(size=(4, 32, 3))
        batched = dihedral_angles(*pts)
        one_by_one = np.array(
            [dihedral_angles(*[p[i][None, :] for p in pts])[0] for i in range(32)]
        )
        assert np.allclose(batched, one_by_one)

    def test_nucleic_and_protein_helpers_agree(self):
        from plmol.nucleic_acid.featurizer import _dihedral
        from plmol.protein.atom_featurizer import AtomFeaturizer

        rng = np.random.default_rng(1)
        for _ in range(20):
            p = rng.normal(size=(4, 3))
            scalar = _dihedral(*p)
            batched = float(AtomFeaturizer._dihedral_angles(*[p[i][None, :] for i in range(4)])[0])
            assert np.isclose(scalar, batched, atol=1e-12)


class TestNumpyNeighbourHelpers:
    """What the torch versions used to guarantee, now asserted directly."""

    def test_knn_mask_selects_the_k_nearest(self):
        from plmol.utils import knn_mask

        rng = np.random.default_rng(0)
        distances = (rng.random((40, 40)) * 10).astype(np.float32)
        distances = (distances + distances.T) / 2
        for k in (1, 5, 20, 39, 100):
            mask = knn_mask(distances, k)
            expected_k = min(k, 39)
            assert (mask.sum(axis=1) == expected_k).all(), f"k={k}"
            working = distances.copy()
            np.fill_diagonal(working, np.inf)
            for row in range(40):
                chosen = working[row][mask[row]]
                rejected = working[row][~mask[row]]
                assert chosen.max() <= rejected.min(), f"k={k}, row={row}"

    def test_knn_mask_never_selects_the_diagonal(self):
        from plmol.utils import knn_mask

        rng = np.random.default_rng(1)
        distances = (rng.random((12, 12))).astype(np.float32)
        np.fill_diagonal(distances, 0.0)
        assert not np.diag(knn_mask(distances, 3)).any()

    def test_knn_mask_leaves_its_input_alone(self):
        from plmol.utils import knn_mask

        distances = np.ones((5, 5), dtype=np.float32)
        knn_mask(distances, 2)
        assert np.array_equal(distances, np.ones((5, 5), dtype=np.float32))

    def test_dense_to_edges_finds_every_nonzero_pair(self):
        from plmol.utils import dense_to_edges

        rng = np.random.default_rng(2)
        adjacency = (rng.random((20, 20, 3)) * (rng.random((20, 20, 1)) > 0.7)).astype(np.float32)
        src, dst, values = dense_to_edges(adjacency)
        expected = np.argwhere(adjacency.any(axis=-1))
        assert np.array_equal(np.stack([src, dst], axis=1), expected)
        assert np.array_equal(values, adjacency[src, dst])

    def test_dense_to_edges_handles_a_two_dimensional_adjacency(self):
        from plmol.utils import dense_to_edges

        adjacency = np.array([[0.0, 1.5], [0.0, 0.0]], dtype=np.float32)
        src, dst, values = dense_to_edges(adjacency)
        assert src.tolist() == [0] and dst.tolist() == [1] and values.tolist() == [1.5]
