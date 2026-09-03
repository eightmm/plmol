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


class TestBurialIndexFreeSasaReuse:
    """compute_burial_index may reuse a file-based FreeSASA result."""

    @staticmethod
    def _atoms(pdb_path):
        from plmol.parsers import PDBParser

        atoms = PDBParser(pdb_path).protein_atoms
        positions = np.array([a.coords for a in atoms], dtype=np.float64)
        res_names = [a.res_name for a in atoms]
        atom_names = [a.atom_name for a in atoms]
        return positions, res_names, atom_names, len(atoms)

    @staticmethod
    def _standardized(pdb_path):
        """The path the featurizers actually hand to compute_burial_index."""
        from plmol.protein.featurizer import ProteinFeaturizer

        featurizer = ProteinFeaturizer(pdb_path)
        return featurizer, featurizer.tmp_pdb or featurizer.input_file

    def test_reuse_matches_the_atom_by_atom_path(self, example_pdb):
        from plmol.utils import compute_burial_index

        featurizer, path = self._standardized(example_pdb)
        positions, res_names, atom_names, n = self._atoms(path)
        built = compute_burial_index(positions, res_names, atom_names, n)
        reused = compute_burial_index(
            positions, res_names, atom_names, n, pdb_file=path
        )
        assert reused.shape == built.shape
        assert np.allclose(reused, built, atol=1e-6)
        del featurizer

    def test_reuse_is_taken_for_a_standardized_file(self, example_pdb):
        from plmol.utils import _burial_index_from_file

        featurizer, path = self._standardized(example_pdb)
        _, res_names, atom_names, n = self._atoms(path)
        assert _burial_index_from_file(path, res_names, atom_names, n) is not None
        del featurizer

    def test_guard_rejects_a_raw_pdb_with_extra_records(self, example_pdb):
        """FreeSASA reads records the parser drops, so the atom lists differ."""
        from plmol.utils import _burial_index_from_file

        _, res_names, atom_names, n = self._atoms(example_pdb)
        assert _burial_index_from_file(example_pdb, res_names, atom_names, n) is None

    def test_guard_rejects_mismatched_names(self, example_pdb):
        from plmol.utils import _burial_index_from_file

        featurizer, path = self._standardized(example_pdb)
        _, res_names, atom_names, n = self._atoms(path)
        renamed = list(atom_names)
        renamed[0] = "ZZZ"
        assert _burial_index_from_file(path, res_names, renamed, n) is None
        del featurizer

    def test_falls_back_when_the_path_is_unusable(self, example_pdb, tmp_path):
        from plmol.utils import compute_burial_index

        positions, res_names, atom_names, n = self._atoms(example_pdb)
        built = compute_burial_index(positions, res_names, atom_names, n)
        fallback = compute_burial_index(
            positions, res_names, atom_names, n, pdb_file=str(tmp_path / "missing.pdb")
        )
        assert np.allclose(fallback, built)


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
    """The numpy spellings must agree with the torch ones they replace."""

    def test_knn_mask_matches_the_torch_version(self):
        from plmol.utils import knn_mask

        rng = np.random.default_rng(0)
        distances = (rng.random((40, 40)) * 10).astype(np.float32)
        distances = (distances + distances.T) / 2
        for k in (1, 5, 20, 39, 100):
            assert np.array_equal(
                knn_mask(distances, k),
                knn_mask_torch(torch.from_numpy(distances), k).numpy(),
            ), f"k={k}"

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

    def test_dense_to_edges_matches_the_torch_version(self):
        from plmol.utils import dense_to_edges, dense_to_edges_torch

        rng = np.random.default_rng(2)
        adjacency = (rng.random((20, 20, 3)) * (rng.random((20, 20, 1)) > 0.7)).astype(np.float32)
        src, dst, values = dense_to_edges(adjacency)
        t_src, t_dst, t_values = dense_to_edges_torch(torch.from_numpy(adjacency))
        assert np.array_equal(src, t_src.numpy())
        assert np.array_equal(dst, t_dst.numpy())
        assert np.array_equal(values, t_values.numpy())

    def test_dense_to_edges_handles_a_two_dimensional_adjacency(self):
        from plmol.utils import dense_to_edges

        adjacency = np.array([[0.0, 1.5], [0.0, 0.0]], dtype=np.float32)
        src, dst, values = dense_to_edges(adjacency)
        assert src.tolist() == [0] and dst.tolist() == [1] and values.tolist() == [1.5]
