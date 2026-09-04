"""Tests for plmol/protein/backbone_featurizer.py."""

import numpy as np

from plmol.protein.backbone_featurizer import (
    compute_backbone_dihedrals,
    build_backbone_knn_graph,
    compute_edge_frame_features,
    compute_backbone_features,
)
from plmol.protein.geometry import calculate_local_frames


def _make_coords(L: int, atoms_per_res: int = 5) -> np.ndarray:
    coords = np.zeros((L, atoms_per_res, 3), dtype=np.float32)
    for i in range(L):
        z = i * 3.8
        coords[i, 0] = [-0.5, 0.0, z - 1.0]  # N
        coords[i, 1] = [0.0, 0.0, z]          # CA
        coords[i, 2] = [0.5, 0.0, z + 0.5]    # C
        coords[i, 3] = [0.5, 1.0, z + 0.5]    # O
        coords[i, 4] = [0.0, 1.5, z]          # CB
    return coords


class TestComputeBackboneDihedrals:
    def test_single_chain(self):
        L = 6
        coords = _make_coords(L)
        chain_indices = {"A": list(range(L))}
        dihedrals, mask = compute_backbone_dihedrals(coords, chain_indices)
        assert dihedrals.shape == (L, 3)
        assert mask.shape == (L, 3)
        assert mask.dtype == np.bool_

    def test_multi_chain(self):
        L = 8
        coords = _make_coords(L)
        chain_indices = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
        dihedrals, mask = compute_backbone_dihedrals(coords, chain_indices)
        assert dihedrals.shape == (L, 3)

    def test_short_chain_skipped(self):
        L = 4
        coords = _make_coords(L)
        chain_indices = {"A": [0], "B": [1, 2, 3]}
        dihedrals, mask = compute_backbone_dihedrals(coords, chain_indices)
        # Chain A (single residue) should have no valid dihedrals
        assert mask[0].sum() == 0


class TestBuildBackboneKnnGraph:
    def test_basic(self):
        L = 10
        coords = _make_coords(L)
        k = 5
        graph = build_backbone_knn_graph(coords, k=k)
        assert graph["edge_index"].shape[0] == 2
        assert graph["edge_index"].shape[1] == L * k
        assert graph["edge_dist"].shape[0] == L * k
        assert graph["edge_unit_vec"].shape == (L * k, 3)
        assert graph["edge_seq_sep"].shape[0] == L * k
        assert graph["edge_same_chain"].shape[0] == L * k

    def test_k_clamped(self):
        L = 3
        coords = _make_coords(L)
        graph = build_backbone_knn_graph(coords, k=100)
        # k clamped to L-1 = 2
        assert graph["edge_index"].shape[1] == L * 2

    def test_chain_indices(self):
        L = 6
        coords = _make_coords(L)
        chain_indices = {"A": [0, 1, 2], "B": [3, 4, 5]}
        graph = build_backbone_knn_graph(coords, k=3, chain_indices=chain_indices)
        # Verify same_chain flag
        src = graph["edge_index"][0]
        dst = graph["edge_index"][1]
        for i in range(src.shape[0]):
            s, d = src[i].item(), dst[i].item()
            same = (s < 3 and d < 3) or (s >= 3 and d >= 3)
            assert graph["edge_same_chain"][i].item() == same

    def test_unit_vectors_normalized(self):
        coords = _make_coords(8)
        graph = build_backbone_knn_graph(coords, k=3)
        norms = np.linalg.norm(graph["edge_unit_vec"], axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestComputeEdgeFrameFeatures:
    def test_shape(self):
        L = 5
        coords = _make_coords(L)
        ca_coords = coords[:, 1]
        frames = calculate_local_frames(coords)
        # Make simple edge_index: fully connected minus self
        src = np.repeat(np.arange(L), L - 1)
        dst = np.concatenate([np.concatenate([np.arange(i), np.arange(i + 1, L)]) for i in range(L)])
        edge_index = np.stack([src, dst])
        E = edge_index.shape[1]
        result = compute_edge_frame_features(ca_coords, frames, edge_index)
        assert result["edge_local_pos"].shape == (E, 3)
        assert result["edge_rel_orient"].shape == (E, 3, 3)


class TestComputeBackboneFeatures:
    def test_full_assembly(self):
        L = 8
        coords = _make_coords(L)
        residues = [("A", i + 1, 0) for i in range(4)] + [("B", i + 1, 4) for i in range(4)]
        residue_types = np.zeros(L, dtype=np.int64)
        result = compute_backbone_features(coords, residues, residue_types, k_neighbors=3)

        assert result["backbone_coords"].shape == (L, 4, 3)
        assert result["cb_coords"].shape == (L, 3)
        assert result["dihedrals"].shape == (L, 3)
        assert result["dihedrals_sincos"].shape == (L, 6)
        assert result["dihedrals_mask"].shape == (L, 3)
        assert result["orientation_frames"].shape == (L, 3, 3)
        assert result["residue_types"].shape == (L,)
        assert result["chain_ids"].shape == (L,)
        assert result["residue_mask"].shape == (L,)
        assert result["edge_rbf"].shape[-1] == 16
        assert result["num_residues"] == L
        assert result["num_chains"] == 2
        assert result["k_neighbors"] == 3


class TestBackboneDihedralsStopAtAGap:
    """Consecutive within a chain is not the same as bonded."""

    @staticmethod
    def _gapped(example_pdb, tmp_path):
        kept = [
            line.rstrip("\n")
            for line in open(example_pdb)
            if line.startswith(("ATOM  ", "HETATM"))
            and not (line[21] == "A" and 101 <= int(line[22:26]) <= 110)
        ]
        path = tmp_path / "backbone_gap.pdb"
        path.write_text("\n".join(kept) + "\nEND\n")
        return str(path)

    def test_a_missing_loop_invalidates_three_angles(self, example_pdb, tmp_path):
        from plmol import Protein
        from plmol.parsers.pdb_parser import PDBParser

        PDBParser.clear_cache()
        intact = Protein.from_pdb(example_pdb, standardize=False).featurize(mode="backbone")["backbone"]
        gapped_path = self._gapped(example_pdb, tmp_path)
        gapped = Protein.from_pdb(gapped_path, standardize=False).featurize(mode="backbone")["backbone"]

        atoms = PDBParser(gapped_path).protein_atoms
        seen, order = set(), []
        for atom in atoms:
            key = (atom.chain_id, atom.res_num, atom.insertion_code)
            if key not in seen:
                seen.add(key)
                order.append(key)
        order.sort()
        before = order.index(("A", 100, ""))
        after = order.index(("A", 111, ""))
        assert after == before + 1

        mask = np.asarray(gapped["dihedrals_mask"])
        dihedrals = np.asarray(gapped["dihedrals"])
        # psi of 100, phi and omega of 111 all read across the break.
        assert not mask[before, 1]
        assert not mask[after, 0]
        assert not mask[after, 2]
        assert dihedrals[before, 1] == 0.0
        assert dihedrals[after, 0] == 0.0
        assert dihedrals[after, 2] == 0.0
        # The intact structure keeps them.
        assert np.asarray(intact["dihedrals_mask"])[:, 1].sum() > 400
