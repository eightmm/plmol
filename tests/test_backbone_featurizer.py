"""Tests for plmol/protein/backbone_featurizer.py."""

import torch

from plmol.protein.backbone_featurizer import (
    compute_backbone_dihedrals,
    build_backbone_knn_graph,
    compute_edge_frame_features,
    compute_backbone_features,
)
from plmol.protein.geometry import calculate_local_frames


def _make_coords(L: int, atoms_per_res: int = 5) -> torch.Tensor:
    coords = torch.zeros(L, atoms_per_res, 3)
    for i in range(L):
        z = i * 3.8
        coords[i, 0] = torch.tensor([-0.5, 0.0, z - 1.0])  # N
        coords[i, 1] = torch.tensor([0.0, 0.0, z])          # CA
        coords[i, 2] = torch.tensor([0.5, 0.0, z + 0.5])    # C
        coords[i, 3] = torch.tensor([0.5, 1.0, z + 0.5])    # O
        coords[i, 4] = torch.tensor([0.0, 1.5, z])          # CB
    return coords


class TestComputeBackboneDihedrals:
    def test_single_chain(self):
        L = 6
        coords = _make_coords(L)
        chain_indices = {"A": list(range(L))}
        dihedrals, mask = compute_backbone_dihedrals(coords, chain_indices)
        assert dihedrals.shape == (L, 3)
        assert mask.shape == (L, 3)
        assert mask.dtype == torch.bool

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
        norms = torch.norm(graph["edge_unit_vec"], dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestComputeEdgeFrameFeatures:
    def test_shape(self):
        L = 5
        coords = _make_coords(L)
        ca_coords = coords[:, 1]
        frames = calculate_local_frames(coords)
        # Make simple edge_index: fully connected minus self
        src = torch.arange(L).repeat_interleave(L - 1)
        dst = torch.cat([torch.cat([torch.arange(i), torch.arange(i + 1, L)]) for i in range(L)])
        edge_index = torch.stack([src, dst])
        E = edge_index.shape[1]
        result = compute_edge_frame_features(ca_coords, frames, edge_index)
        assert result["edge_local_pos"].shape == (E, 3)
        assert result["edge_rel_orient"].shape == (E, 3, 3)


class TestComputeBackboneFeatures:
    def test_full_assembly(self):
        L = 8
        coords = _make_coords(L)
        residues = [("A", i + 1, 0) for i in range(4)] + [("B", i + 1, 4) for i in range(4)]
        residue_types = torch.zeros(L, dtype=torch.long)
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
