"""Tests for plmol/protein/protein_featurizer.py — main orchestrator."""

import torch
import pytest

from plmol.protein.protein_featurizer import ProteinFeaturizer
from plmol.protein.utils import PDBParser


class TestProteinFeaturizerMini:
    def test_init(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        assert pf is not None

    def test_get_sequence_features(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        seq = pf.get_sequence_features()
        assert isinstance(seq, dict)
        assert "residue_types" in seq
        assert "num_residues" in seq

    def test_get_sequence_by_chain(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        chains = pf.get_sequence_by_chain()
        assert isinstance(chains, dict)
        assert len(chains) == 2

    def test_get_features(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        node, edge = pf.get_features()
        assert isinstance(node, dict)
        assert isinstance(edge, dict)

    def test_get_node_features(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        node = pf.get_node_features()
        assert isinstance(node, dict)
        assert "scalar_features" in node
        assert "coords" in node

    def test_get_edge_features(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        edge = pf.get_edge_features()
        assert isinstance(edge, dict)

    def test_get_backbone(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        backbone = pf.get_backbone(k_neighbors=3)
        assert isinstance(backbone, dict)
        assert "backbone_coords" in backbone
        assert "edge_index" in backbone

    def test_get_contact_map(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        cm = pf.get_contact_map(distance_cutoff=8.0)
        assert isinstance(cm, dict)

    def test_get_terminal_flags(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        flags = pf.get_terminal_flags()
        assert isinstance(flags, dict)

    def test_get_sasa_features(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        sasa = pf.get_sasa_features()
        assert isinstance(sasa, torch.Tensor)

    def test_get_geometric_features(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        geo = pf.get_geometric_features()
        assert isinstance(geo, dict)

    def test_get_relative_position(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        rp = pf.get_relative_position()
        assert isinstance(rp, torch.Tensor)

    def test_get_atom_graph(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        node, edge = pf.get_atom_graph()
        assert isinstance(node, dict)
        assert isinstance(edge, dict)

    def test_get_atom_tokens_and_coords(self, mini_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(mini_pdb, standardize=False)
        tokens, coords = pf.get_atom_tokens_and_coords()
        assert isinstance(tokens, torch.Tensor)
        assert isinstance(coords, torch.Tensor)
        assert coords.shape[1] == 3


class TestProteinFeaturizerReal:
    def test_get_all_features(self, example_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(example_pdb)
        all_features = pf.get_all_features()
        assert isinstance(all_features, dict)
        assert "node" in all_features
        assert "edge" in all_features

    def test_standardize_enabled(self, example_pdb):
        PDBParser.clear_cache()
        pf = ProteinFeaturizer(example_pdb, standardize=True)
        seq = pf.get_sequence_features()
        assert seq["num_residues"] > 0
