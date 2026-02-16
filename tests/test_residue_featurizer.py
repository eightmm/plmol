"""Tests for plmol/protein/residue_featurizer.py."""

import torch
import pytest

from plmol.protein.residue_featurizer import ResidueFeaturizer
from plmol.protein.utils import PDBParser


class TestResidueFeaturizerMini:
    """Tests with mini PDB fixture."""

    def test_init_from_file(self, mini_pdb):
        PDBParser.clear_cache()
        rf = ResidueFeaturizer(mini_pdb)
        residues = rf.get_residues()
        assert len(residues) == 10

    def test_from_parser(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        rf = ResidueFeaturizer.from_parser(parser, mini_pdb)
        assert len(rf.get_residues()) == 10

    def test_get_sequence_by_chain(self, mini_pdb):
        PDBParser.clear_cache()
        rf = ResidueFeaturizer(mini_pdb)
        seq = rf.get_sequence_by_chain()
        assert "A" in seq
        assert "B" in seq

    def test_get_terminal_flags(self, mini_pdb):
        PDBParser.clear_cache()
        rf = ResidueFeaturizer(mini_pdb)
        n_term, c_term = rf.get_terminal_flags()
        assert n_term.dtype == torch.bool
        assert c_term.dtype == torch.bool
        assert n_term.shape[0] == 10
        # At least 2 n-terminals (one per chain)
        assert n_term.sum() >= 2
        assert c_term.sum() >= 2

    def test_get_relative_position(self, mini_pdb):
        PDBParser.clear_cache()
        rf = ResidueFeaturizer(mini_pdb)
        rel_pos = rf.get_relative_position(cutoff=32, onehot=True)
        assert rel_pos.shape[0] == 10
        assert rel_pos.shape[1] == 10

    def test_get_features(self, mini_pdb):
        PDBParser.clear_cache()
        rf = ResidueFeaturizer(mini_pdb)
        node_dict, edge_dict = rf.get_features()
        assert isinstance(node_dict, dict)
        assert isinstance(edge_dict, dict)
        assert "node_scalar_features" in node_dict
        assert "node_vector_features" in node_dict
        assert "coords" in node_dict


class TestResidueFeaturizerReal:
    """Tests with real 10gs PDB."""

    def test_get_features_real(self, example_pdb):
        PDBParser.clear_cache()
        rf = ResidueFeaturizer(example_pdb)
        node_dict, edge_dict = rf.get_features()
        # node_scalar_features is a tuple of tensors
        scalar = node_dict["node_scalar_features"]
        assert isinstance(scalar, tuple)
        assert all(isinstance(t, torch.Tensor) for t in scalar)

    def test_calculate_sasa(self, example_pdb):
        PDBParser.clear_cache()
        rf = ResidueFeaturizer(example_pdb)
        sasa = rf.calculate_sasa()
        assert isinstance(sasa, torch.Tensor)
        assert sasa.ndim == 2
        assert sasa.shape[1] == 10
