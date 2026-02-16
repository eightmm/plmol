"""Tests for plmol/protein/atom_featurizer.py."""

import torch
import pytest

from plmol.protein.atom_featurizer import AtomFeaturizer
from plmol.protein.utils import PDBParser


class TestAtomFeaturizerMini:
    def test_get_protein_atom_features(self, mini_pdb):
        PDBParser.clear_cache()
        af = AtomFeaturizer()
        token, coord = af.get_protein_atom_features(mini_pdb)
        assert isinstance(token, torch.Tensor)
        assert isinstance(coord, torch.Tensor)
        assert token.ndim == 1
        assert coord.ndim == 2
        assert coord.shape[1] == 3
        assert token.shape[0] == coord.shape[0]

    def test_from_parser(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        af = AtomFeaturizer()
        token, coord = af.get_protein_atom_features_from_parser(parser)
        assert token.shape[0] > 0

    def test_get_all_atom_features(self, mini_pdb):
        PDBParser.clear_cache()
        af = AtomFeaturizer()
        features = af.get_all_atom_features(mini_pdb)
        assert isinstance(features, dict)
        expected_keys = ["token", "coords", "sasa", "relative_sasa", "residue_token"]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_get_atom_sasa(self, mini_pdb):
        PDBParser.clear_cache()
        af = AtomFeaturizer()
        sasa, rel_sasa_dict = af.get_atom_sasa(mini_pdb)
        assert isinstance(sasa, torch.Tensor)
        assert sasa.ndim == 1
        assert isinstance(rel_sasa_dict, dict)


class TestAtomFeaturizerReal:
    def test_get_all_atom_features_real(self, example_pdb):
        PDBParser.clear_cache()
        af = AtomFeaturizer()
        features = af.get_all_atom_features(example_pdb)
        assert features["token"].shape[0] > 100
        assert features["coords"].shape == (features["token"].shape[0], 3)

    def test_token_values(self, example_pdb):
        PDBParser.clear_cache()
        af = AtomFeaturizer()
        features = af.get_all_atom_features(example_pdb)
        # Tokens should be non-negative integers
        assert (features["token"] >= 0).all()
        # Check additional feature keys
        assert "sasa" in features
        assert "is_backbone" in features
