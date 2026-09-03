"""Tests for plmol/protein/atom_featurizer.py."""

import numpy as np
import pytest

from plmol.protein.atom_featurizer import AtomFeaturizer
from plmol.protein.utils import PDBParser


class TestTheTwoEntryPointsAreOne:
    """get_protein_atom_features(path) and ..._from_parser(parser) tokenise the
    same atoms. The path one used to walk the file itself with a filter that
    dropped only water and hydrogen, so a metal or a ligand was handed back
    tokenised as protein."""

    STRUCTURE = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n"
        "ATOM      5  OXT ALA A   1       3.300   1.500   0.000  1.00  0.00           O\n"
        "HETATM    6 ZN    ZN A 301       8.000   8.000   8.000  1.00 20.00          ZN\n"
        "HETATM    7  C1  LIG A 401       5.000   5.000   5.000  1.00 20.00           C\n"
        "HETATM    8  O   HOH A 501      12.000  12.000  12.000  1.00 20.00           O\n"
        "END\n"
    )

    def test_a_metal_and_a_ligand_are_not_protein_atoms(self, tmp_path):
        path = tmp_path / "metal.pdb"
        path.write_text(self.STRUCTURE)
        PDBParser.clear_cache()
        token, coord = AtomFeaturizer().get_protein_atom_features(str(path))
        assert token.shape[0] == 4, "N, CA, C, O -- not the zinc, the ligand or OXT"
        assert coord.shape == (4, 3)

    def test_both_entry_points_agree(self, example_pdb):
        PDBParser.clear_cache()
        featurizer = AtomFeaturizer()
        by_path = featurizer.get_protein_atom_features(example_pdb)
        by_parser = featurizer.get_protein_atom_features_from_parser(PDBParser(example_pdb))
        assert np.array_equal(by_path[0], by_parser[0])
        assert np.array_equal(by_path[1], by_parser[1])


class TestAtomFeaturizerMini:
    def test_get_protein_atom_features(self, mini_pdb):
        PDBParser.clear_cache()
        af = AtomFeaturizer()
        token, coord = af.get_protein_atom_features(mini_pdb)
        assert isinstance(token, np.ndarray)
        assert isinstance(coord, np.ndarray)
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
        assert isinstance(sasa, np.ndarray)
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
