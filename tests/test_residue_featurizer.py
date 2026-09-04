"""Tests for plmol/protein/residue_featurizer.py."""

import numpy as np
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
        assert n_term.dtype == np.bool_
        assert c_term.dtype == np.bool_
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
        assert all(isinstance(t, np.ndarray) for t in scalar)

    def test_calculate_sasa(self, example_pdb):
        PDBParser.clear_cache()
        rf = ResidueFeaturizer(example_pdb)
        sasa = rf.calculate_sasa()
        assert isinstance(sasa, np.ndarray)
        assert sasa.ndim == 2
        assert sasa.shape[1] == 11


class TestAtomOrderInsideResidue:
    """A PDB may list a residue's atoms in any order; the features may not move."""

    @staticmethod
    def _shuffled(source: str, target: str, seed: int = 0) -> str:
        """Copy ``source`` to ``target`` with each residue's atoms reordered."""
        import random

        head, body, tail, seen_body = [], [], [], False
        for line in open(source):
            line = line.rstrip("\n")
            if line.startswith(("ATOM  ", "HETATM")):
                body.append(line)
                seen_body = True
            elif seen_body:
                tail.append(line)
            else:
                head.append(line)

        rng = random.Random(seed)
        out, block, key = [], [], None
        for line in body + [None]:
            k = None if line is None else (line[21], line[22:26], line[26])
            if k != key and block:
                rng.shuffle(block)
                out += block
                block = []
            key = k
            if line is not None:
                block.append(line)
        out += block

        renumbered = [l[:6] + f"{i + 1:5d}" + l[11:] for i, l in enumerate(out)]
        with open(target, "w") as fh:
            fh.write("\n".join(head + renumbered + tail) + "\n")
        return target

    def test_features_survive_a_shuffled_residue(self, example_pdb, tmp_path):
        # standardize=False is the path under test: the standardizer rewrites the
        # atoms into canonical order and would hide this.
        from plmol import Protein

        PDBParser.clear_cache()
        shuffled = self._shuffled(example_pdb, str(tmp_path / "shuffled.pdb"))

        reference = Protein.from_pdb(example_pdb, standardize=False).featurize(mode="graph")["graph"]
        moved = Protein.from_pdb(shuffled, standardize=False).featurize(mode="graph")["graph"]

        assert np.array_equal(reference["coords"], moved["coords"])
        assert np.array_equal(reference["edge_index"], moved["edge_index"])
        for a, b in zip(reference["node_features"], moved["node_features"]):
            assert np.allclose(a, b, equal_nan=True)
        for a, b in zip(reference["node_vector_features"], moved["node_vector_features"]):
            assert np.allclose(a, b, equal_nan=True)

    def test_the_ca_row_is_the_ca(self, example_pdb, tmp_path):
        # coords[:, 0] is documented as the CA; row 1 of the coordinate cache is
        # what it comes from, so the cache has to be in residue order.
        from plmol import Protein

        PDBParser.clear_cache()
        shuffled = self._shuffled(example_pdb, str(tmp_path / "shuffled_ca.pdb"))
        rf = ResidueFeaturizer(shuffled)
        for residue in rf.get_residues()[:50]:
            named = rf._atom_coords[residue].get("CA")
            if named is None:
                continue
            assert np.array_equal(rf.get_residue_coordinates_numpy(residue)[1], named)
