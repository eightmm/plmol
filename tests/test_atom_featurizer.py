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


def _glycine_chain(chain, numbers, x0=0.0):
    """ATOM lines for a run of glycines, one per (number, insertion code)."""
    lines = []
    for step, (number, icode) in enumerate(numbers):
        base = np.array([x0, step * 3.8, 0.0])
        for name, offset in (("N", (0.0, 0.0, 0.0)), ("CA", (1.45, 0.0, 0.0)),
                             ("C", (2.4, 1.0, 0.0)), ("O", (2.4, 2.2, 0.0))):
            x, y, z = base + np.array(offset)
            lines.append(
                f"ATOM      1  {name:<3s} GLY {chain}{number:4d}{icode:1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {name[0]}"
            )
    return lines


def _write(path, lines):
    numbered = [l[:6] + f"{i + 1:5d}" + l[11:] for i, l in enumerate(lines)]
    path.write_text("\n".join(numbered) + "\nEND\n")
    return str(path)


class TestResiduesAreNotPooledByNumberAlone:
    """100 and 100A are two residues; the atom graph used to merge them."""

    def test_insertion_codes_split_the_residue(self, tmp_path):
        from plmol import Protein

        PDBParser.clear_cache()
        path = _write(tmp_path / "icode.pdb",
                      _glycine_chain("A", [(3, " "), (3, "A"), (3, "B")]))
        graph = Protein.from_pdb(path, standardize=False).featurize(mode="atom_graph")["atom_graph"]

        assert len(np.unique(np.asarray(graph["atom_to_residue"]))) == 3
        assert len(graph["residue_atom_indices"]) == 3
        assert all(len(atoms) == 4 for atoms in graph["residue_atom_indices"])


class TestSequenceSeparationMeansSequence:
    """It is a distance along one chain, not a distance in the atom list."""

    def test_two_chains_are_not_a_few_residues_apart(self, tmp_path):
        from plmol import Protein

        PDBParser.clear_cache()
        lines = (_glycine_chain("A", [(1, " "), (2, " "), (3, " ")], x0=0.0)
                 + _glycine_chain("B", [(1, " "), (2, " "), (3, " ")], x0=5.0))
        path = _write(tmp_path / "twochain.pdb", lines)
        graph = Protein.from_pdb(path, standardize=False).featurize(mode="atom_graph")["atom_graph"]

        chains = np.array([a.chain_id for a in PDBParser(path).protein_atoms])
        src, dst = np.asarray(graph["edge_index"])
        separation = np.asarray(graph["sequence_separation"]).ravel()
        cross = chains[src] != chains[dst]

        assert cross.any(), "the two chains must be close enough to share edges"
        assert (separation[cross] == 32.0).all()

    def test_a_missing_loop_is_not_a_peptide_bond(self, tmp_path):
        from plmol import Protein

        PDBParser.clear_cache()
        # Residues 1-3 then 20-22: the numbering says ten residues are missing,
        # and the coordinates continue as if nothing were.
        path = _write(tmp_path / "gap.pdb",
                      _glycine_chain("A", [(1, " "), (2, " "), (3, " "),
                                           (20, " "), (21, " "), (22, " ")]))
        graph = Protein.from_pdb(path, standardize=False).featurize(mode="atom_graph")["atom_graph"]

        numbers = np.asarray(graph["residue_number"])
        src, dst = np.asarray(graph["edge_index"])
        separation = np.asarray(graph["sequence_separation"]).ravel()
        across = (numbers[src] == 3) & (numbers[dst] == 20)
        assert across.any()
        assert (separation[across] == 17.0).all()

    def test_secondary_structure_stops_at_the_gap(self, example_pdb, tmp_path):
        # Delete residues 101-110 of chain A, the shape a disordered loop leaves
        # in a deposited structure. 100 and 111 are then adjacent in the file
        # and 15 A apart; measuring psi and phi across that pair called 100 a
        # sheet and 111 a helix, from a peptide bond that is not there.
        from plmol import Protein

        PDBParser.clear_cache()
        kept = [
            line.rstrip("\n")
            for line in open(example_pdb)
            if line.startswith(("ATOM  ", "HETATM"))
            and not (line[21] == "A" and 101 <= int(line[22:26]) <= 110)
        ]
        path = _write(tmp_path / "loop_gap.pdb", kept)
        graph = Protein.from_pdb(path, standardize=False).featurize(mode="atom_graph")["atom_graph"]

        atoms = PDBParser(path).protein_atoms
        numbers = np.array([a.res_num for a in atoms])
        chains = np.array([a.chain_id for a in atoms])
        ss = np.asarray(graph["secondary_structure"])
        for number in (100, 111):
            rows = ss[(chains == "A") & (numbers == number)]
            assert rows.size
            assert np.allclose(rows, [0.0, 0.0, 1.0]), f"A{number} took an angle across the gap"
