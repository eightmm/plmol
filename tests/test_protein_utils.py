"""Tests for plmol/protein/utils.py — PDB parsing utilities."""

import numpy as np
import pytest

from plmol.protein.utils import (
    ParsedAtom,
    PDBParser,
    is_atom_record,
    is_hetatm_record,
    is_hydrogen,
    normalize_residue_name,
    parse_pdb_line,
    parse_pdb_atom_line,
    calculate_sidechain_centroid,
)


# -- Low-level functions --

class TestIsAtomRecord:
    def test_atom(self):
        assert is_atom_record("ATOM  12345  CA  ALA A   1") is True

    def test_hetatm(self):
        assert is_atom_record("HETATM12345  CA  ALA A   1") is False

    def test_short_line(self):
        assert is_atom_record("ATOM") is False

    def test_remark(self):
        assert is_atom_record("REMARK this is a remark") is False


class TestIsHetatmRecord:
    def test_hetatm(self):
        assert is_hetatm_record("HETATM12345  CA  ALA A   1") is True

    def test_atom(self):
        assert is_hetatm_record("ATOM  12345  CA  ALA A   1") is False

    def test_short(self):
        assert is_hetatm_record("HET") is False


class TestIsHydrogen:
    def test_element_column(self):
        #                    |   atom  |res|c|resn|      x       y       z       occ  bf        el
        line = "ATOM      1  H   ALA A   1       1.000   2.000   3.000  1.00  0.00           H  "
        assert is_hydrogen(line) is True

    def test_atom_name_h(self):
        line = "ATOM      1  HB2 ALA A   1       1.000   2.000   3.000"
        assert is_hydrogen(line) is True

    def test_not_hydrogen(self):
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  "
        assert is_hydrogen(line) is False

    def test_short_line(self):
        assert is_hydrogen("ATOM") is False


class TestParsePdbLine:
    def test_basic_parse(self):
        # Standard PDB ATOM line with proper column formatting
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  \n"
        atom = parse_pdb_line(line)
        assert isinstance(atom, ParsedAtom)
        assert atom.record_type == "ATOM"
        assert atom.atom_name == "CA"
        assert atom.res_name == "ALA"
        assert atom.chain_id == "A"
        assert atom.res_num == 1
        assert abs(atom.coords[0] - 1.0) < 1e-3
        assert abs(atom.coords[1] - 2.0) < 1e-3
        assert abs(atom.coords[2] - 3.0) < 1e-3
        assert atom.element == "C"
        assert abs(atom.b_factor - 10.0) < 1e-3

    def test_hetatm(self):
        line = "HETATM    1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  \n"
        atom = parse_pdb_line(line)
        assert atom.record_type == "HETATM"

    def test_element_inference(self):
        # Short line without element column
        line = "ATOM      1  CB  ALA A   1       1.000   2.000   3.000"
        atom = parse_pdb_line(line)
        assert atom.element == "C"


class TestNormalizeResidueName:
    def test_standard(self):
        assert normalize_residue_name("ALA") == "ALA"

    def test_histidine_variant(self):
        result = normalize_residue_name("HID")
        assert result == "HIS"

    def test_unknown(self):
        assert normalize_residue_name("XYZ") == "UNK"

    def test_metal_detection(self):
        assert normalize_residue_name("ZN", "ZN") == "METAL"


class TestParsePdbAtomLine:
    def test_returns_tuple(self):
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  \n"
        result = parse_pdb_atom_line(line)
        assert isinstance(result, tuple)
        assert len(result) == 7
        assert result[0] == "ATOM"
        assert result[1] == "CA"
        assert result[2] == "ALA"
        assert result[3] == 1
        assert result[4] == "A"


class TestCalculateSidechainCentroid:
    def test_with_sidechain(self):
        # N, CA, C, O, CB, CG (6 atoms)
        coords = np.array([
            [0, 0, 0],  # N
            [1, 0, 0],  # CA
            [2, 0, 0],  # C
            [2, 1, 0],  # O
            [1, 1, 0],  # CB
            [1, 2, 0],  # CG
        ], dtype=float)
        centroid = calculate_sidechain_centroid(coords)
        expected = np.array([1, 1.5, 0])  # mean of CB, CG
        np.testing.assert_allclose(centroid, expected)

    def test_gly_no_sidechain(self):
        # N, CA, C, O (4 atoms, index 1=CA used)
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
        ], dtype=float)
        centroid = calculate_sidechain_centroid(coords)
        np.testing.assert_allclose(centroid, [1, 0, 0])

    def test_single_atom(self):
        coords = np.array([[5.0, 6.0, 7.0]])
        centroid = calculate_sidechain_centroid(coords)
        np.testing.assert_allclose(centroid, [5, 6, 7])

    def test_1d_input(self):
        coords = np.array([5.0, 6.0, 7.0])
        centroid = calculate_sidechain_centroid(coords)
        np.testing.assert_allclose(centroid, [5, 6, 7])


# -- PDBParser class --

class TestPDBParser:
    def test_parse_mini(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        assert parser.get_num_residues() == 10
        assert parser.get_num_atoms() > 0

    def test_all_atoms(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        assert len(parser.all_atoms) > 0
        assert len(parser.protein_atoms) > 0
        assert len(parser.protein_atoms) <= len(parser.all_atoms)

    def test_residues(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        residues = parser.residues
        assert len(residues) == 10
        # Check chain A present
        chain_a_keys = [k for k in residues if k[0] == "A"]
        assert len(chain_a_keys) == 5

    def test_get_sequence(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        seq = parser.get_sequence()
        assert len(seq) == 10
        assert "A" in seq  # ALA -> A
        assert "G" in seq  # GLY -> G

    def test_get_sequence_by_chain(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        seq_dict = parser.get_sequence_by_chain()
        assert "A" in seq_dict
        assert "B" in seq_dict
        assert len(seq_dict["A"]) == 5
        assert len(seq_dict["B"]) == 5
        assert seq_dict["A"] == "AAAAA"
        assert seq_dict["B"] == "GGGGG"

    def test_get_sequence_single_chain(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        seq = parser.get_sequence(chain_id="A")
        assert seq == "AAAAA"

    def test_get_atom_coords(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        coords = parser.get_atom_coords()
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        assert coords.shape[0] == parser.get_num_atoms()

    def test_get_atom_data(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        data = parser.get_atom_data()
        keys = ["atom_names", "res_names", "res_nums", "chain_ids", "coords", "elements", "residue_keys"]
        for key in keys:
            assert key in data
            assert len(data[key]) == parser.get_num_atoms()

    def test_get_residue_list(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        res_list = parser.get_residue_list()
        assert len(res_list) == 10
        assert all(len(r) == 3 for r in res_list)

    def test_cache(self, mini_pdb):
        PDBParser.clear_cache()
        p1 = PDBParser(mini_pdb)
        p2 = PDBParser(mini_pdb)
        assert p2._all_atoms is p1._all_atoms  # Same cached data

    def test_clear_cache(self, mini_pdb):
        PDBParser.clear_cache()
        PDBParser(mini_pdb)
        assert PDBParser.get_cached(mini_pdb) is not None
        PDBParser.clear_cache()
        assert PDBParser.get_cached(mini_pdb) is None

    def test_skip_cache(self, mini_pdb):
        PDBParser.clear_cache()
        p1 = PDBParser(mini_pdb, skip_cache=True)
        assert PDBParser.get_cached(mini_pdb) is None

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDBParser(str(tmp_path / "nonexistent.pdb"))

    def test_empty_file(self, tmp_path):
        empty_pdb = str(tmp_path / "empty.pdb")
        with open(empty_pdb, "w") as f:
            pass
        PDBParser.clear_cache()
        with pytest.raises(ValueError, match="empty"):
            PDBParser(empty_pdb)

    def test_filters_water(self, tmp_path):
        """Water molecules should be excluded from protein_atoms."""
        pdb_content = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  \n"
            "HETATM    2  O   HOH A   2       5.000   5.000   5.000  1.00  0.00           O  \n"
            "END\n"
        )
        pdb_file = str(tmp_path / "water.pdb")
        with open(pdb_file, "w") as f:
            f.write(pdb_content)
        PDBParser.clear_cache()
        parser = PDBParser(pdb_file)
        assert parser.get_num_atoms() == 1
        assert parser.protein_atoms[0].atom_name == "CA"

    def test_example_pdb(self, example_pdb):
        """Test with real PDB file."""
        PDBParser.clear_cache()
        parser = PDBParser(example_pdb)
        assert parser.get_num_residues() > 0
        assert parser.get_num_atoms() > 0
        seq = parser.get_sequence()
        assert len(seq) > 0
