"""Tests for plmol/protein/pdb_standardizer.py."""

import os

import pytest

from plmol.protein.pdb_standardizer import PDBStandardizer, standardize_pdb


@pytest.fixture
def simple_pdb(tmp_path):
    """PDB with standard ALA + water + hydrogen."""
    content = (
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C  \n"
        "ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C  \n"
        "ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00  0.00           O  \n"
        "ATOM      5  CB  ALA A   1       2.000   1.000   3.000  1.00  0.00           C  \n"
        "ATOM      6  H   ALA A   1       0.500   2.000   3.000  1.00  0.00           H  \n"
        "HETATM    7  O   HOH A   2       5.000   5.000   5.000  1.00  0.00           O  \n"
        "END\n"
    )
    path = str(tmp_path / "simple.pdb")
    with open(path, "w") as f:
        f.write(content)
    return path


@pytest.fixture
def ptm_pdb(tmp_path):
    """PDB with a PTM residue (MSE = selenomethionine)."""
    content = (
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C  \n"
        "ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C  \n"
        "ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00  0.00           O  \n"
        "ATOM      5  CB  ALA A   1       2.000   1.000   3.000  1.00  0.00           C  \n"
        "HETATM    6  N   MSE A   2       5.000   2.000   3.000  1.00  0.00           N  \n"
        "HETATM    7  CA  MSE A   2       6.000   2.000   3.000  1.00  0.00           C  \n"
        "HETATM    8  C   MSE A   2       7.000   2.000   3.000  1.00  0.00           C  \n"
        "HETATM    9  O   MSE A   2       7.000   3.000   3.000  1.00  0.00           O  \n"
        "HETATM   10  CB  MSE A   2       6.000   1.000   3.000  1.00  0.00           C  \n"
        "HETATM   11  CG  MSE A   2       6.000   0.000   3.000  1.00  0.00           C  \n"
        "HETATM   12  SE  MSE A   2       6.000  -1.000   3.000  1.00  0.00          SE  \n"
        "END\n"
    )
    path = str(tmp_path / "ptm.pdb")
    with open(path, "w") as f:
        f.write(content)
    return path


class TestPDBStandardizer:
    def test_default_init(self):
        s = PDBStandardizer()
        assert s.remove_hydrogens is True
        assert s.ptm_handling == "base_aa"

    def test_invalid_ptm_mode(self):
        with pytest.raises(ValueError, match="Invalid ptm_handling"):
            PDBStandardizer(ptm_handling="invalid_mode")

    def test_standardize_removes_water_and_hydrogen(self, simple_pdb, tmp_path):
        output = str(tmp_path / "output.pdb")
        s = PDBStandardizer()
        result = s.standardize(simple_pdb, output)
        assert os.path.exists(result)
        with open(result) as f:
            lines = f.readlines()
        atom_lines = [l for l in lines if l.startswith("ATOM") or l.startswith("HETATM")]
        # H atom and HOH should be removed
        for line in atom_lines:
            assert "HOH" not in line
            atom_name = line[12:16].strip()
            assert atom_name != "H"

    def test_remove_hydrogens_false_parses_h(self, simple_pdb, tmp_path):
        """remove_hydrogens=False should not filter H in _process_atom_line.
        However, the standard atom order for ALA may not include H in output.
        Verify the internal parse step retains H atoms."""
        s = PDBStandardizer(remove_hydrogens=False)
        protein_residues, hetatm_residues = s._parse_pdb(simple_pdb)
        # Flatten all atom names from parsed residues
        all_atoms = set()
        for residue_key, atoms in protein_residues.items():
            all_atoms.update(atoms.keys())
        assert "H" in all_atoms, "H should be parsed when remove_hydrogens=False"

    def test_remove_hydrogens_true_filters_h(self, simple_pdb, tmp_path):
        """remove_hydrogens=True should filter H during parsing."""
        s = PDBStandardizer(remove_hydrogens=True)
        protein_residues, hetatm_residues = s._parse_pdb(simple_pdb)
        all_atoms = set()
        for residue_key, atoms in protein_residues.items():
            all_atoms.update(atoms.keys())
        assert "H" not in all_atoms

    def test_ptm_base_aa(self, ptm_pdb, tmp_path):
        """base_aa mode converts MSE -> MET."""
        output = str(tmp_path / "output.pdb")
        s = PDBStandardizer(ptm_handling="base_aa")
        s.standardize(ptm_pdb, output)
        with open(output) as f:
            content = f.read()
        assert "MSE" not in content
        assert "MET" in content

    def test_ptm_unk(self, ptm_pdb, tmp_path):
        """unk mode converts MSE -> UNK with backbone only."""
        output = str(tmp_path / "output.pdb")
        s = PDBStandardizer(ptm_handling="unk")
        s.standardize(ptm_pdb, output)
        with open(output) as f:
            lines = f.readlines()
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        # Find UNK residue atoms
        unk_atoms = [l[12:16].strip() for l in atom_lines if "UNK" in l[17:20]]
        # Should only have backbone atoms
        backbone = {"N", "CA", "C", "O", "CB"}
        for a in unk_atoms:
            assert a in backbone, f"Non-backbone atom {a} found in UNK residue"

    def test_ptm_remove(self, ptm_pdb, tmp_path):
        """remove mode removes the PTM residue entirely."""
        output = str(tmp_path / "output.pdb")
        s = PDBStandardizer(ptm_handling="remove")
        s.standardize(ptm_pdb, output)
        with open(output) as f:
            content = f.read()
        assert "MSE" not in content

    def test_ptm_preserve(self, ptm_pdb, tmp_path):
        """preserve mode keeps PTM residue name intact."""
        output = str(tmp_path / "output.pdb")
        s = PDBStandardizer(ptm_handling="preserve")
        s.standardize(ptm_pdb, output)
        with open(output) as f:
            content = f.read()
        assert "MSE" in content

    def test_output_dir_creation(self, simple_pdb, tmp_path):
        output = str(tmp_path / "subdir" / "output.pdb")
        s = PDBStandardizer()
        s.standardize(simple_pdb, output)
        assert os.path.exists(output)


class TestStandardizePdbFunction:
    def test_convenience_function(self, simple_pdb, tmp_path):
        output = str(tmp_path / "output.pdb")
        result = standardize_pdb(simple_pdb, output)
        assert os.path.exists(result)

    def test_example_pdb(self, example_pdb, tmp_path):
        """Standardize real PDB file."""
        output = str(tmp_path / "standardized.pdb")
        standardize_pdb(example_pdb, output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0


class TestResidueOrdering:
    """Standardising must not reorder the chain.

    _sort_residue_key built the number with ``filter(str.isdigit, ...)``, which
    drops the minus sign. A construct numbered -2, -1, 0, 1, 2 -- an expression
    tag, a propeptide, any mature protein numbered from its own first residue --
    sorted as 2, 1, 0, 1, 2, so the sequence came out scrambled and every
    backbone neighbour with it.
    """

    @staticmethod
    def _build(tmp_path, spec, name):
        def line(serial, atom_name, res, resnum, icode, xyz, element):
            row = list(" " * 80)
            row[0:6] = "ATOM  "
            row[6:11] = f"{serial:5d}"
            row[12:16] = (" " + atom_name).ljust(4)[:4]
            row[17:20] = res.rjust(3)[:3]
            row[21] = "A"
            row[22:26] = f"{resnum:4d}"
            row[26] = icode
            row[30:38] = f"{xyz[0]:8.3f}"
            row[38:46] = f"{xyz[1]:8.3f}"
            row[46:54] = f"{xyz[2]:8.3f}"
            row[54:60] = "  1.00"
            row[60:66] = "  0.00"
            row[76:78] = element.rjust(2)
            return "".join(row).rstrip() + "\n"

        backbone = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
        text = "".join(
            line(i * 10 + j, atom_name, res, num, icode,
                 ((i + 1) * 3.8 + j * 0.4, 0.0, 0.0), element)
            for i, (num, icode, res) in enumerate(spec)
            for j, (atom_name, element) in enumerate(backbone, 1)
        )
        path = tmp_path / name
        path.write_text(text + "END\n")
        return str(path)

    CASES = [
        ([(-2, " ", "CYS"), (-1, " ", "ASP"), (0, " ", "GLU"),
          (1, " ", "PHE"), (2, " ", "GLY")], "CDEFG", "negative numbers"),
        ([(-1, " ", "CYS"), (-1, "A", "ASP"), (0, " ", "GLU"),
          (1, " ", "PHE")], "CDEF", "negative with an insertion code"),
        ([(1, " ", "CYS"), (2, " ", "ASP"), (3, " ", "GLU")], "CDE", "plain"),
        ([(998, " ", "CYS"), (999, " ", "ASP"), (1000, " ", "GLU")], "CDE", "four digits"),
    ]

    @pytest.mark.parametrize("spec,sequence,label", CASES, ids=[c[2] for c in CASES])
    @pytest.mark.parametrize("standardize", [True, False])
    def test_the_chain_keeps_its_order(self, tmp_path, spec, sequence, label, standardize):
        from plmol import Protein

        path = self._build(tmp_path, spec, f"{label.replace(' ', '_')}.pdb")
        protein = Protein.from_pdb(path, standardize=standardize)
        assert protein.featurize(mode="sequence")["sequence"] == sequence

    def test_the_sort_key_keeps_the_sign(self):
        from plmol.protein.pdb_standardizer import PDBStandardizer

        key = PDBStandardizer()._sort_residue_key
        assert key(("A", "-2", "ALA")) == ("A", -2, "")
        assert key(("A", "-1", "ALA")) < key(("A", "0", "ALA")) < key(("A", "1", "ALA"))
        assert key(("A", "100A", "ALA")) == ("A", 100, "A")
        assert key(("A", "100", "ALA")) < key(("A", "100A", "ALA"))


class TestFourCharacterAtomNames:
    """The atom name field is columns 13-16, and a four-character name uses all of it."""

    STRUCTURE = (
        "ATOM      1  N   LEU A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  LEU A   1       1.450   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  C   LEU A   1       2.400   1.000   0.000  1.00 20.00           C\n"
        "ATOM      4  O   LEU A   1       2.400   2.200   0.000  1.00 20.00           O\n"
        "HETATM    5 CL21 LIG A 401       9.000   9.000   9.000  1.00 20.00          CL\n"
        "HETATM    6 FE   HEM A 402       5.000   5.000   5.000  1.00 20.00          FE\n"
        "END\n"
    )

    def test_the_name_survives_standardisation(self, tmp_path):
        from plmol.parsers.pdb_parser import parse_pdb_line
        from plmol.protein.pdb_standardizer import PDBStandardizer

        source = tmp_path / "fourchar.pdb"
        source.write_text(self.STRUCTURE)
        target = tmp_path / "fourchar_std.pdb"
        PDBStandardizer().standardize(str(source), str(target))

        written = {}
        for line in open(target):
            if line.startswith(("ATOM  ", "HETATM")):
                atom = parse_pdb_line(line)
                written[atom.atom_name] = atom

        assert "CL21" in written, "a four-character name used to come back as CL2"
        assert written["CL21"].alt_loc == "", "and its fourth character as an alternate location"
        assert written["CL21"].element == "CL"
        assert written["FE"].element == "FE"


class TestPdbLineRoundTrip:
    """format_pdb_line is the inverse of parse_pdb_line, column for column."""

    LINES = [
        "ATOM      1  N   LEU A   1       0.000   0.000   0.000  1.00 20.00           N",
        "ATOM      2  CA BLEU A 100A      1.450  -2.000   3.000  0.60 20.00           C",
        "HETATM    5 CL21 LIG A 401       9.000   9.000   9.000  1.00 20.00          CL",
        "ATOM      9 HD11 LEU A   1       4.900  -2.900   0.100  1.00 20.00           H",
        "HETATM    7 FE   HEM A 402       5.000   5.000   5.000  1.00 20.00          FE",
        "ATOM     11  O5' DA  B  -2      -1.000   2.000   3.000  1.00 20.00           O",
    ]

    @pytest.mark.parametrize("line", LINES)
    def test_every_field_comes_back(self, line):
        from plmol.parsers.pdb_parser import format_pdb_line, parse_pdb_line

        before = parse_pdb_line(line)
        after = parse_pdb_line(format_pdb_line(before, serial=1).rstrip("\n"))
        for field in ("record_type", "atom_name", "res_name", "res_num", "chain_id",
                      "element", "insertion_code", "alt_loc", "coords"):
            assert getattr(before, field) == getattr(after, field), field
        assert before.occupancy == after.occupancy
        assert before.b_factor == after.b_factor
