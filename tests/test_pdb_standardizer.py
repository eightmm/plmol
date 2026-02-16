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
