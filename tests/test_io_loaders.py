"""Tests for plmol/io/loaders.py."""

import pytest

from plmol.io.loaders import load_protein_input, load_ligand_input
from plmol.protein.core import Protein
from plmol.ligand.core import Ligand
from plmol.errors import InputError, DependencyError


class TestLoadProteinInput:
    def test_protein_object(self):
        p = Protein.from_sequence("ACDEFG")
        result = load_protein_input(p)
        assert result is p

    def test_pdb_path(self, example_pdb):
        result = load_protein_input(example_pdb)
        assert isinstance(result, Protein)

    def test_invalid_type(self):
        with pytest.raises(InputError, match="Unsupported"):
            load_protein_input(12345)


class TestLoadLigandInput:
    def test_ligand_object(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        result = load_ligand_input(lig)
        assert result is lig

    def test_smiles_string(self, ethanol_smiles):
        result = load_ligand_input(ethanol_smiles)
        assert isinstance(result, Ligand)
        assert result.smiles is not None

    def test_rdkit_mol(self, ethanol_smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(ethanol_smiles)
        result = load_ligand_input(mol)
        assert isinstance(result, Ligand)

    def test_sdf_path(self, example_sdf):
        result = load_ligand_input(example_sdf)
        assert isinstance(result, Ligand)
        assert result.has_3d is True

    def test_invalid_type(self):
        with pytest.raises(InputError, match="Unsupported"):
            load_ligand_input(12345)

    def test_invalid_extension(self, tmp_path):
        bad_file = str(tmp_path / "file.xyz")
        with open(bad_file, "w") as f:
            f.write("dummy")
        with pytest.raises(InputError, match="Unsupported ligand file"):
            load_ligand_input(bad_file)
