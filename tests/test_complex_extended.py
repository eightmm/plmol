"""Extended tests for plmol/complex.py — Complex class."""

import pytest
import numpy as np

from plmol import Complex, Protein
from plmol.errors import InputError
from plmol.complex import _freeze


class TestFreeze:
    def test_dict(self):
        result = _freeze({"b": 2, "a": 1})
        assert isinstance(result, tuple)
        # Should be sorted by key
        assert result[0][0] == "a"

    def test_list(self):
        result = _freeze([1, 2, 3])
        assert result == (1, 2, 3)

    def test_set(self):
        result = _freeze({3, 1, 2})
        assert result == (1, 2, 3)

    def test_nested(self):
        result = _freeze({"key": [1, {"inner": 2}]})
        assert isinstance(result, tuple)

    def test_scalar(self):
        assert _freeze(42) == 42
        assert _freeze("hello") == "hello"
        assert _freeze(None) is None


class TestComplexFromInputs:
    def test_sequence_and_smiles(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        assert c.ligand_obj is not None
        assert c.protein_obj is not None

    def test_from_files(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        assert c.ligand_obj is not None
        assert c.protein_obj is not None


class TestComplexSetters:
    def test_set_ligand(self, ethanol_smiles, aspirin_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        # Cache something
        c.ligand(mode=["smiles"])
        c.set_ligand(aspirin_smiles)
        # Cache should be cleared
        result = c.ligand(mode=["smiles"])
        assert result["smiles"] != ethanol_smiles  # Should be aspirin now

    def test_set_protein(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        c.protein(mode=["sequence"])
        new_protein = Protein.from_sequence("WWWWWW")
        c.set_protein(new_protein)
        result = c.protein(mode=["sequence"])
        assert result["sequence"] == "WWWWWW"


class TestComplexLigand:
    def test_no_ligand_raises(self):
        c = Complex(protein_obj=Protein.from_sequence("AAA"))
        with pytest.raises(InputError, match="Ligand"):
            c.ligand()

    def test_graph_and_fingerprint(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        result = c.ligand(mode=["graph", "fingerprint"])
        assert "graph" in result
        assert "fingerprint" in result

    def test_cache_hit(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        a = c.ligand(mode=["graph"])
        b = c.ligand(mode=["graph"])
        assert a is b


class TestComplexProtein:
    def test_no_protein_raises(self):
        from plmol.ligand.core import Ligand
        c = Complex(ligand_obj=Ligand.from_smiles("CCO"))
        with pytest.raises(InputError, match="Protein"):
            c.protein()

    def test_sequence_mode(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        result = c.protein(mode=["sequence"])
        assert result["sequence"] == "ACDEFG"

    def test_graph_mode_real(self, example_pdb, ethanol_smiles):
        c = Complex.from_inputs(protein=example_pdb, ligand=ethanol_smiles)
        result = c.protein(mode=["graph"])
        assert "graph" in result


class TestComplexInteraction:
    def test_missing_components(self, ethanol_smiles):
        c = Complex(protein_obj=Protein.from_sequence("A"))
        with pytest.raises(InputError, match="both"):
            c.interaction()

    def test_real_interaction(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        result = c.interaction()
        assert isinstance(result, dict)
        assert "edges" in result

    def test_with_pocket_cutoff(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        result = c.interaction(pocket_cutoff=6.0)
        assert isinstance(result, dict)


class TestComplexFeaturize:
    def test_all_requests(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        result = c.featurize(requests="all")
        assert "ligand" in result
        assert "protein" in result
        assert "interaction" in result

    def test_single_request(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        result = c.featurize(requests=["ligand"])
        assert "ligand" in result
        assert "protein" not in result

    def test_invalid_request(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        with pytest.raises(InputError):
            c.featurize(requests=["bad_request"])

    def test_with_kwargs(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        result = c.featurize(
            requests=["ligand"],
            ligand_kwargs={"mode": ["smiles"]},
        )
        assert "smiles" in result["ligand"]
