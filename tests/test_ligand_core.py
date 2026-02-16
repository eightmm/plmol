"""Tests for plmol/ligand/core.py — Ligand class."""

import numpy as np
import pytest

from plmol.ligand.core import Ligand


class TestLigandFromSmiles:
    def test_basic(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        assert lig.smiles is not None
        assert "O" in lig.smiles or "CCO" in lig.smiles

    def test_sequence_alias(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        assert lig.sequence == lig.smiles

    def test_invalid_smiles(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            Ligand.from_smiles("not_a_smiles_XXXXX")

    def test_add_hs(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles, add_hs=True)
        # With Hs, should have more atoms reflected in graph
        assert lig._rdmol.GetNumAtoms() > 3


class TestLigandFromSDF:
    def test_from_sdf(self, example_sdf):
        lig = Ligand.from_sdf(example_sdf)
        assert lig.smiles is not None
        assert lig._rdmol is not None

    def test_has_3d(self, example_sdf):
        lig = Ligand.from_sdf(example_sdf)
        assert lig.has_3d is True
        assert lig.coords is not None
        assert lig.coords.shape[1] == 3


class TestLigandProperties:
    def test_smiles_setter(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        lig.smiles = "CCC"
        assert lig.smiles == "CCC"
        assert lig.sequence == "CCC"

    def test_sequence_setter(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        lig.sequence = "CCCC"
        assert lig.smiles == "CCCC"

    def test_no_mol_raises(self):
        lig = Ligand()
        with pytest.raises(ValueError, match="no RDKit"):
            lig.featurize()


class TestLigandFeaturize:
    def test_graph_mode(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        result = lig.featurize(mode="graph")
        assert "graph" in result
        graph = result["graph"]
        assert "node_features" in graph
        assert "adjacency" in graph
        assert isinstance(graph["node_features"], np.ndarray)

    def test_fingerprint_mode(self, aspirin_smiles):
        lig = Ligand.from_smiles(aspirin_smiles)
        result = lig.featurize(mode="fingerprint")
        assert "fingerprint" in result
        assert "descriptors" in result["fingerprint"]
        assert "ecfp4" in result["fingerprint"]

    def test_smiles_mode(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        result = lig.featurize(mode="smiles")
        assert "smiles" in result
        assert "sequence" in result

    def test_multiple_modes(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        result = lig.featurize(mode=["graph", "fingerprint", "smiles"])
        assert "graph" in result
        assert "fingerprint" in result
        assert "smiles" in result

    def test_lazy_graph(self, ethanol_smiles):
        lig = Ligand.from_smiles(ethanol_smiles)
        graph = lig.graph
        assert graph is not None
        assert "node_features" in graph

    def test_lazy_fingerprint(self, aspirin_smiles):
        lig = Ligand.from_smiles(aspirin_smiles)
        fp = lig.fingerprint
        assert fp is not None
        assert "ecfp4" in fp

    def test_graph_generates_coords(self, aspirin_smiles):
        """Graph featurization generates 3D coords even without explicit conformer."""
        lig = Ligand.from_smiles(aspirin_smiles)
        assert lig.has_3d is False
        result = lig.featurize(mode="graph")
        # Graph should have coordinates from conformer generation
        assert result["graph"]["coords"].shape[1] == 3

    def test_numpy_conversion(self, ethanol_smiles):
        """Output should be numpy arrays, not torch tensors."""
        lig = Ligand.from_smiles(ethanol_smiles)
        result = lig.featurize(mode="graph")
        graph = result["graph"]
        assert isinstance(graph["node_features"], np.ndarray)
        assert isinstance(graph["adjacency"], np.ndarray)
