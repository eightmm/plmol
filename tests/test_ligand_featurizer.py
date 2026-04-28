"""Tests for plmol/ligand/featurizer.py — LigandFeaturizer."""

import torch
import numpy as np
import pytest

from plmol.ligand.featurizer import LigandFeaturizer
from plmol.errors import InputError


class TestLigandFeaturizerInit:
    def test_from_smiles(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        assert lf.num_atoms == 3
        assert lf.input_smiles is not None

    def test_from_mol(self, ethanol_smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(ethanol_smiles)
        lf = LigandFeaturizer(mol)
        assert lf.num_atoms == 3

    def test_properties(self, aspirin_smiles):
        lf = LigandFeaturizer(aspirin_smiles)
        assert lf.num_atoms > 0
        assert lf.num_bonds > 0
        assert lf.num_rings > 0
        assert lf.has_3d is False


class TestGetGraph:
    def test_standardized(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        graph = lf.get_graph(standardized=True)
        assert isinstance(graph, dict)
        assert "node_features" in graph
        assert "adjacency" in graph
        assert "bond_mask" in graph
        assert "distance_matrix" in graph

    def test_raw_tuple(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        result = lf.get_graph(standardized=False)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_adjacency_dim(self, aspirin_smiles):
        lf = LigandFeaturizer(aspirin_smiles)
        graph = lf.get_graph(standardized=True)
        assert graph["adjacency"].shape[-1] == 37  # 27 bond + 10 pair


class TestGetFeatures:
    def test_basic(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        features = lf.get_features()
        assert isinstance(features, dict)
        assert "descriptors" in features

    def test_include_fps(self, aspirin_smiles):
        lf = LigandFeaturizer(aspirin_smiles)
        features = lf.get_features(include_fps=("ecfp4",))
        assert "ecfp4" in features


class TestGetMorganFingerprint:
    def test_default(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        result = lf.get_morgan_fingerprint()
        assert isinstance(result, dict)
        assert "fingerprint" in result
        assert result["type"] == "morgan"
        assert result["radius"] == 2
        assert result["n_bits"] == 2048


class TestFeaturize:
    def test_all_modes(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        result = lf.featurize(mode=["graph", "fingerprint", "smiles"])
        assert "graph" in result
        assert "fingerprint" in result
        assert "smiles" in result

    def test_all_uses_default_modes(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        result = lf.featurize(mode="all")
        assert set(result) == {"graph", "fingerprint", "smiles", "sequence"}

    def test_descriptor_mode(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        result = lf.featurize(mode="descriptor")
        assert "descriptor" in result
        assert result["descriptor"]["descriptors"].shape[0] == 62
        assert len(result["descriptor"]["descriptor_names"]) == 62

    def test_morgan_mode(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        result = lf.featurize(mode="morgan")
        assert "fingerprint" in result
        assert "ecfp4" in result["fingerprint"]
        assert "morgan" in result
        assert result["morgan"]["fingerprint"].shape[0] == 2048

    def test_morgan_include_fps_alias(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        result = lf.featurize(
            mode="fingerprint",
            fingerprint_kwargs={"include_fps": ("morgan",)},
        )
        assert set(result["fingerprint"]) == {"descriptors", "ecfp4"}

    def test_brics_fragment_kwargs(self, aspirin_smiles):
        lf = LigandFeaturizer(aspirin_smiles)
        result = lf.featurize(
            mode="fragment",
            fragment_kwargs={"method": "brics"},
        )
        assert result["fragment"]["fragment_method"] == "brics"

    def test_invalid_mode_raises(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        with pytest.raises(InputError):
            lf.featurize(mode="bad_mode")

    def test_fingerprint_kwargs_not_mutated(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        kwargs = {"include_fps": ("ecfp4",)}
        result = lf.featurize(mode="fingerprint", fingerprint_kwargs=kwargs)
        assert kwargs == {"include_fps": ("ecfp4",)}
        assert set(result["fingerprint"]) == {"descriptors", "ecfp4"}

    def test_smiles_mode(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        result = lf.featurize(mode="smiles")
        assert "smiles" in result
        assert "sequence" in result


class TestSetMolecule:
    def test_reset(self, ethanol_smiles, aspirin_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        assert lf.num_atoms == 3
        lf.set_molecule(aspirin_smiles)
        assert lf.num_atoms > 3


class TestAdjacencyToBondEdges:
    def test_basic(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        graph = lf.get_graph(standardized=True)
        edge_index, edge_features = LigandFeaturizer.adjacency_to_bond_edges(graph["adjacency"])
        assert edge_index.shape[0] == 2
        assert edge_features.shape[0] == edge_index.shape[1]

    def test_numpy_input(self, ethanol_smiles):
        lf = LigandFeaturizer(ethanol_smiles)
        graph = lf.get_graph(standardized=True)
        adj_np = graph["adjacency"].numpy() if isinstance(graph["adjacency"], torch.Tensor) else graph["adjacency"]
        edge_index, edge_features = LigandFeaturizer.adjacency_to_bond_edges(adj_np)
        assert edge_index.shape[0] == 2

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="adjacency"):
            LigandFeaturizer.adjacency_to_bond_edges(torch.zeros(3, 3))
