"""Tests for plmol/ligand/graph.py — MoleculeGraphFeaturizer."""

import torch
import pytest

from rdkit import Chem

from plmol.ligand.graph import MoleculeGraphFeaturizer


@pytest.fixture
def graph_featurizer():
    return MoleculeGraphFeaturizer()


@pytest.fixture
def ethanol_mol():
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def aspirin_mol():
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


class TestFeaturize:
    def test_ethanol(self, graph_featurizer, ethanol_mol):
        node, edge, adj = graph_featurizer.featurize(ethanol_mol)
        assert isinstance(node, dict)
        assert isinstance(edge, dict)
        assert "node_feats" in node
        assert node["node_feats"].shape[0] == 3  # C, C, O

    def test_adjacency_shape(self, graph_featurizer, aspirin_mol):
        node, edge, adj = graph_featurizer.featurize(aspirin_mol)
        n = node["node_feats"].shape[0]
        assert adj.shape[:2] == (n, n)
        assert adj.shape[2] == 27  # bond feature channels

    def test_edge_keys(self, graph_featurizer, ethanol_mol):
        node, edge, adj = graph_featurizer.featurize(ethanol_mol)
        expected_keys = ["pair_features", "distance_matrix"]
        for key in expected_keys:
            assert key in edge, f"Missing edge key: {key}"


class TestAtomFeatures:
    def test_shape(self, graph_featurizer, aspirin_mol):
        feats, coords = graph_featurizer.get_atom_features(aspirin_mol)
        n = aspirin_mol.GetNumAtoms()
        assert feats.shape[0] == n
        assert feats.shape[1] == 98  # 98-dim node features

    def test_coords_shape(self, graph_featurizer, aspirin_mol):
        feats, coords = graph_featurizer.get_atom_features(aspirin_mol)
        n = aspirin_mol.GetNumAtoms()
        assert coords.shape == (n, 3)


class TestBondFeatures:
    def test_shape(self, graph_featurizer, ethanol_mol):
        bond_feats = graph_featurizer.get_bond_features(ethanol_mol)
        n = ethanol_mol.GetNumAtoms()
        assert bond_feats.shape == (n, n, 27)

    def test_symmetric(self, graph_featurizer, ethanol_mol):
        bond_feats = graph_featurizer.get_bond_features(ethanol_mol)
        # Bond features should be symmetric for undirected bonds
        for ch in range(bond_feats.shape[2]):
            diff = (bond_feats[:, :, ch] - bond_feats[:, :, ch].T).abs().max()
            assert diff < 1e-5, f"Channel {ch} not symmetric"


class TestPartialCharges:
    def test_shape(self, graph_featurizer, aspirin_mol):
        charges = graph_featurizer.get_partial_charges(aspirin_mol)
        n = aspirin_mol.GetNumAtoms()
        assert charges.shape == (n, 2)

    def test_finite(self, graph_featurizer, ethanol_mol):
        charges = graph_featurizer.get_partial_charges(ethanol_mol)
        assert torch.isfinite(charges).all()


class TestPhysicalProperties:
    def test_shape(self, graph_featurizer, ethanol_mol):
        props = graph_featurizer.get_physical_properties(ethanol_mol)
        assert props.shape[0] == ethanol_mol.GetNumAtoms()
        assert props.shape[1] == 6


class TestTopologicalFeatures:
    def test_shape(self, graph_featurizer, aspirin_mol):
        topo = graph_featurizer.get_topological_features(aspirin_mol)
        assert topo.shape[0] == aspirin_mol.GetNumAtoms()
        assert topo.shape[1] == 5


class TestPairFeatures:
    def test_shape(self, graph_featurizer, ethanol_mol):
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(ethanol_mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)
        coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        pair = graph_featurizer.get_pair_features(mol, coords)
        n = mol.GetNumAtoms()
        assert pair.shape[:2] == (n, n)

    def test_distance_matrix(self, graph_featurizer, ethanol_mol):
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(ethanol_mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)
        coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        dm = graph_featurizer.get_distance_matrix(mol, coords)
        n = mol.GetNumAtoms()
        assert dm.shape == (n, n)
        # Diagonal should be zero
        assert (dm.diag() < 1e-5).all()


class TestOneHot:
    def test_basic(self, graph_featurizer):
        result = graph_featurizer.one_hot("C", ["C", "N", "O", "S"])
        assert result[0] is True or result[0] == 1
        assert len(result) == 4  # same length as allowable_set

    def test_unknown(self, graph_featurizer):
        result = graph_featurizer.one_hot("X", ["C", "N", "O"])
        assert len(result) == 3


class TestNormalize:
    def test_basic(self, graph_featurizer):
        assert graph_featurizer.normalize(0.5, 0.0, 1.0) == 0.5

    def test_clip(self, graph_featurizer):
        assert graph_featurizer.normalize(2.0, 0.0, 1.0, clip=True) == 1.0
        assert graph_featurizer.normalize(-1.0, 0.0, 1.0, clip=True) == 0.0
