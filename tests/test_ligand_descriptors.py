"""Tests for plmol/ligand/descriptors.py — MoleculeFeaturizer."""

import torch
import numpy as np

from plmol.ligand.descriptors import MoleculeFeaturizer


class TestMoleculeFeaturizerInit:
    def test_from_smiles(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        assert mf.get_rdkit_mol() is not None

    def test_from_aspirin(self, aspirin_smiles):
        mf = MoleculeFeaturizer(aspirin_smiles)
        assert mf.get_rdkit_mol() is not None


class TestGetDescriptors:
    def test_shape(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        desc = mf.get_descriptors()
        assert isinstance(desc, torch.Tensor)
        assert desc.ndim == 1
        assert desc.shape[0] == 62

    def test_finite(self, aspirin_smiles):
        mf = MoleculeFeaturizer(aspirin_smiles)
        desc = mf.get_descriptors()
        assert torch.isfinite(desc).all()


class TestGetFingerprints:
    def test_default_fps(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        fps = mf.get_fingerprints(mf.get_rdkit_mol())
        assert isinstance(fps, dict)
        expected = ["ecfp4", "ecfp6", "maccs", "rdkit"]
        for key in expected:
            assert key in fps, f"Missing fingerprint: {key}"
            assert isinstance(fps[key], torch.Tensor)

    def test_ecfp4_shape(self, aspirin_smiles):
        mf = MoleculeFeaturizer(aspirin_smiles)
        fps = mf.get_fingerprints(mf.get_rdkit_mol())
        assert fps["ecfp4"].shape[0] == 2048

    def test_maccs_shape(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        fps = mf.get_fingerprints(mf.get_rdkit_mol())
        assert fps["maccs"].shape[0] == 167


class TestGetFeatures:
    def test_has_descriptors_and_fps(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        features = mf.get_features()
        assert isinstance(features, dict)
        assert "descriptors" in features
        assert "ecfp4" in features

    def test_include_fps_filter(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        features = mf.get_features(include_fps=("ecfp4", "maccs"))
        assert "ecfp4" in features
        assert "maccs" in features


class TestGetMorganFingerprint:
    def test_default(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        fp = mf.get_morgan_fingerprint()
        assert isinstance(fp, torch.Tensor)
        assert fp.shape[0] == 2048

    def test_custom_params(self, aspirin_smiles):
        mf = MoleculeFeaturizer(aspirin_smiles)
        fp = mf.get_morgan_fingerprint(radius=3, n_bits=1024)
        assert fp.shape[0] == 1024


class TestFeaturize:
    def test_graph_output(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        node, edge, adj = mf.featurize()
        assert isinstance(node, dict)
        assert isinstance(edge, dict)
        assert "node_feats" in node
        # Ethanol: C-C-O = 3 heavy atoms
        assert node["node_feats"].shape[0] == 3

    def test_adjacency_shape(self, aspirin_smiles):
        mf = MoleculeFeaturizer(aspirin_smiles)
        node, edge, adj = mf.featurize()
        n = node["node_feats"].shape[0]
        assert adj.shape[0] == n
        assert adj.shape[1] == n


class TestGet3dCoordinates:
    def test_returns_tensor(self, ethanol_smiles):
        mf = MoleculeFeaturizer(ethanol_smiles)
        coords = mf.get_3d_coordinates()
        if coords is not None:
            assert isinstance(coords, torch.Tensor)
            assert coords.shape[1] == 3


class TestGetAllFeatures:
    def test_comprehensive(self, aspirin_smiles):
        mf = MoleculeFeaturizer(aspirin_smiles)
        all_feats = mf.get_all_features()
        assert isinstance(all_feats, dict)
        assert "graph" in all_feats
        assert "fingerprints" in all_feats
        assert "descriptors" in all_feats
