"""Tests for plmol/interaction/ — PLInteractionFeaturizer + PocketExtractor."""

import torch
import pytest

from rdkit import Chem

from plmol.interaction.pli_featurizer import PLInteractionFeaturizer
from plmol.interaction.pocket_extractor import PocketExtractor, extract_pocket


@pytest.fixture
def protein_ligand_mols(example_pdb, example_sdf):
    """Load protein and ligand as RDKit Mol objects."""
    protein_mol = Chem.MolFromPDBFile(example_pdb, removeHs=True)
    suppl = Chem.SDMolSupplier(example_sdf)
    ligand_mol = next(suppl)
    return protein_mol, ligand_mol


class TestPLInteractionFeaturizer:
    def test_init(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        assert plif.num_protein_atoms > 0
        assert plif.num_ligand_atoms > 0

    def test_detect_all_interactions(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        interactions = plif.detect_all_interactions()
        assert isinstance(interactions, list)
        # 10gs has known interactions
        assert len(interactions) > 0

    def test_get_interaction_edges(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        edge_index, edge_features = plif.get_interaction_edges()
        assert edge_index.shape[0] == 2
        assert edge_features.shape[0] == edge_index.shape[1]
        assert edge_features.shape[1] == 74

    def test_get_interaction_graph(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        graph = plif.get_interaction_graph()
        assert isinstance(graph, dict)
        assert "edges" in graph
        assert "edge_features" in graph

    def test_get_heavy_atom_coords(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        p_coords, l_coords = plif.get_heavy_atom_coords()
        assert isinstance(p_coords, torch.Tensor)
        assert isinstance(l_coords, torch.Tensor)
        assert p_coords.shape[1] == 3
        assert l_coords.shape[1] == 3

    def test_get_atom_pharmacophore_features(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        p_feats, l_feats = plif.get_atom_pharmacophore_features()
        assert p_feats.shape[0] == plif.num_protein_atoms
        assert l_feats.shape[0] == plif.num_ligand_atoms

    def test_get_atom_chemical_features(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        p_feats, l_feats = plif.get_atom_chemical_features()
        assert p_feats.shape[0] == plif.num_protein_atoms
        assert l_feats.shape[0] == plif.num_ligand_atoms

    def test_get_distance_based_edges(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        edge_index, edge_features = plif.get_distance_based_edges()
        assert edge_index.shape[0] == 2

    def test_get_interaction_summary(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        summary = plif.get_interaction_summary()
        assert isinstance(summary, str)

    def test_get_feature_description(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        desc = plif.get_feature_description()
        assert isinstance(desc, dict)


class TestPocketExtractor:
    def test_from_files(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        pocket = pe.extract()
        assert pocket is not None
        assert pocket.pocket_mol is not None

    def test_from_protein(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_protein(example_pdb)
        ligand_mol = next(Chem.SDMolSupplier(example_sdf))
        pocket = pe.extract_for_ligand(ligand_mol)
        assert pocket is not None

    def test_get_pocket_pdb_block(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        pdb_block = pe.get_pocket_pdb_block()
        assert isinstance(pdb_block, str)
        assert "ATOM" in pdb_block

    def test_get_pocket_residue_mask(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        mask = pe.get_pocket_residue_mask()
        assert mask.dtype == bool
        assert mask.any()

    def test_get_residue_distances(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        dists = pe.get_residue_distances()
        assert dists.ndim == 1
        assert (dists >= 0).all()

    def test_save_pocket_pdb(self, example_pdb, example_sdf, tmp_path):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        output = str(tmp_path / "pocket.pdb")
        pe.save_pocket_pdb(output)
        import os
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_properties(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        assert pe.num_residues > 0
        assert len(pe.residue_keys) > 0


class TestExtractPocket:
    def test_convenience_function(self, example_pdb, example_sdf):
        ligand_mol = next(Chem.SDMolSupplier(example_sdf))
        pockets = extract_pocket(example_pdb, ligand_mol)
        assert isinstance(pockets, list)
        assert len(pockets) > 0
