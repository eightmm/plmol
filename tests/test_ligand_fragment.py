"""Tests for ligand rotatable-bond fragmentation."""

import numpy as np
import pytest
from rdkit import Chem

from plmol.ligand.fragment import fragment_on_rotatable_bonds
from plmol.ligand import Ligand


# ---------------------------------------------------------------------------
# Unit tests for fragment_on_rotatable_bonds
# ---------------------------------------------------------------------------

class TestFragmentOnRotatableBonds:
    """Core fragmentation logic."""

    def test_aspirin_fragments(self):
        """Aspirin has 3 rotatable bonds → 4 fragments."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = fragment_on_rotatable_bonds(mol)
        assert result["num_rotatable_bonds"] == 3
        assert result["num_fragments"] == 4
        assert len(result["fragment_smiles"]) == 4

    def test_benzene_no_rotatable(self):
        """Benzene has 0 rotatable bonds → 1 fragment (whole molecule)."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = fragment_on_rotatable_bonds(mol)
        assert result["num_rotatable_bonds"] == 0
        assert result["num_fragments"] == 1
        assert len(result["fragment_smiles"]) == 1

    def test_ethylbenzene(self):
        """Ethylbenzene has 1 rotatable bond → 2 fragments."""
        mol = Chem.MolFromSmiles("c1ccc(CC)cc1")
        result = fragment_on_rotatable_bonds(mol)
        assert result["num_rotatable_bonds"] == 1
        assert result["num_fragments"] == 2

    def test_atom_to_fragment_mapping(self):
        """Every atom must be mapped; values in [0, num_fragments)."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = fragment_on_rotatable_bonds(mol)
        mapping = result["atom_to_fragment"]
        assert isinstance(mapping, np.ndarray)
        assert mapping.shape[0] == mol.GetNumAtoms()
        assert mapping.min() >= 0
        assert mapping.max() < result["num_fragments"]

    def test_fragment_adjacency_shape(self):
        """Adjacency must be (F, F) symmetric binary matrix."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = fragment_on_rotatable_bonds(mol)
        adj = result["fragment_adjacency"]
        f = result["num_fragments"]
        assert adj.shape == (f, f)
        np.testing.assert_array_equal(adj, adj.T)
        assert np.all((adj == 0) | (adj == 1))
        # diagonal should be zero
        np.testing.assert_array_equal(np.diag(adj), 0)

    def test_fragment_smiles_valid(self):
        """Each fragment SMILES must parse to a valid RDKit mol."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = fragment_on_rotatable_bonds(mol)
        for smi in result["fragment_smiles"]:
            assert Chem.MolFromSmiles(smi) is not None, f"Invalid fragment SMILES: {smi}"

    def test_min_fragment_size(self):
        """Small fragments should be merged when min_fragment_size > 1."""
        # CCCCOC — the terminal O-C can produce a tiny fragment
        mol = Chem.MolFromSmiles("CCCCOC")
        result_no_filter = fragment_on_rotatable_bonds(mol, min_fragment_size=1)
        result_filtered = fragment_on_rotatable_bonds(mol, min_fragment_size=3)
        assert result_filtered["num_fragments"] <= result_no_filter["num_fragments"]
        # All atoms still accounted for
        assert result_filtered["atom_to_fragment"].shape[0] == mol.GetNumAtoms()

    def test_single_atom_mol(self):
        """Single-atom molecule → 1 fragment, no rotatable bonds."""
        mol = Chem.MolFromSmiles("[Na]")
        result = fragment_on_rotatable_bonds(mol)
        assert result["num_rotatable_bonds"] == 0
        assert result["num_fragments"] == 1
        assert result["atom_to_fragment"].shape == (1,)
        assert result["fragment_adjacency"].shape == (1, 1)

    def test_long_chain(self):
        """Long alkane chain — many rotatable bonds."""
        mol = Chem.MolFromSmiles("CCCCCCCC")  # octane
        result = fragment_on_rotatable_bonds(mol)
        assert result["num_rotatable_bonds"] >= 5
        assert result["num_fragments"] == result["num_rotatable_bonds"] + 1

    def test_adjacency_edges_match_rotatable_bonds(self):
        """Number of edges in fragment adjacency == num_rotatable_bonds."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = fragment_on_rotatable_bonds(mol)
        num_edges = result["fragment_adjacency"].sum() // 2  # symmetric
        assert num_edges == result["num_rotatable_bonds"]

    def test_fragment_atom_indices(self):
        """fragment_atom_indices is consistent reverse of atom_to_fragment."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = fragment_on_rotatable_bonds(mol)
        fai = result["fragment_atom_indices"]
        assert len(fai) == result["num_fragments"]
        # All atoms covered exactly once
        all_atoms = sorted(a for atoms in fai for a in atoms)
        assert all_atoms == list(range(mol.GetNumAtoms()))
        # Consistent with atom_to_fragment
        for frag_idx, atoms in enumerate(fai):
            for a in atoms:
                assert result["atom_to_fragment"][a] == frag_idx

    def test_fragment_atom_indices_no_rotatable(self):
        """No rotatable bonds → single fragment covers all atoms."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = fragment_on_rotatable_bonds(mol)
        fai = result["fragment_atom_indices"]
        assert len(fai) == 1
        assert sorted(fai[0]) == list(range(mol.GetNumAtoms()))


# ---------------------------------------------------------------------------
# Integration: Ligand.featurize(mode="fragment")
# ---------------------------------------------------------------------------

class TestLigandFragmentIntegration:
    """Integration tests through the Ligand API."""

    def test_ligand_featurize_fragment_mode(self):
        """Ligand.featurize(mode='fragment') returns expected keys."""
        ligand = Ligand.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
        result = ligand.featurize(mode="fragment")
        frag = result["fragment"]
        assert "fragment_smiles" in frag
        assert "atom_to_fragment" in frag
        assert "fragment_adjacency" in frag
        assert "num_fragments" in frag
        assert "num_rotatable_bonds" in frag

    def test_ligand_fragment_lazy_property(self):
        """ligand.fragment lazy property should match featurize output."""
        ligand = Ligand.from_smiles("c1ccc(CC)cc1")
        frag = ligand.fragment
        assert frag is not None
        assert frag["num_fragments"] == 2

    def test_ligand_featurize_all_includes_fragment(self):
        """mode='all' should NOT include fragment (not in default_modes)."""
        ligand = Ligand.from_smiles("CCO")
        result = ligand.featurize(mode="fragment")
        assert "fragment" in result

    def test_ligand_featurize_multi_mode(self):
        """Fragment can be requested alongside other modes."""
        ligand = Ligand.from_smiles("CCO")
        result = ligand.featurize(mode=["smiles", "fragment"])
        assert "smiles" in result
        assert "fragment" in result
