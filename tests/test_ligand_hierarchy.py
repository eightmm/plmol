"""Tests for atom ↔ fragment ↔ molecule hierarchical mappings."""

import numpy as np
import pytest

from plmol import Ligand


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def aspirin():
    """Aspirin: has rotatable bonds → multiple fragments."""
    return Ligand.from_smiles("CC(=O)Oc1ccccc1C(=O)O")


@pytest.fixture
def benzene():
    """Benzene: no rotatable bonds → single fragment."""
    return Ligand.from_smiles("c1ccccc1")


@pytest.fixture
def methane():
    """Methane: single-atom molecule (edge case)."""
    return Ligand.from_smiles("C")


# -- Graph dict hierarchical keys -------------------------------------------

class TestGraphHierarchyKeys:
    """Graph dict contains all hierarchical mapping keys."""

    def test_graph_has_fragment_keys(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        assert "atom_to_fragment" in graph
        assert "fragment_atom_indices" in graph
        assert "fragment_adjacency" in graph
        assert "num_fragments" in graph
        assert "molecule_features" in graph

    def test_atom_to_fragment_shape(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        n_atoms = graph["node_features"].shape[0]
        assert graph["atom_to_fragment"].shape == (n_atoms,)
        assert graph["atom_to_fragment"].dtype == np.int64

    def test_fragment_atom_indices_length(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        assert len(graph["fragment_atom_indices"]) == graph["num_fragments"]

    def test_fragment_adjacency_shape(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        f = graph["num_fragments"]
        assert graph["fragment_adjacency"].shape == (f, f)

    def test_molecule_features_shape(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        assert graph["molecule_features"].shape == (62,)
        assert graph["molecule_features"].dtype == np.float32


# -- Fragment dict fragment_features -----------------------------------------

class TestFragmentFeatures:
    """Fragment dict contains per-fragment descriptors."""

    def test_fragment_features_exists(self, aspirin):
        frag = aspirin.featurize(mode="fragment")["fragment"]
        assert "fragment_features" in frag

    def test_fragment_features_shape(self, aspirin):
        frag = aspirin.featurize(mode="fragment")["fragment"]
        f = frag["num_fragments"]
        assert frag["fragment_features"].shape == (f, 62)
        assert frag["fragment_features"].dtype == np.float32

    def test_fragment_features_finite(self, aspirin):
        frag = aspirin.featurize(mode="fragment")["fragment"]
        assert np.all(np.isfinite(frag["fragment_features"]))


# -- Consistency checks ------------------------------------------------------

class TestHierarchyConsistency:
    """Cross-level consistency between atom, fragment, and molecule."""

    def test_atom_count_matches(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        n_atoms = graph["node_features"].shape[0]
        assert graph["atom_to_fragment"].shape[0] == n_atoms

    def test_fragment_count_matches(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        frag = aspirin.featurize(mode="fragment")["fragment"]
        assert graph["num_fragments"] == frag["num_fragments"]
        assert frag["fragment_features"].shape[0] == frag["num_fragments"]

    def test_all_atoms_assigned(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        atf = graph["atom_to_fragment"]
        # All values should be valid fragment indices
        assert np.all(atf >= 0)
        assert np.all(atf < graph["num_fragments"])

    def test_fragment_atom_indices_cover_all_atoms(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        n_atoms = graph["node_features"].shape[0]
        all_atoms = sorted(
            a for group in graph["fragment_atom_indices"] for a in group
        )
        assert all_atoms == list(range(n_atoms))

    def test_atom_to_fragment_matches_fragment_atom_indices(self, aspirin):
        graph = aspirin.featurize(mode="graph")["graph"]
        atf = graph["atom_to_fragment"]
        for frag_idx, atom_list in enumerate(graph["fragment_atom_indices"]):
            for atom_idx in atom_list:
                assert atf[atom_idx] == frag_idx


# -- Edge cases --------------------------------------------------------------

class TestEdgeCases:
    """Single-atom and no-rotatable-bond molecules."""

    def test_single_atom_graph(self, methane):
        graph = methane.featurize(mode="graph")["graph"]
        assert graph["num_fragments"] == 1
        assert graph["atom_to_fragment"].shape == (1,)
        assert graph["atom_to_fragment"][0] == 0
        assert len(graph["fragment_atom_indices"]) == 1
        assert graph["fragment_adjacency"].shape == (1, 1)

    def test_single_atom_fragment(self, methane):
        frag = methane.featurize(mode="fragment")["fragment"]
        assert frag["num_fragments"] == 1
        assert frag["fragment_features"].shape == (1, 62)

    def test_no_rotatable_bonds_graph(self, benzene):
        graph = benzene.featurize(mode="graph")["graph"]
        assert graph["num_fragments"] == 1
        n_atoms = graph["node_features"].shape[0]
        assert graph["atom_to_fragment"].shape == (n_atoms,)
        assert np.all(graph["atom_to_fragment"] == 0)

    def test_no_rotatable_bonds_fragment(self, benzene):
        frag = benzene.featurize(mode="fragment")["fragment"]
        assert frag["num_fragments"] == 1
        assert frag["fragment_features"].shape == (1, 62)

    def test_molecule_features_finite_single_atom(self, methane):
        graph = methane.featurize(mode="graph")["graph"]
        assert np.all(np.isfinite(graph["molecule_features"]))
