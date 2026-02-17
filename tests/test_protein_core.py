"""Tests for plmol/protein/core.py — Protein class."""

import pytest
import torch

from plmol.protein.core import Protein
from plmol.protein.utils import PDBParser


class TestProteinFromSequence:
    def test_basic(self):
        p = Protein.from_sequence("ACDEFG")
        assert p.sequence == "ACDEFG"

    def test_no_pdb_path(self):
        p = Protein.from_sequence("ACDEFG")
        assert p._pdb_path is None

    def test_featurize_sequence_only(self):
        p = Protein.from_sequence("ACDEFG")
        result = p.featurize(mode="sequence")
        assert "sequence" in result
        assert result["sequence"] == "ACDEFG"

    def test_graph_without_pdb_raises(self):
        p = Protein.from_sequence("ACDEFG")
        with pytest.raises(ValueError, match="PDB"):
            p.featurize(mode="graph")


class TestProteinFromPDB:
    def test_from_pdb(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        assert p._pdb_path == mini_pdb
        assert p.metadata["source"] == mini_pdb

    def test_sequence(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        seq = p.sequence
        # Multi-chain returns dict
        assert isinstance(seq, dict)
        assert "A" in seq
        assert "B" in seq

    def test_featurize_graph_residue(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        result = p.featurize(mode="graph", graph_kwargs={"level": "residue"})
        assert "graph" in result
        graph = result["graph"]
        assert graph["level"] == "residue"
        assert "node_features" in graph
        assert "edge_index" in graph

    def test_featurize_graph_atom(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        result = p.featurize(mode="graph", graph_kwargs={"level": "atom"})
        assert "graph" in result
        graph = result["graph"]
        assert graph["level"] == "atom"

    def test_featurize_graph_atom_bidirectional_mapping(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        result = p.featurize(mode="graph", graph_kwargs={"level": "atom"})
        graph = result["graph"]
        # Both keys present
        assert "atom_to_residue" in graph
        assert "residue_atom_indices" in graph
        # atom_to_residue is the same tensor as residue_count
        assert graph["atom_to_residue"] is graph["residue_count"]
        # Reverse mapping is consistent with forward mapping
        for res_idx, atoms in enumerate(graph["residue_atom_indices"]):
            for a in atoms:
                assert graph["atom_to_residue"][a].item() == res_idx
        # All atoms are covered
        all_atoms = sorted(a for atoms in graph["residue_atom_indices"] for a in atoms)
        num_atoms = graph["atom_to_residue"].shape[0]
        assert all_atoms == list(range(num_atoms))

    def test_featurize_backbone(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        result = p.featurize(mode="backbone", backbone_kwargs={"k_neighbors": 3})
        assert "backbone" in result
        bb = result["backbone"]
        assert "backbone_coords" in bb
        assert "edge_index" in bb

    def test_featurize_multiple_modes(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        result = p.featurize(mode=["graph", "sequence", "backbone"])
        assert "graph" in result
        assert "sequence" in result
        assert "backbone" in result

    def test_lazy_graph_property(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        graph = p.graph
        assert graph is not None
        assert graph["level"] == "residue"

    def test_get_graph(self, mini_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(mini_pdb, standardize=False)
        graph = p.get_graph(level="residue")
        assert graph is not None


class TestProteinFromPDBReal:
    def test_real_pdb(self, example_pdb):
        PDBParser.clear_cache()
        p = Protein.from_pdb(example_pdb)
        seq = p.sequence
        assert seq is not None
        result = p.featurize(mode=["graph", "sequence"])
        assert "graph" in result
        assert "sequence" in result
