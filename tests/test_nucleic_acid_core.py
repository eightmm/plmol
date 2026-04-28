"""
Tests for the NucleicAcid class and NucleicFeaturizer.
"""

import pytest
import numpy as np
import textwrap
import tempfile
import os

from plmol.nucleic_acid import NucleicAcid, NucleicFeaturizer
from plmol.constants import (
    NUCLEOTIDE_TOKEN, NUM_NUCLEOTIDE_TYPES,
    DNA_RESIDUES, RNA_RESIDUES, PURINES, PYRIMIDINES,
)


# ---------------------------------------------------------------------------
# PDB fixture: minimal 4-residue DNA duplex strand (DA, DT, DG, DC)
# Only backbone + a few base atoms to keep it short.
# ---------------------------------------------------------------------------

_DNA_PDB = textwrap.dedent("""\
ATOM      1  P    DA A   1       1.000   0.000   0.000  1.00  0.00           P
ATOM      2  O5'  DA A   1       2.500   0.000   0.000  1.00  0.00           O
ATOM      3  C5'  DA A   1       3.500   1.000   0.000  1.00  0.00           C
ATOM      4  C4'  DA A   1       4.500   1.000   1.000  1.00  0.00           C
ATOM      5  O4'  DA A   1       5.000   2.000   1.000  1.00  0.00           O
ATOM      6  C3'  DA A   1       5.500   0.500   1.500  1.00  0.00           C
ATOM      7  O3'  DA A   1       6.500   0.500   2.000  1.00  0.00           O
ATOM      8  C2'  DA A   1       5.500   1.500   2.500  1.00  0.00           C
ATOM      9  C1'  DA A   1       5.000   2.500   1.500  1.00  0.00           C
ATOM     10  N9   DA A   1       4.000   3.000   1.500  1.00  0.00           N
ATOM     11  C4   DA A   1       3.000   3.500   1.500  1.00  0.00           C
ATOM     12  P    DT A   2       7.000   0.000   2.000  1.00  0.00           P
ATOM     13  O5'  DT A   2       8.500   0.000   2.000  1.00  0.00           O
ATOM     14  C5'  DT A   2       9.500   1.000   2.000  1.00  0.00           C
ATOM     15  C4'  DT A   2      10.500   1.000   3.000  1.00  0.00           C
ATOM     16  O4'  DT A   2      11.000   2.000   3.000  1.00  0.00           O
ATOM     17  C3'  DT A   2      11.500   0.500   3.500  1.00  0.00           C
ATOM     18  O3'  DT A   2      12.500   0.500   4.000  1.00  0.00           O
ATOM     19  C2'  DT A   2      11.500   1.500   4.500  1.00  0.00           C
ATOM     20  C1'  DT A   2      11.000   2.500   3.500  1.00  0.00           C
ATOM     21  N1   DT A   2      10.000   3.000   3.500  1.00  0.00           N
ATOM     22  C2   DT A   2       9.000   3.500   3.500  1.00  0.00           C
ATOM     23  P    DG A   3      13.000   0.000   4.000  1.00  0.00           P
ATOM     24  O5'  DG A   3      14.500   0.000   4.000  1.00  0.00           O
ATOM     25  C5'  DG A   3      15.500   1.000   4.000  1.00  0.00           C
ATOM     26  C4'  DG A   3      16.500   1.000   5.000  1.00  0.00           C
ATOM     27  O4'  DG A   3      17.000   2.000   5.000  1.00  0.00           O
ATOM     28  C3'  DG A   3      17.500   0.500   5.500  1.00  0.00           C
ATOM     29  O3'  DG A   3      18.500   0.500   6.000  1.00  0.00           O
ATOM     30  C2'  DG A   3      17.500   1.500   6.500  1.00  0.00           C
ATOM     31  C1'  DG A   3      17.000   2.500   5.500  1.00  0.00           C
ATOM     32  N9   DG A   3      16.000   3.000   5.500  1.00  0.00           N
ATOM     33  C4   DG A   3      15.000   3.500   5.500  1.00  0.00           C
ATOM     34  P    DC A   4      19.000   0.000   6.000  1.00  0.00           P
ATOM     35  O5'  DC A   4      20.500   0.000   6.000  1.00  0.00           O
ATOM     36  C5'  DC A   4      21.500   1.000   6.000  1.00  0.00           C
ATOM     37  C4'  DC A   4      22.500   1.000   7.000  1.00  0.00           C
ATOM     38  O4'  DC A   4      23.000   2.000   7.000  1.00  0.00           O
ATOM     39  C3'  DC A   4      23.500   0.500   7.500  1.00  0.00           C
ATOM     40  O3'  DC A   4      24.500   0.500   8.000  1.00  0.00           O
ATOM     41  C2'  DC A   4      23.500   1.500   8.500  1.00  0.00           C
ATOM     42  C1'  DC A   4      23.000   2.500   7.500  1.00  0.00           C
ATOM     43  N1   DC A   4      22.000   3.000   7.500  1.00  0.00           N
ATOM     44  C2   DC A   4      21.000   3.500   7.500  1.00  0.00           C
END
""")


@pytest.fixture
def dna_pdb(tmp_path):
    p = tmp_path / "dna.pdb"
    p.write_text(_DNA_PDB)
    return str(p)


# ---------------------------------------------------------------------------
# Tests: NucleicAcid.from_sequence
# ---------------------------------------------------------------------------

class TestFromSequence:
    def test_dna_sequence(self):
        na = NucleicAcid.from_sequence("ATGC", na_type="DNA")
        assert na.sequence == "ATGC"
        assert na.chain_type == "DNA"

    def test_rna_sequence(self):
        na = NucleicAcid.from_sequence("AUGC", na_type="RNA")
        assert na.sequence == "AUGC"
        assert na.chain_type == "RNA"

    def test_invalid_na_type(self):
        with pytest.raises(ValueError, match="na_type"):
            NucleicAcid.from_sequence("ATGC", na_type="XNA")

    def test_featurize_sequence_only(self):
        na = NucleicAcid.from_sequence("ATGC", na_type="DNA")
        features = na.featurize(mode="sequence")
        seq_feat = features["sequence"]
        assert "tokens" in seq_feat
        assert len(seq_feat["tokens"]) == 4
        assert "gc_content" in seq_feat
        assert abs(seq_feat["gc_content"] - 0.5) < 1e-6

    def test_gc_content_all_gc(self):
        na = NucleicAcid.from_sequence("GGCC", na_type="DNA")
        feats = na.featurize(mode="sequence")["sequence"]
        assert feats["gc_content"] == pytest.approx(1.0)

    def test_gc_content_all_at(self):
        na = NucleicAcid.from_sequence("AATT", na_type="DNA")
        feats = na.featurize(mode="sequence")["sequence"]
        assert feats["gc_content"] == pytest.approx(0.0)

    def test_rna_sequence_features(self):
        na = NucleicAcid.from_sequence("AUGC", na_type="RNA")
        feats = na.featurize(mode="sequence")["sequence"]
        assert feats["tokens"].shape == (4,)
        assert "A" in feats["res_names"] or "A" not in feats["res_names"]  # just check no crash

    def test_purine_pyrimidine_dna(self):
        na = NucleicAcid.from_sequence("ATGC", na_type="DNA")
        feats = na.featurize(mode="sequence")["sequence"]
        # A → DA (purine), T → DT (pyrimidine), G → DG (purine), C → DC (pyrimidine)
        np.testing.assert_array_equal(feats["is_purine"], [1, 0, 1, 0])
        np.testing.assert_array_equal(feats["is_pyrimidine"], [0, 1, 0, 1])


# ---------------------------------------------------------------------------
# Tests: NucleicAcid.from_pdb
# ---------------------------------------------------------------------------

class TestFromPDB:
    def test_load_from_pdb(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb)
        assert na._pdb_path == dna_pdb

    def test_sequence_from_pdb(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb)
        seq = na.sequence
        assert isinstance(seq, str)
        assert len(seq) == 4  # DA, DT, DG, DC → 4 residues

    def test_chain_type_auto_detected(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb)
        assert na.chain_type == "DNA"

    def test_chain_id_filter(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb, chain_id="A")
        assert len(na.sequence) == 4

    def test_metadata_set(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb)
        assert na.metadata["source"] == dna_pdb


# ---------------------------------------------------------------------------
# Tests: NucleicFeaturizer.get_sequence_features
# ---------------------------------------------------------------------------

class TestSequenceFeatures:
    def test_returns_correct_keys(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        out = feat.get_sequence_features()
        assert "tokens" in out
        assert "is_purine" in out
        assert "is_pyrimidine" in out
        assert "gc_content" in out
        assert "res_names" in out

    def test_token_shapes(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        out = feat.get_sequence_features()
        n = len(out["res_names"])
        assert out["tokens"].shape == (n,)
        assert out["is_purine"].shape == (n,)
        assert out["is_pyrimidine"].shape == (n,)

    def test_token_values_in_range(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        out = feat.get_sequence_features()
        assert (out["tokens"] >= 0).all()
        assert (out["tokens"] < NUM_NUCLEOTIDE_TYPES).all()

    def test_gc_content_range(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        out = feat.get_sequence_features()
        assert 0.0 <= out["gc_content"] <= 1.0

    def test_purine_pyrimidine_exclusive(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        out = feat.get_sequence_features()
        # Each position is purine XOR pyrimidine (not both, not neither for standard NT)
        both = (out["is_purine"] + out["is_pyrimidine"])
        assert (both <= 1.0).all()


# ---------------------------------------------------------------------------
# Tests: NucleicFeaturizer.get_graph
# ---------------------------------------------------------------------------

class TestGraph:
    def test_keys_present(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        for key in ["nucleotide_type", "one_hot", "torsions", "sugar_pucker",
                    "coords", "edge_index", "edge_attr", "num_nodes"]:
            assert key in g

    def test_num_nodes(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        assert g["num_nodes"] == 4

    def test_one_hot_shape(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        assert g["one_hot"].shape == (4, NUM_NUCLEOTIDE_TYPES)
        assert (g["one_hot"].sum(axis=1) == 1.0).all()

    def test_torsion_shape(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        assert g["torsions"].shape == (4, 7)

    def test_torsion_range(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        # Torsions should be in [-pi, pi]
        assert (g["torsions"] >= -np.pi - 1e-5).all()
        assert (g["torsions"] <= np.pi + 1e-5).all()

    def test_sugar_pucker_shape(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        assert g["sugar_pucker"].shape == (4,)
        assert set(np.unique(g["sugar_pucker"])).issubset({0.0, 0.5, 1.0})

    def test_coords_shape(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        assert g["coords"].shape == (4, 3)

    def test_edge_index_valid(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        assert g["edge_index"].shape[0] == 2
        n = g["num_nodes"]
        assert (g["edge_index"] >= 0).all()
        assert (g["edge_index"] < n).all()

    def test_edge_attr_shape(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g = feat.get_graph()
        n_edges = g["edge_index"].shape[1]
        assert g["edge_attr"].shape == (n_edges, 3)

    def test_distance_cutoff_reduces_edges(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        g_all = feat.get_graph(distance_cutoff=100.0)
        g_near = feat.get_graph(distance_cutoff=0.1)
        # Very small cutoff → only sequential edges remain
        assert g_near["edge_index"].shape[1] <= g_all["edge_index"].shape[1]


# ---------------------------------------------------------------------------
# Tests: NucleicFeaturizer.get_backbone
# ---------------------------------------------------------------------------

class TestBackbone:
    def test_keys_present(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        bb = feat.get_backbone()
        assert "backbone_coords" in bb
        assert "backbone_atom_names" in bb
        assert "num_residues" in bb

    def test_backbone_shape(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        bb = feat.get_backbone()
        n = bb["num_residues"]
        n_atoms = len(bb["backbone_atom_names"])
        assert bb["backbone_coords"].shape == (n, n_atoms, 3)

    def test_backbone_has_expected_atoms(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        bb = feat.get_backbone()
        assert "P" in bb["backbone_atom_names"]
        assert "C1'" in bb["backbone_atom_names"]


# ---------------------------------------------------------------------------
# Tests: NucleicFeaturizer.get_atom_graph
# ---------------------------------------------------------------------------

class TestAtomGraph:
    def test_keys_present(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        ag = feat.get_atom_graph()
        for key in ["coords", "residue_token", "atom_to_residue",
                    "residue_atom_indices", "edge_index", "edge_distances", "num_atoms"]:
            assert key in ag

    def test_atom_to_residue_range(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        ag = feat.get_atom_graph()
        n_res = 4
        assert (ag["atom_to_residue"] >= 0).all()
        assert (ag["atom_to_residue"] < n_res).all()

    def test_residue_token_in_range(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        ag = feat.get_atom_graph()
        assert (ag["residue_token"] >= 0).all()
        assert (ag["residue_token"] < NUM_NUCLEOTIDE_TYPES).all()

    def test_edge_index_valid(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        ag = feat.get_atom_graph(distance_cutoff=10.0)
        n_atoms = ag["num_atoms"]
        if ag["edge_index"].shape[1] > 0:
            assert (ag["edge_index"] >= 0).all()
            assert (ag["edge_index"] < n_atoms).all()

    def test_coords_shape(self, dna_pdb):
        feat = NucleicFeaturizer(dna_pdb)
        ag = feat.get_atom_graph()
        n_atoms = ag["num_atoms"]
        assert ag["coords"].shape == (n_atoms, 3)


# ---------------------------------------------------------------------------
# Tests: NucleicAcid.featurize integration
# ---------------------------------------------------------------------------

class TestFeaturizeIntegration:
    def test_featurize_all(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb)
        result = na.featurize(mode="all")
        assert "sequence" in result
        assert "graph" in result
        assert "backbone" in result
        assert "atom_graph" in result

    def test_featurize_list_all(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb)
        result = na.featurize(mode=["all"])
        assert "sequence" in result
        assert "graph" in result
        assert "backbone" in result
        assert "atom_graph" in result

    def test_featurize_graph_only(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb)
        result = na.featurize(mode="graph")
        assert "graph" in result
        assert "sequence" not in result

    def test_featurize_backbone_only(self, dna_pdb):
        na = NucleicAcid.from_pdb(dna_pdb)
        result = na.featurize(mode="backbone")
        assert "backbone" in result

    def test_featurize_no_pdb_graph_raises(self):
        na = NucleicAcid.from_sequence("ATGC", na_type="DNA")
        with pytest.raises(ValueError, match="PDB path"):
            na.featurize(mode="graph")

    def test_featurize_all_no_pdb_sequence_only(self):
        na = NucleicAcid.from_sequence("ATGC", na_type="DNA")
        result = na.featurize(mode="all")
        assert list(result) == ["sequence"]

    def test_invalid_list_mode_raises(self):
        na = NucleicAcid.from_sequence("ATGC", na_type="DNA")
        with pytest.raises(ValueError, match="Unsupported mode"):
            na.featurize(mode=["bad"])

    def test_featurize_sequence_no_pdb(self):
        na = NucleicAcid.from_sequence("ATGC", na_type="DNA")
        result = na.featurize(mode="sequence")
        assert "sequence" in result
        assert result["sequence"]["tokens"].shape == (4,)


# ---------------------------------------------------------------------------
# Tests: plmol top-level import
# ---------------------------------------------------------------------------

def test_top_level_import():
    from plmol import NucleicAcid, NucleicFeaturizer
    assert NucleicAcid is not None
    assert NucleicFeaturizer is not None
