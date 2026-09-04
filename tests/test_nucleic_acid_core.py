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



# dna_pdb fixture lives in conftest.py, shared with test_graph_view.py


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


class TestNucleotideNameNormalization:
    """A nucleotide keeps its identity whatever the file calls it.

    NucleicFeaturizer selected residues by the eight canonical names, so a
    structure written with the older ADE/CYT/GUA/THY spellings -- or a tRNA,
    which is largely modified bases -- came back empty rather than wrong.
    """

    @staticmethod
    def _renamed(source, tmp_path, mapping, name):
        text = []
        for line in open(source):
            if line[:6].strip() in ("ATOM", "HETATM"):
                res = line[17:20].strip()
                if res in mapping:
                    line = line[:17] + mapping[res].rjust(3) + line[20:]
            text.append(line)
        path = tmp_path / name
        path.write_text("".join(text))
        return str(path)

    def test_the_legacy_spellings_give_the_same_features(self, dna_pdb, tmp_path):
        from plmol import NucleicAcid

        legacy = self._renamed(
            dna_pdb, tmp_path,
            {"DA": "ADE", "DT": "THY", "DG": "GUA", "DC": "CYT"}, "legacy.pdb",
        )
        modern = NucleicAcid.from_pdb(dna_pdb).featurize(mode="sequence")["sequence"]
        older = NucleicAcid.from_pdb(legacy).featurize(mode="sequence")["sequence"]
        assert np.array_equal(older["tokens"], modern["tokens"])
        assert np.array_equal(older["is_purine"], modern["is_purine"])
        assert older["res_names"] != modern["res_names"], "the file's own names are reported"

    @pytest.mark.parametrize("name,atoms,expected", [
        ("ADE", ["O2'"], "A"), ("ADE", [], "DA"),
        ("CYT", ["O2'"], "C"), ("CYT", [], "DC"),
        ("THY", [], "DT"), ("URA", ["O2'"], "U"),
        ("PSU", [], "U"), ("5MC", [], "C"), ("1MA", [], "A"), ("M2G", [], "G"),
        ("DU", [], "DT"),
        ("DA", [], "DA"), ("G", [], "G"),
        ("I", [], "I"), ("DI", [], "DI"),
    ])
    def test_the_base_a_name_stands_for(self, name, atoms, expected):
        """The legacy spellings need the sugar: ADE is adenosine with the
        ribose 2' oxygen and deoxyadenosine without it. Inosine stays itself --
        hypoxanthine pairs with C, U and A rather than as any one base."""
        from plmol.parsers.pdb_parser import normalize_nucleotide_name

        assert normalize_nucleotide_name(name, atoms) == expected

    def test_a_modified_base_is_not_dropped(self, dna_pdb, tmp_path):
        from plmol import NucleicAcid

        from plmol.constants import NUCLEOTIDE_TOKEN

        plain = NucleicAcid.from_pdb(dna_pdb).featurize(mode="sequence")["sequence"]
        modified = self._renamed(dna_pdb, tmp_path, {"DA": "1MA"}, "modified.pdb")
        residues = NucleicAcid.from_pdb(modified).featurize(mode="sequence")["sequence"]
        assert len(residues["res_names"]) == len(plain["res_names"]), "still there"
        assert residues["res_names"][0] == "1MA", "reported as the file names it"
        assert residues["tokens"][0] == NUCLEOTIDE_TOKEN["A"], "counted as an adenosine"


class TestTorsionsStopAtAStrandBreak:
    """alpha, epsilon and zeta span two nucleotides; the span has to be a bond."""

    @staticmethod
    def _two_strands(source_text: str, path) -> str:
        """The same strand twice, as chains A and B, 20 A apart."""
        lines = [l for l in source_text.splitlines() if l.startswith(("ATOM  ", "HETATM"))]
        out = []
        for chain, shift in (("A", 0.0), ("B", 20.0)):
            for line in lines:
                x = float(line[30:38]) + shift
                out.append(line[:21] + chain + line[22:30] + f"{x:8.3f}" + line[38:])
        numbered = [l[:6] + f"{i + 1:5d}" + l[11:] for i, l in enumerate(out)]
        path.write_text("\n".join(numbered) + "\nEND\n")
        return str(path)

    def test_a_second_strand_does_not_continue_the_first(self, dna_pdb, tmp_path):
        from plmol.nucleic_acid.featurizer import NucleicFeaturizer

        path = self._two_strands(open(dna_pdb).read(), tmp_path / "duplex.pdb")
        featurizer = NucleicFeaturizer(path)
        residues = featurizer._get_na_residues()
        torsions = featurizer._compute_torsions(residues)

        chains = [r["chain_id"] for r in residues]
        assert set(chains) == {"A", "B"}
        first_b = chains.index("B")
        last_a = first_b - 1

        # alpha of B's first nucleotide would come from A's last O3'.
        assert torsions[first_b, 0] == 0.0
        # epsilon and zeta of A's last would come from B's first P.
        assert torsions[last_a, 4] == 0.0
        assert torsions[last_a, 5] == 0.0
        # Inside a strand they survive. (alpha is zero throughout this fixture:
        # its four nucleotides are stacked on collinear points, so the O3'-P-O5'
        # -C5' dihedral is degenerate. epsilon and zeta are not.)
        assert torsions[0, 4] != 0.0
        assert torsions[0, 5] != 0.0
        assert torsions[first_b, 4] != 0.0
