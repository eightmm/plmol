"""Tests for Watson-Crick base pair detection.

The repository's DNA fixture is a single strand, so there is nothing to pair.
These build the geometry instead: a planar A.T with the canonical bond lengths,
then the ways it can fail -- pulled apart, twisted out of plane, given the
wrong partner.
"""

import numpy as np
import pytest

from plmol.constants import WC_HBOND_ATOMS
from plmol.nucleic_acid.base_pairs import BasePair, base_pair_arrays, find_base_pairs

# Ring atoms laid out in the z = 0 plane with the Watson-Crick edges facing
# each other: N1(A)-N3(T) at 2.82 A and N6(A)-O4(T) at 2.73, which is what a
# real pair measures.
ADENINE = {
    "N1": (0.0, 0.0, 0.0), "C6": (-0.40, 1.35, 0.0), "N6": (0.20, 2.50, 0.0),
    "C5": (-1.80, 1.55, 0.0), "C4": (-2.55, 0.40, 0.0), "N3": (-2.20, -0.90, 0.0),
    "C2": (-0.90, -1.05, 0.0), "N7": (-2.65, 2.65, 0.0), "C8": (-3.90, 2.15, 0.0),
    "N9": (-3.85, 0.80, 0.0), "C1'": (-5.00, 0.00, 0.0),
}
THYMINE = {
    "N3": (2.82, 0.0, 0.0), "C4": (3.50, 1.15, 0.0), "O4": (2.92, 2.25, 0.0),
    "C5": (4.90, 1.10, 0.0), "C5M": (5.65, 2.35, 0.0), "C6": (5.50, -0.10, 0.0),
    "N1": (4.85, -1.25, 0.0), "C2": (3.50, -1.25, 0.0), "O2": (2.90, -2.30, 0.0),
    "C1'": (5.60, -2.50, 0.0),
}


def residue(name, atoms, number=1, offset=(0.0, 0.0, 0.0), rotation=None):
    placed = {}
    for atom, xyz in atoms.items():
        vector = np.array(xyz, dtype=np.float32)
        if rotation is not None:
            vector = rotation @ vector
        placed[atom] = tuple(vector + np.array(offset, dtype=np.float32))
    return {"chain_id": "A", "res_num": number, "res_name": name, "atoms": placed}


def rotation_about_x(degrees):
    angle = np.radians(degrees)
    return np.array(
        [[1, 0, 0],
         [0, np.cos(angle), -np.sin(angle)],
         [0, np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )


def pair_residues(**kwargs):
    return [residue("DA", ADENINE, 1), residue("DT", THYMINE, 2, **kwargs)]


class TestARealPair:
    def test_a_canonical_at_pair_is_found(self):
        pairs = find_base_pairs(pair_residues())
        assert len(pairs) == 1
        assert pairs[0].kind == "AT"
        assert pairs[0].purine_index == 0 and pairs[0].pyrimidine_index == 1

    def test_the_bond_lengths_are_reported(self):
        pair = find_base_pairs(pair_residues())[0]
        assert len(pair.hbond_distances) == len(WC_HBOND_ATOMS[("A", "T")])
        assert abs(pair.hbond_distances[0] - 2.82) < 0.01
        assert all(distance < 3.0 for distance in pair.hbond_distances)

    def test_the_bases_come_out_coplanar(self):
        assert find_base_pairs(pair_residues())[0].plane_angle < 1.0

    def test_the_c1_separation_is_the_canonical_one(self):
        """A Watson-Crick pair holds its sugars about 10.5 A apart."""
        assert abs(find_base_pairs(pair_residues())[0].c1_distance - 10.5) < 1.0


class TestWhatIsNotAPair:
    def test_bases_pulled_apart_are_not_paired(self):
        assert find_base_pairs(pair_residues(offset=(6.0, 0.0, 0.0))) == []

    def test_a_base_twisted_out_of_plane_is_not_paired(self):
        assert find_base_pairs(pair_residues(rotation=rotation_about_x(90))) == []

    def test_the_wrong_partner_is_not_paired(self):
        """G does not pair with T, however close they are."""
        guanine = dict(ADENINE)
        assert find_base_pairs([residue("DG", guanine, 1), residue("DT", THYMINE, 2)]) == []

    def test_one_hydrogen_bond_alone_is_a_contact_not_a_pair(self):
        """Only the anchor in range: a close approach, not Watson-Crick."""
        thymine = dict(THYMINE)
        thymine["O4"] = (2.92, 8.0, 0.0)
        assert find_base_pairs([residue("DA", ADENINE, 1), residue("DT", thymine, 2)]) == []

    def test_a_residue_without_its_ring_is_skipped(self):
        bare = {"N1": (0.0, 0.0, 0.0)}
        assert find_base_pairs([residue("DA", bare, 1), residue("DT", THYMINE, 2)]) == []

    def test_an_empty_list_pairs_nothing(self):
        assert find_base_pairs([]) == []


class TestOnePartnerEach:
    def test_a_base_is_used_once(self):
        """Two thymines near one adenine: the closer one wins, not both."""
        residues = [
            residue("DA", ADENINE, 1),
            residue("DT", THYMINE, 2),
            residue("DT", THYMINE, 3, offset=(0.3, 0.0, 0.0)),
        ]
        pairs = find_base_pairs(residues)
        assert len(pairs) == 1
        assert pairs[0].pyrimidine_index == 1        # the shorter anchor bond

    def test_two_separate_pairs_are_both_found(self):
        residues = pair_residues() + [
            residue("DA", ADENINE, 3, offset=(40.0, 0.0, 0.0)),
            residue("DT", THYMINE, 4, offset=(40.0, 0.0, 0.0)),
        ]
        pairs = find_base_pairs(residues)
        assert len(pairs) == 2
        assert [p.purine_index for p in pairs] == [0, 2]

    def test_pairs_come_back_in_residue_order(self):
        residues = [
            residue("DA", ADENINE, 1, offset=(40.0, 0.0, 0.0)),
            residue("DT", THYMINE, 2, offset=(40.0, 0.0, 0.0)),
        ] + pair_residues()
        pairs = find_base_pairs(residues)
        assert [p.purine_index for p in pairs] == sorted(p.purine_index for p in pairs)


class TestArrays:
    def test_the_arrays_describe_the_pairs(self):
        pairs = find_base_pairs(pair_residues())
        arrays = base_pair_arrays(pairs, num_residues=2)
        assert arrays["pair_index"].shape == (2, 1)
        assert arrays["pair_kind"].tolist() == [0]
        assert arrays["is_paired"].tolist() == [1.0, 1.0]

    def test_no_pairs_still_gives_the_right_shapes(self):
        arrays = base_pair_arrays([], num_residues=5)
        assert arrays["pair_index"].shape == (2, 0)
        assert arrays["is_paired"].shape == (5,)
        assert not arrays["is_paired"].any()


class TestInTheGraph:
    def test_the_graph_carries_the_pairing(self, dna_pdb):
        from plmol import NucleicAcid

        graph = NucleicAcid.from_pdb(dna_pdb).featurize(mode="graph")["graph"]
        for key in ("pair_index", "pair_kind", "pair_c1_distance",
                    "pair_plane_angle", "is_paired"):
            assert key in graph, key
        assert graph["is_paired"].shape == (graph["num_nodes"],)

    def test_a_single_strand_has_no_pairs(self, dna_pdb):
        """The fixture is one strand, so the honest answer is none."""
        from plmol import NucleicAcid

        graph = NucleicAcid.from_pdb(dna_pdb).featurize(mode="graph")["graph"]
        assert graph["pair_index"].shape == (2, 0)
        assert not graph["is_paired"].any()

    def test_the_edge_features_keep_their_width(self, dna_pdb):
        """Pairing is reported alongside the edges, not folded into them."""
        from plmol import NucleicAcid, feature_dims

        graph = NucleicAcid.from_pdb(dna_pdb).featurize(mode="graph")["graph"]
        assert graph["edge_attr"].shape[-1] == feature_dims("nucleic_acid", "graph")["edge_features"]
