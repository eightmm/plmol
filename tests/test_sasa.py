"""Tests for solvent accessible surface area.

plmol computes SASA itself as of 0.4.0 -- there is no backend to choose. These
pin the algorithm against its definition and against the failure it used to
have: a missing dependency turning the SASA block into zeros and every
burial_index into 0.5, handed back as features.
"""

import numpy as np
import pytest

from plmol import InputError, Protein, shrake_rupley


# -- The algorithm ------------------------------------------------------------


class TestShrakeRupley:
    def test_a_lone_atom_is_a_full_sphere(self):
        radius, probe = 1.7, 1.4
        area = shrake_rupley(np.zeros((1, 3)), np.array([radius]), probe_radius=probe)
        assert np.isclose(area[0], 4 * np.pi * (radius + probe) ** 2)

    def test_two_touching_atoms_lose_area(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.7, 1.7])
        areas = shrake_rupley(coords, radii)
        lone = shrake_rupley(coords[:1], radii[:1])[0]
        assert areas[0] < lone and areas[1] < lone
        # Symmetric atoms differ by at most a sample or two: the Fibonacci
        # lattice is fixed in space, so each sphere loses its own points.
        quantum = lone / 100.0
        assert abs(areas[0] - areas[1]) <= 2 * quantum

    def test_far_apart_atoms_are_independent(self):
        radii = np.array([1.7, 1.7])
        lone = shrake_rupley(np.zeros((1, 3)), radii[:1])[0]
        far = shrake_rupley(np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]), radii)
        assert np.allclose(far, lone)

    def test_a_fully_engulfed_atom_has_no_area(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        radii = np.array([0.3, 3.0])
        areas = shrake_rupley(coords, radii)
        assert areas[0] == 0.0
        assert areas[1] > 0.0

    def test_empty_input(self):
        assert shrake_rupley(np.zeros((0, 3)), np.zeros(0)).shape == (0,)

    def test_bad_shape_is_rejected(self):
        with pytest.raises(InputError, match=r"\(N, 3\)"):
            shrake_rupley(np.zeros((4, 2)), np.zeros(4))

    def test_matches_a_brute_force_occlusion(self):
        """Every neighbour counts, not just the nearest few.

        The occlusion test used to look at a fixed number of nearest atoms,
        which quietly overestimated areas in a crowded pocket. This compares it
        to the definition: a sample point survives unless *some* other sphere
        covers it, checked against all of them.
        """
        from plmol.surface.point_cloud import _fibonacci_sphere

        rng = np.random.default_rng(7)
        coords = rng.normal(scale=2.0, size=(40, 3)).astype(np.float32)
        radii = rng.uniform(1.2, 1.9, size=40).astype(np.float32)

        expanded = radii + 1.4
        sphere = _fibonacci_sphere(100)
        expected = np.empty(len(coords))
        for i in range(len(coords)):
            points = coords[i] + expanded[i] * sphere
            covered = np.zeros(len(sphere), dtype=bool)
            for j in range(len(coords)):
                if j != i:
                    covered |= np.linalg.norm(points - coords[j], axis=1) < expanded[j]
            expected[i] = 4 * np.pi * expanded[i] ** 2 * (~covered).mean()

        assert np.allclose(shrake_rupley(coords, radii), expected)


# -- Backend selection --------------------------------------------------------


class TestResults:
    def test_structure_and_result_expose_what_the_featurizers_read(self, example_pdb):
        from plmol.sasa import native_structure_result

        structure, result = native_structure_result(example_pdb)
        assert structure.nAtoms() == result.nAtoms() > 0
        assert result.totalArea() > 0
        assert isinstance(structure.residueNumber(0), str)
        areas = result.residueAreas()
        assert areas and all(isinstance(chain, str) for chain in areas)
        first = next(iter(next(iter(areas.values())).values()))
        assert first.total >= 0
        assert np.isclose(first.total, first.polar + first.apolar, atol=1e-6)
        assert np.isclose(first.total, first.mainChain + first.sideChain, atol=1e-6)

    def test_relative_values_are_fractions_not_percentages(self, example_pdb):
        from plmol.sasa import native_structure_result

        _, result = native_structure_result(example_pdb)
        values = [v for chain in result.residueAreas().values() for v in chain.values()]
        relative = np.array([v.relativeTotal for v in values])
        assert relative.max() <= 2.0, "fractions, matching SASA's convention"

    def test_polar_classification_matches_the_element_rule(self):
        from plmol.sasa import is_polar_element

        assert is_polar_element("N") and is_polar_element("O") and is_polar_element("S")
        assert not is_polar_element("C")
        assert not is_polar_element("")


class TestFeaturesAreReal:
    """The behaviour this module exists to fix."""

    def test_residue_sasa_is_not_zeros(self, example_pdb):
        graph = Protein.from_pdb(example_pdb).featurize(mode="graph")["graph"]
        block = [t for t in graph["node_features"] if t.shape[-1] == 12][0]
        assert np.abs(block).sum() > 0
        assert block.std() > 0.01

    def test_burial_index_is_not_a_constant(self, example_pdb):
        atom_graph = Protein.from_pdb(example_pdb).featurize(mode="atom_graph")["atom_graph"]
        burial = np.asarray(atom_graph["burial_index"])
        assert not np.allclose(burial, 0.5)
        assert burial.std() > 0.01


class TestResidueBurialIndexIsInformative:
    """Guards the scale fix: relativeTotal is a fraction, not a percentage."""

    def test_burial_column_spans_a_real_range(self, example_pdb):
        graph = Protein.from_pdb(example_pdb).featurize(mode="graph")["graph"]
        burial = [t for t in graph["node_features"] if t.shape[-1] == 12][0][:, 10]
        assert burial.std() > 0.1, "a near-constant column means the /100 bug is back"
        assert burial.max() > 0.9 and burial.min() < 0.5


class TestEveryModeProducesSasa:
    """Every mode that derives something from SASA gets real numbers.

    This is the shape of the failure 0.2.x had when freesasa was missing:
    zeros in the residue block and 0.5 everywhere else, handed back as
    features rather than raised.
    """

    def test_residue_graph(self, example_pdb):
        graph = Protein.from_pdb(example_pdb).featurize(mode="graph")["graph"]
        block = [t for t in graph["node_features"] if t.shape[-1] == 12][0]
        assert block.shape[0] > 0 and np.abs(block).sum() > 0

    def test_atom_graph_keeps_every_atom(self, example_pdb):
        protein = Protein.from_pdb(example_pdb)
        atom_graph = protein.featurize(mode="atom_graph")["atom_graph"]
        coords = np.asarray(atom_graph["coords"])
        assert coords.shape[0] > 0
        for key in ("sasa", "relative_sasa", "burial_index", "is_polar_sasa"):
            assert np.asarray(atom_graph[key]).shape[0] == coords.shape[0]
        assert np.asarray(atom_graph["sasa"]).sum() > 0

    def test_surface(self, example_pdb):
        surface = Protein.from_pdb(example_pdb).featurize(mode="surface")["surface"]
        burial = np.asarray(surface["feature_dict"]["burial_index"])
        assert burial.size > 0 and burial.std() > 0.01

    def test_voxel(self, example_pdb):
        voxel = Protein.from_pdb(example_pdb).featurize(mode="voxel")["voxel"]
        assert np.asarray(voxel["voxel"]).shape[0] == 16
