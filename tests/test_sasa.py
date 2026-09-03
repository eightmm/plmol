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
        block = [t for t in graph["node_features"] if t.shape[-1] == 11][0]
        assert np.abs(block).sum() > 0
        assert block.std() > 0.01

    def test_burial_index_is_not_a_constant(self, example_pdb):
        atom_graph = Protein.from_pdb(example_pdb).featurize(mode="atom_graph")["atom_graph"]
        burial = np.asarray(atom_graph["burial_index"])
        assert not np.allclose(burial, 0.5)
        assert burial.std() > 0.01


class TestTheBlockCarriesWhatItClaims:
    """It was 12 columns holding four dimensions until 0.4.0: five of them were
    bit-identical to five others, because plmol normalises every class by the
    residue's total where freesasa used a separate reference for each."""

    def test_no_column_repeats_another(self, example_pdb):
        block = [t for t in
                 Protein.from_pdb(example_pdb).featurize(mode="graph")["graph"]["node_features"]
                 if t.shape[-1] == 11][0]
        for left in range(block.shape[1]):
            for right in range(left + 1, block.shape[1]):
                assert not np.array_equal(block[:, left], block[:, right]), (left, right)

    def test_the_named_columns_are_where_they_say(self, example_pdb):
        block = [t for t in
                 Protein.from_pdb(example_pdb).featurize(mode="graph")["graph"]["node_features"]
                 if t.shape[-1] == 11][0]
        (polar, apolar, main, side, rel_total, rel_polar, rel_apolar,
         rel_main, rel_side, burial, ratio) = block.T
        assert np.allclose(polar + apolar, main + side, atol=1e-6)
        assert np.allclose(rel_total, polar + apolar, atol=1e-6)
        assert np.allclose(burial, 1.0 - rel_total, atol=1e-6)
        assert np.allclose(ratio, polar / (polar + apolar + 1e-8), atol=1e-5)

    def test_the_relative_columns_measure_against_their_own_class(self, example_pdb):
        """This is what the per-class references buy: relativePolar asks how
        exposed the polar surface is, not what fraction of the residue it is."""
        block = [t for t in
                 Protein.from_pdb(example_pdb).featurize(mode="graph")["graph"]["node_features"]
                 if t.shape[-1] == 11][0]
        for absolute, relative in ((0, 5), (1, 6), (2, 7), (3, 8)):
            assert not np.array_equal(block[:, absolute], block[:, relative]), absolute
        # A relative exposure reaches close to 1 for a residue on the surface;
        # a fraction of the whole residue cannot, because the classes share it.
        assert block[:, 5:9].max() > 0.85
        assert block[:, 0:4].max() < 0.75


class TestResidueBurialIndexIsInformative:
    """Guards the scale fix: relativeTotal is a fraction, not a percentage."""

    def test_burial_column_spans_a_real_range(self, example_pdb):
        graph = Protein.from_pdb(example_pdb).featurize(mode="graph")["graph"]
        burial = [t for t in graph["node_features"] if t.shape[-1] == 11][0][:, 5]
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
        block = [t for t in graph["node_features"] if t.shape[-1] == 11][0]
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


class TestHowTheAnswerDependsOnOrientation:
    """Point-sampled SASA reads a lattice fixed in space, not one carried with
    the molecule, so the answer moves when the structure is rotated. These pin
    what is guaranteed and characterise what is not, because neither was
    written down anywhere before 0.4.x."""

    @staticmethod
    def _atoms(pdb_path):
        from plmol.parsers import PDBParser
        from plmol.sasa import element_radius

        atoms = PDBParser(pdb_path).protein_atoms
        coords = np.array([a.coords for a in atoms], dtype=np.float32)
        radii = np.array([element_radius(a.element) for a in atoms], dtype=np.float32)
        return coords, radii

    @staticmethod
    def _rotation(seed):
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        return q.astype(np.float32)

    def test_translation_changes_nothing(self, example_pdb):
        """This one is exact and worth relying on."""
        coords, radii = self._atoms(example_pdb)
        here = shrake_rupley(coords, radii)
        there = shrake_rupley(coords + np.float32([37.0, -12.0, 5.0]), radii)
        assert np.array_equal(here, there)

    def test_rotation_changes_the_answer(self, example_pdb):
        """Not a guarantee -- a warning. If this ever starts passing as
        invariant, the lattice has been made to follow the molecule and the
        docstring in plmol/sasa.py needs its numbers redone."""
        coords, radii = self._atoms(example_pdb)
        areas = np.stack([shrake_rupley(coords @ self._rotation(s).T, radii)
                          for s in range(3)])
        spread = (areas.max(0) - areas.min(0)) / np.maximum(areas.mean(0), 1e-6)
        assert spread.mean() > 0.1, "the documented sensitivity is gone; update the docs"
        flipped = ((areas.min(0) == 0) & (areas.max(0) > 0)).sum()
        assert flipped > 0, "atoms used to move between zero and non-zero"

    def test_more_points_help_and_do_not_cure(self, example_pdb):
        coords, radii = self._atoms(example_pdb)

        def spread(n):
            areas = np.stack([shrake_rupley(coords @ self._rotation(s).T, radii, n_points=n)
                              for s in range(3)])
            return float(((areas.max(0) - areas.min(0)) / np.maximum(areas.mean(0), 1e-6)).mean())

        coarse, fine = spread(100), spread(1000)
        assert fine < coarse, "ten times the samples should be quieter"
        assert fine > 0.05, "and still not silent"

    def test_the_polar_ratio_swings_end_to_end_on_a_buried_residue(self, example_pdb):
        """polar / (polar + apolar + 1e-8) on a residue with no measurable
        surface is 0.0 when nothing survives the occlusion test and 1.0 when
        one polar sample point does. Which of those you get is decided by the
        structure's orientation."""
        from plmol import Protein, as_graph

        import tempfile, os

        def rotated(seed):
            q = self._rotation(seed)
            out = os.path.join(tempfile.mkdtemp(), "rot.pdb")
            with open(out, "w") as handle:
                for line in open(example_pdb):
                    if line[:6].strip() in ("ATOM", "HETATM"):
                        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        x, y, z = q @ xyz
                        line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                    handle.write(line)
            return out

        ratios = np.stack([
            np.asarray(as_graph(Protein.from_pdb(p).featurize(mode="graph")["graph"])
                       ["node_features"])[:, 68]
            for p in (example_pdb, rotated(0), rotated(1))
        ])
        swing = ratios.max(0) - ratios.min(0)
        assert swing.max() > 0.9, "a residue swinging the full range is the point"
