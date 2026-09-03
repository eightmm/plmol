"""Tests for ligand-free cavity detection.

There is no reference output to regress against here -- the module finds
something that was never computed before -- so the tests are constructive:
shapes whose cavities are known by construction, then the real structure,
where the ligand says where the answer has to be.
"""

import numpy as np
import pytest

from plmol import InputError
from plmol.cavity import Cavity, detect_cavities, element_vdw_radii
from plmol.surface.point_cloud import _fibonacci_sphere

PROBE = 1.4
CARBON = 1.7


def hollow_sphere(radius: float, atoms: int = 400) -> np.ndarray:
    """A sealed shell, so the interior is one cavity of known size."""
    return (_fibonacci_sphere(atoms) * radius).astype(np.float32)


def uniform(count: int) -> np.ndarray:
    return np.full(count, CARBON, dtype=np.float32)


class TestKnownShapes:
    def test_a_hollow_sphere_has_one_cavity_of_the_right_size(self):
        shell = hollow_sphere(10.0)
        cavities = detect_cavities(shell, uniform(len(shell)))
        assert len(cavities) == 1
        cavity = cavities[0]
        # The cavity is the interior minus the shell's own reach.
        expected = 4 / 3 * np.pi * (10.0 - CARBON - PROBE) ** 3
        assert abs(cavity.volume - expected) / expected < 0.2
        assert np.allclose(cavity.center, 0.0, atol=1.0)

    def test_a_bigger_sphere_holds_a_bigger_cavity(self):
        small = detect_cavities(hollow_sphere(8.0), uniform(400))[0]
        large = detect_cavities(hollow_sphere(12.0, 700), uniform(700))[0]
        assert large.volume > small.volume

    def test_the_scan_length_bounds_how_deep_a_cavity_can_be(self):
        """A point only counts as enclosed if a wall is within scan_length, so
        the middle of a wide cavity is invisible to a short scan."""
        shell = hollow_sphere(12.0, 700)
        interior = 4 / 3 * np.pi * (12.0 - CARBON - PROBE) ** 3
        assert detect_cavities(shell, uniform(700), scan_length=6.0) == []
        short = detect_cavities(shell, uniform(700), scan_length=8.0)[0]
        long = detect_cavities(shell, uniform(700), scan_length=12.0)[0]
        assert short.volume < 0.5 * interior
        assert long.volume > 0.95 * interior

    def test_a_solid_block_has_no_cavity(self):
        grid = np.stack(
            np.meshgrid(*[np.arange(-6, 7, 2.0)] * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3).astype(np.float32)
        assert detect_cavities(grid, uniform(len(grid))) == []

    def test_two_separate_shells_are_two_cavities(self):
        left = hollow_sphere(9.0)
        right = left + np.array([40.0, 0, 0], dtype=np.float32)
        both = np.vstack([left, right])
        cavities = detect_cavities(both, uniform(len(both)))
        assert len(cavities) == 2
        centres = np.array([cavity.center for cavity in cavities])
        assert np.abs(np.sort(centres[:, 0]) - np.array([0.0, 40.0])).max() < 1.5

    def test_a_shell_with_a_hole_is_less_buried(self):
        """Cutting the shell open lets rays escape, so buriedness drops."""
        shell = hollow_sphere(10.0)
        sealed = detect_cavities(shell, uniform(len(shell)))[0]
        opened = shell[shell[:, 2] < 7.0]
        loose = detect_cavities(opened, uniform(len(opened)), psp_threshold=3)
        assert loose
        assert loose[0].buriedness < sealed.buriedness


class TestThresholdBehaviour:
    def test_a_stricter_threshold_finds_less(self):
        shell = hollow_sphere(11.0, 500)
        volumes = [
            detect_cavities(shell, uniform(500), psp_threshold=t)[0].volume
            for t in (4, 5, 6)
        ]
        assert volumes[0] >= volumes[1] >= volumes[2]

    def test_min_points_drops_the_small_ones(self):
        shell = hollow_sphere(10.0)
        many = detect_cavities(shell, uniform(len(shell)), min_points=1)
        few = detect_cavities(shell, uniform(len(shell)), min_points=500)
        assert len(many) >= len(few)

    def test_a_finer_grid_still_finds_the_same_cavity(self):
        shell = hollow_sphere(10.0)
        coarse = detect_cavities(shell, uniform(len(shell)), resolution=1.0)[0]
        fine = detect_cavities(shell, uniform(len(shell)), resolution=0.7)[0]
        assert abs(fine.volume - coarse.volume) / coarse.volume < 0.15
        assert np.allclose(fine.center, coarse.center, atol=1.0)


class TestOnARealStructure:
    def test_the_ligand_sits_in_the_top_cavity(self, example_pdb, example_sdf):
        """The strongest statement available without a reference implementation:
        the largest cavity is the one the crystallographic ligand occupies."""
        from rdkit import Chem

        from plmol.parsers import PDBParser

        atoms = PDBParser(example_pdb).protein_atoms
        coords = np.array([atom.coords for atom in atoms], dtype=np.float32)
        radii = element_vdw_radii([atom.element for atom in atoms])
        cavities = detect_cavities(coords, radii)
        assert cavities

        ligand = Chem.MolFromMolFile(example_sdf).GetConformer().GetPositions()
        centre = ligand.mean(axis=0).astype(np.float32)
        nearest = [float(np.linalg.norm(c.points - centre, axis=1).min()) for c in cavities]
        assert int(np.argmin(nearest)) == 0, "the ligand is not in the largest cavity"
        assert nearest[0] < 2.0, "the ligand centre is not inside the cavity"

    def test_the_top_cavity_lines_the_ligand_guided_pocket(self, example_pdb, example_sdf):
        from plmol.interaction import extract_pocket
        from plmol.parsers import PDBParser

        atoms = PDBParser(example_pdb).protein_atoms
        coords = np.array([atom.coords for atom in atoms], dtype=np.float32)
        radii = element_vdw_radii([atom.element for atom in atoms])
        residues = [(a.chain_id, a.res_num, a.res_name) for a in atoms]
        top = detect_cavities(coords, radii, residues=residues)[0]

        pocket = extract_pocket(example_pdb, example_sdf, distance_cutoff=6.0)[0]
        wanted = {(chain, number) for chain, number, _ in pocket.pocket_residues}
        found = {(chain, number) for chain, number, _ in top.lining_residues}
        assert len(wanted & found) / len(wanted) > 0.8

    def test_lining_atoms_really_touch_the_cavity(self, example_pdb):
        from plmol.parsers import PDBParser

        atoms = PDBParser(example_pdb).protein_atoms
        coords = np.array([atom.coords for atom in atoms], dtype=np.float32)
        radii = element_vdw_radii([atom.element for atom in atoms])
        top = detect_cavities(coords, radii)[0]
        gap = np.linalg.norm(
            coords[top.lining_atom_indices][:, None] - top.points[None], axis=-1
        ).min(axis=1)
        assert (gap <= radii[top.lining_atom_indices] + PROBE + 1.0 + 1e-4).all()

    def test_cavities_come_back_largest_first(self, example_pdb):
        from plmol.parsers import PDBParser

        atoms = PDBParser(example_pdb).protein_atoms
        coords = np.array([atom.coords for atom in atoms], dtype=np.float32)
        radii = element_vdw_radii([atom.element for atom in atoms])
        volumes = [cavity.volume for cavity in detect_cavities(coords, radii)]
        assert volumes == sorted(volumes, reverse=True)


class TestContract:
    def test_volume_is_the_point_count_times_the_cell(self):
        shell = hollow_sphere(10.0)
        cavity = detect_cavities(shell, uniform(len(shell)), resolution=0.8)[0]
        assert abs(cavity.volume - cavity.num_points * 0.8 ** 3) < 1e-6

    def test_the_record_is_frozen(self):
        shell = hollow_sphere(9.0)
        cavity = detect_cavities(shell, uniform(len(shell)))[0]
        assert isinstance(cavity, Cavity)
        with pytest.raises(Exception):
            cavity.volume = 1.0

    def test_no_atoms_means_no_cavities(self):
        assert detect_cavities(np.zeros((0, 3), np.float32), np.zeros(0, np.float32)) == []

    def test_bad_shape_is_rejected(self):
        with pytest.raises(InputError, match=r"\(N, 3\)"):
            detect_cavities(np.zeros((4, 2)), np.zeros(4))

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(InputError, match="radii"):
            detect_cavities(np.zeros((4, 3)), np.zeros(3))

    def test_an_impossible_threshold_is_rejected(self):
        with pytest.raises(InputError, match="psp_threshold"):
            detect_cavities(np.zeros((4, 3)), np.zeros(4), psp_threshold=9)

    def test_a_nonpositive_resolution_is_rejected(self):
        with pytest.raises(InputError, match="resolution"):
            detect_cavities(np.zeros((4, 3)), np.zeros(4), resolution=0.0)

    def test_residue_labels_must_match_the_atoms(self):
        with pytest.raises(InputError, match="residues"):
            detect_cavities(
                np.zeros((4, 3)), np.zeros(4), residues=[("A", 1, "ALA")]
            )

    def test_element_radii_fall_back_for_the_unknown(self):
        radii = element_vdw_radii(["C", "N", "O", None, "Xx"])
        assert radii[0] != radii[1] and (radii > 0).all()


class TestFeaturizeMode:
    def test_the_mode_returns_one_row_per_cavity(self, example_pdb):
        from plmol import Protein

        result = Protein.from_pdb(example_pdb).featurize(mode="cavity")["cavity"]
        count = result["num_cavities"]
        assert count > 0
        assert result["center"].shape == (count, 3)
        assert result["extent"].shape == (count, 3)
        assert result["volume"].shape == (count,)
        assert result["buriedness"].shape == (count,)
        assert len(result["points"]) == count

    def test_the_arrays_are_float32(self, example_pdb):
        from plmol import Protein

        result = Protein.from_pdb(example_pdb).featurize(mode="cavity")["cavity"]
        for key in ("center", "volume", "buriedness", "extent"):
            assert result[key].dtype == np.float32, key

    def test_it_is_not_part_of_all(self, example_pdb):
        """Detection costs more than a graph, so asking for everything should
        not silently pay for it."""
        from plmol import Protein

        assert "cavity" in Protein.from_pdb(example_pdb).featurize(mode="all")

    def test_kwargs_reach_the_detector(self, example_pdb):
        from plmol import Protein

        protein = Protein.from_pdb(example_pdb)
        loose = protein.featurize(mode="cavity", cavity_kwargs={"psp_threshold": 4})["cavity"]
        strict = Protein.from_pdb(example_pdb).featurize(
            mode="cavity", cavity_kwargs={"psp_threshold": 6}
        )["cavity"]
        assert loose["volume"][0] > strict["volume"][0]


class TestFeaturizeCavity:
    def test_it_featurizes_the_lining_residues(self, example_pdb):
        from plmol import Protein

        result = Protein.from_pdb(example_pdb).featurize_cavity(0, mode="graph")
        assert set(result) == {"graph", "cavity"}
        residues = int(result["cavity"]["num_lining_residues"][0])
        assert result["graph"]["node_features"][0].shape[0] == residues

    def test_it_agrees_with_the_ligand_guided_pocket(self, example_pdb, example_sdf):
        """The point of the whole module: an apo structure gets the same site."""
        from plmol import Protein

        protein = Protein.from_pdb(example_pdb)
        from_cavity = protein.featurize_cavity(0, mode="graph")
        from_ligand = protein.featurize_pocket(example_sdf, mode="graph")
        assert from_cavity["graph"]["node_features"][0].shape[0] >= \
            from_ligand["graph"]["node_features"][0].shape[0]

    def test_an_out_of_range_cavity_is_rejected(self, example_pdb):
        from plmol import Protein

        with pytest.raises(InputError, match="only .* were found"):
            Protein.from_pdb(example_pdb).featurize_cavity(999)

    def test_a_protein_without_a_file_is_rejected(self):
        from plmol import Protein

        with pytest.raises(InputError, match="no PDB path"):
            Protein.from_sequence("MKTIIALSYIFCLVFA").featurize_cavity(0)
