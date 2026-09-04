"""Tests for plmol/protein/geometry.py — stateless geometric computations."""

import numpy as np
import pytest

from plmol.protein.geometry import (
    calculate_dihedral,
    calculate_local_frames,
    calculate_backbone_curvature,
    calculate_backbone_torsion,
    calculate_virtual_cb,
    calculate_self_distances_vectors,
    rbf_encode,
)


_rng = np.random.default_rng(0)


def _make_coords(L: int, atoms_per_res: int = 5) -> np.ndarray:
    """Create synthetic residue coords (L, atoms_per_res, 3) along z-axis."""
    coords = np.zeros((L, atoms_per_res, 3), dtype=np.float32)
    backbone_atoms = [
        (-0.5, 0.0, -1.0),   # N
        (0.0, 0.0, 0.0),     # CA
        (0.5, 0.0, 0.5),     # C
        (0.5, 1.0, 0.5),     # O
        (0.0, 1.5, 0.0),     # CB/sidechain
    ]
    for i in range(L):
        z = i * 3.8
        for j in range(min(atoms_per_res, len(backbone_atoms))):
            bx, by, bz = backbone_atoms[j]
            coords[i, j] = [bx, by, z + bz]
    return coords


class TestCalculateDihedral:
    def test_shape(self):
        # calculate_dihedral expects (N, M, 3) — N residues, M atoms/res
        coords = _make_coords(6, atoms_per_res=3)
        result = calculate_dihedral(coords)
        assert result.shape == (6, 3)

    def test_values_finite(self):
        coords = _make_coords(10, atoms_per_res=3)
        result = calculate_dihedral(coords)
        assert np.isfinite(result).all()

    def test_it_is_the_same_dihedral_the_rest_of_plmol_uses(self):
        """The chain walk is a layout, not a second definition. It used to be
        a second implementation, reading sign(u2.n1) * arccos(n2.n1)."""
        from plmol.utils import dihedral_angles

        chain = _rng.normal(size=(9, 3, 3)).astype(np.float32) * 3
        flat = chain.reshape(-1, 3)
        expected = dihedral_angles(flat[:-3], flat[1:-2], flat[2:-1], flat[3:])
        got = calculate_dihedral(chain).reshape(-1)[1:-2]
        assert np.array_equal(got, expected)

    def test_an_angle_beside_a_plane_survives(self):
        """float32 spacing next to cos = +-1 is 6e-8, so an arc cosine there
        resolves no finer than about 3.5e-4 rad and snaps anything smaller to
        zero. Every peptide omega sits in that blind spot."""
        for angle in (1e-4, -1e-4, np.pi - 1e-4, -(np.pi - 1e-4)):
            # Central bond along x; the two arms differ by exactly *angle*.
            quad = np.array([[[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                             [[2.0, np.cos(angle), np.sin(angle)],
                              [9.0, 9.0, 9.0], [8.0, 9.0, 9.0]]], dtype=np.float32)
            got = float(calculate_dihedral(quad).reshape(-1)[1])
            assert np.isclose(got, angle, atol=1e-6), (angle, got)

    def test_a_degenerate_quadruple_is_zero_not_noise(self):
        """Padding slots leave repeated points behind; they must not turn into
        an angle of a few ten-thousandths."""
        quad = np.zeros((2, 3, 3), dtype=np.float32)
        quad[0, 0] = [1.0, 0.0, 0.0]
        assert calculate_dihedral(quad).reshape(-1)[1] == 0.0


class TestCalculateLocalFrames:
    def test_shape(self):
        coords = _make_coords(5)
        frames = calculate_local_frames(coords)
        assert frames.shape == (5, 3, 3)

    def test_orthogonality(self):
        """Local frames should be approximately orthonormal."""
        coords = _make_coords(5)
        frames = calculate_local_frames(coords)
        for i in range(5):
            R = frames[i]
            # R^T R should be close to identity
            identity = R.T @ R
            assert np.allclose(identity, np.eye(3), atol=1e-5)


class TestCalculateBackboneCurvature:
    def test_shape(self):
        L = 8
        coords = _make_coords(L)
        terminal = (np.zeros(L, dtype=bool), np.zeros(L, dtype=bool))
        terminal[0][0] = True
        terminal[1][-1] = True
        result = calculate_backbone_curvature(coords, terminal)
        assert result.shape == (L,)

    def test_terminal_zero(self):
        """Terminal residues should have zero curvature."""
        L = 5
        coords = _make_coords(L)
        n_term = np.zeros(L, dtype=bool)
        c_term = np.zeros(L, dtype=bool)
        n_term[0] = True
        c_term[-1] = True
        result = calculate_backbone_curvature(coords, (n_term, c_term))
        assert result[0].item() == 0.0
        assert result[-1].item() == 0.0


class TestCalculateBackboneTorsion:
    def test_shape(self):
        L = 8
        coords = _make_coords(L)
        terminal = (np.zeros(L, dtype=bool), np.zeros(L, dtype=bool))
        result = calculate_backbone_torsion(coords, terminal)
        assert result.shape == (L,)

    def test_values_finite(self):
        L = 6
        coords = _make_coords(L)
        terminal = (np.zeros(L, dtype=bool), np.zeros(L, dtype=bool))
        result = calculate_backbone_torsion(coords, terminal)
        assert np.isfinite(result).all()


class TestCalculateVirtualCb:
    def test_shape(self):
        coords = _make_coords(5)
        cb = calculate_virtual_cb(coords)
        assert cb.shape == (5, 3)

    def test_not_at_ca(self):
        """Virtual CB should be displaced from CA."""
        coords = _make_coords(3)
        cb = calculate_virtual_cb(coords)
        ca = coords[:, 1]
        dists = np.linalg.norm(cb - ca, axis=-1)
        assert (dists > 0.1).all()  # CB is not at CA position


class TestCalculateSelfDistancesVectors:
    def test_shape(self):
        coords = _make_coords(5)
        distances, vectors = calculate_self_distances_vectors(coords)
        assert distances.shape == (5, 10)  # upper triangle of 5x5
        assert vectors.shape == (5, 20, 3)

    def test_no_nan(self):
        coords = _make_coords(5)
        distances, vectors = calculate_self_distances_vectors(coords)
        assert np.isfinite(distances).all()
        assert np.isfinite(vectors).all()


class TestRbfEncode:
    def test_shape(self):
        d = np.array([1.0, 5.0, 10.0, 15.0])
        encoded = rbf_encode(d)
        assert encoded.shape == (4, 16)

    def test_custom_params(self):
        d = np.array([0.5, 1.5])
        encoded = rbf_encode(d, d_min=0.0, d_max=5.0, num_rbf=8)
        assert encoded.shape == (2, 8)

    def test_values_positive(self):
        d = np.array([3.0])
        encoded = rbf_encode(d)
        assert (encoded >= 0).all()

    def test_peak_at_center(self):
        """RBF should peak at the center closest to the input distance."""
        d = np.array([0.0])
        encoded = rbf_encode(d, d_min=0.0, d_max=20.0, num_rbf=16)
        # First center (0.0) should have maximum response
        assert encoded[0, 0] > encoded[0, -1]

    def test_2d_input(self):
        d = np.abs(_rng.standard_normal((3, 4))).astype(np.float32)
        encoded = rbf_encode(d, num_rbf=8)
        assert encoded.shape == (3, 4, 8)


class TestFloatWidthIsPreserved:
    """float32 in, float32 out.

    numpy defaults to 64-bit where torch defaulted to 32-bit, so every array
    these build has to say which it wants. A function that quietly widens costs
    twice the memory downstream and stops matching the rest of the features.
    """

    def test_every_function_returns_float32(self):
        coords = _make_coords(8)
        terminal = (np.zeros(8, dtype=bool), np.zeros(8, dtype=bool))
        distances, vectors = calculate_self_distances_vectors(coords)
        outputs = {
            "dihedral": calculate_dihedral(coords[:, :3]),
            "local_frames": calculate_local_frames(coords),
            "curvature": calculate_backbone_curvature(coords, terminal),
            "torsion": calculate_backbone_torsion(coords, terminal),
            "virtual_cb": calculate_virtual_cb(coords),
            "self_distances": distances,
            "self_vectors": vectors,
            "rbf": rbf_encode(distances),
        }
        widened = {name: str(a.dtype) for name, a in outputs.items() if a.dtype != np.float32}
        assert widened == {}


class TestShortChains:
    """A chain shorter than the window still has to come back with one value
    per residue.

    Curvature reads three CA atoms and torsion four, so a short chain yields
    none of either -- and the padding alone was then longer than the chain.
    calculate_backbone_torsion returned three values for a dipeptide and the
    terminal mask, of length two, could not be broadcast against it. Every
    graph-mode featurization of a structure with fewer than three residues
    died on it, with a bare ValueError from inside numpy.
    """

    @pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 8])
    def test_one_value_per_residue(self, length):
        coords = _rng.normal(size=(length, 15, 3)).astype(np.float32) * 3
        flags = (np.zeros(length, dtype=bool), np.zeros(length, dtype=bool))
        assert calculate_backbone_curvature(coords, flags).shape == (length,)
        assert calculate_backbone_torsion(coords, flags).shape == (length,)

    @pytest.mark.parametrize("length", [1, 2, 3])
    def test_a_short_chain_has_no_curvature_or_torsion(self, length):
        """No window fits, so the honest answer is zero rather than an error."""
        coords = _rng.normal(size=(length, 15, 3)).astype(np.float32) * 3
        flags = (np.zeros(length, dtype=bool), np.zeros(length, dtype=bool))
        assert np.all(calculate_backbone_torsion(coords, flags) == 0.0)
        if length < 3:
            assert np.all(calculate_backbone_curvature(coords, flags) == 0.0)

    def test_a_dipeptide_featurizes(self, tmp_path):
        from plmol import Protein

        def line(serial, name, res, resnum, xyz, element):
            row = list(" " * 80)
            row[0:6] = "ATOM  "
            row[6:11] = f"{serial:5d}"
            row[12:16] = (" " + name).ljust(4)[:4]
            row[17:20] = res.rjust(3)[:3]
            row[21] = "A"
            row[22:26] = f"{resnum:4d}"
            row[30:38] = f"{xyz[0]:8.3f}"
            row[38:46] = f"{xyz[1]:8.3f}"
            row[46:54] = f"{xyz[2]:8.3f}"
            row[54:60] = "  1.00"
            row[60:66] = "  0.00"
            row[76:78] = element.rjust(2)
            return "".join(row).rstrip() + "\n"

        backbone = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
        text = "".join(
            line(i * 10 + j, name, "ALA", i, (i * 3.8 + j * 0.5, 0.0, 0.0), element)
            for i in (1, 2) for j, (name, element) in enumerate(backbone, 1)
        )
        path = tmp_path / "dipeptide.pdb"
        path.write_text(text + "END\n")

        for mode in ("sequence", "graph", "atom_graph", "backbone", "surface", "voxel"):
            result = Protein.from_pdb(str(path)).featurize(mode=mode)[mode]
            assert result is not None, mode
