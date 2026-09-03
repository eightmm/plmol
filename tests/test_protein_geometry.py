"""Tests for plmol/protein/geometry.py — stateless geometric computations."""

import numpy as np

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
