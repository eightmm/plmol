"""Tests for plmol/protein/geometry.py — stateless geometric computations."""

import torch

from plmol.protein.geometry import (
    calculate_dihedral,
    calculate_local_frames,
    calculate_backbone_curvature,
    calculate_backbone_torsion,
    calculate_virtual_cb,
    calculate_self_distances_vectors,
    rbf_encode,
)


def _make_coords(L: int, atoms_per_res: int = 5) -> torch.Tensor:
    """Create synthetic residue coords (L, atoms_per_res, 3) along z-axis."""
    coords = torch.zeros(L, atoms_per_res, 3)
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
            coords[i, j] = torch.tensor([bx, by, z + bz])
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
        assert torch.isfinite(result).all()


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
            assert torch.allclose(identity, torch.eye(3), atol=1e-5)


class TestCalculateBackboneCurvature:
    def test_shape(self):
        L = 8
        coords = _make_coords(L)
        terminal = (torch.zeros(L, dtype=torch.bool), torch.zeros(L, dtype=torch.bool))
        terminal[0][0] = True
        terminal[1][-1] = True
        result = calculate_backbone_curvature(coords, terminal)
        assert result.shape == (L,)

    def test_terminal_zero(self):
        """Terminal residues should have zero curvature."""
        L = 5
        coords = _make_coords(L)
        n_term = torch.zeros(L, dtype=torch.bool)
        c_term = torch.zeros(L, dtype=torch.bool)
        n_term[0] = True
        c_term[-1] = True
        result = calculate_backbone_curvature(coords, (n_term, c_term))
        assert result[0].item() == 0.0
        assert result[-1].item() == 0.0


class TestCalculateBackboneTorsion:
    def test_shape(self):
        L = 8
        coords = _make_coords(L)
        terminal = (torch.zeros(L, dtype=torch.bool), torch.zeros(L, dtype=torch.bool))
        result = calculate_backbone_torsion(coords, terminal)
        assert result.shape == (L,)

    def test_values_finite(self):
        L = 6
        coords = _make_coords(L)
        terminal = (torch.zeros(L, dtype=torch.bool), torch.zeros(L, dtype=torch.bool))
        result = calculate_backbone_torsion(coords, terminal)
        assert torch.isfinite(result).all()


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
        dists = torch.norm(cb - ca, dim=-1)
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
        assert torch.isfinite(distances).all()
        assert torch.isfinite(vectors).all()


class TestRbfEncode:
    def test_shape(self):
        d = torch.tensor([1.0, 5.0, 10.0, 15.0])
        encoded = rbf_encode(d)
        assert encoded.shape == (4, 16)

    def test_custom_params(self):
        d = torch.tensor([0.5, 1.5])
        encoded = rbf_encode(d, d_min=0.0, d_max=5.0, num_rbf=8)
        assert encoded.shape == (2, 8)

    def test_values_positive(self):
        d = torch.tensor([3.0])
        encoded = rbf_encode(d)
        assert (encoded >= 0).all()

    def test_peak_at_center(self):
        """RBF should peak at the center closest to the input distance."""
        d = torch.tensor([0.0])
        encoded = rbf_encode(d, d_min=0.0, d_max=20.0, num_rbf=16)
        # First center (0.0) should have maximum response
        assert encoded[0, 0] > encoded[0, -1]

    def test_2d_input(self):
        d = torch.randn(3, 4).abs()
        encoded = rbf_encode(d, num_rbf=8)
        assert encoded.shape == (3, 4, 8)
