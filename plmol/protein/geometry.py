"""Stateless geometric computation functions for protein structures.

Pure functions with no class dependencies. Can be imported and reused
by any featurizer module (residue, backbone, interaction, etc.).
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def calculate_dihedral(coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate dihedral angles from coordinates.

    Args:
        coords: Coordinate tensor of shape (N, M, 3) where N is the number
            of residues and M is the number of atoms per residue.
        eps: Small value for numerical stability.

    Returns:
        Dihedral angles tensor of shape (N, M).
    """
    shape = coords.shape
    coords_flat = coords.reshape(shape[0] * shape[1], shape[2])

    U = F.normalize(coords_flat[1:, :] - coords_flat[:-1, :], dim=-1)
    u_2 = U[:-2, :]
    u_1 = U[1:-1, :]
    u_0 = U[2:, :]

    n_2 = F.normalize(torch.cross(u_2, u_1, dim=1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=1), dim=-1)

    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
    D = F.pad(D, (1, 2), 'constant', 0)

    return D.view((int(D.size(0) / shape[1]), shape[1]))


def calculate_local_frames(coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate local N-CA-C coordinate frames for each residue.

    Args:
        coords: Residue coordinates of shape (L, MAX_ATOMS, 3).
            Index 0=N, 1=CA, 2=C.
        eps: Small value for numerical stability.

    Returns:
        Local frames tensor of shape (L, 3, 3).
    """
    p_N, p_Ca, p_C = coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]

    u = p_N - p_Ca
    v = p_C - p_Ca

    x_axis = F.normalize(u, dim=-1, eps=eps)
    z_axis = F.normalize(torch.cross(u, v, dim=-1), dim=-1, eps=eps)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)

    return torch.stack([x_axis, y_axis, z_axis], dim=2)


def calculate_backbone_curvature(
    coords: torch.Tensor,
    terminal_flags: Tuple[torch.Tensor, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calculate backbone curvature from CA coordinates.

    Args:
        coords: Residue coordinates of shape (L, MAX_ATOMS, 3). Index 1=CA.
        terminal_flags: Tuple of (n_terminal, c_terminal) boolean tensors.
        eps: Small value for numerical stability.

    Returns:
        Backbone curvature tensor of shape (L,).
    """
    ca_coords = coords[:, 1, :]

    p_im1 = ca_coords[:-2]
    p_i = ca_coords[1:-1]
    p_ip1 = ca_coords[2:]

    v1 = p_im1 - p_i
    v2 = p_ip1 - p_i

    cos_theta = (F.normalize(v1, dim=-1, eps=eps) * F.normalize(v2, dim=-1, eps=eps)).sum(dim=-1)
    curvature_rad = torch.acos(torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps))

    curvature_rad = F.pad(curvature_rad, (1, 1), 'constant', 0)
    n_terminal, c_terminal = terminal_flags
    curvature_rad = curvature_rad * ~n_terminal
    curvature_rad = curvature_rad * ~c_terminal

    return curvature_rad


def calculate_backbone_torsion(
    coords: torch.Tensor,
    terminal_flags: Tuple[torch.Tensor, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calculate backbone torsion from CA coordinates.

    Args:
        coords: Residue coordinates of shape (L, MAX_ATOMS, 3). Index 1=CA.
        terminal_flags: Tuple of (n_terminal, c_terminal) boolean tensors.
        eps: Small value for numerical stability.

    Returns:
        Backbone torsion tensor of shape (L,).
    """
    ca_coords = coords[:, 1, :]

    p0 = ca_coords[:-3]
    p1 = ca_coords[1:-2]
    p2 = ca_coords[2:-1]
    p3 = ca_coords[3:]

    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = F.normalize(torch.cross(b1, b2, dim=-1), dim=-1, eps=eps)
    n2 = F.normalize(torch.cross(b2, b3, dim=-1), dim=-1, eps=eps)

    x = (n1 * n2).sum(dim=-1)
    y = (torch.cross(n1, n2, dim=-1) * F.normalize(b2, dim=-1, eps=eps)).sum(dim=-1)
    torsion_rad = torch.atan2(y, x)

    torsion_rad = F.pad(torsion_rad, (1, 2), 'constant', 0)
    n_terminal, c_terminal = terminal_flags
    torsion_rad = torsion_rad * ~n_terminal
    torsion_rad = torsion_rad * ~c_terminal

    return torsion_rad


def calculate_virtual_cb(coords: torch.Tensor) -> torch.Tensor:
    """Virtual CB from N-CA-C geometry (ProteinMPNN coefficients).

    Works uniformly for all residues including GLY.

    Args:
        coords: Tensor of shape (L, MAX_ATOMS, 3). Index 0=N, 1=CA, 2=C.

    Returns:
        Virtual CB positions of shape (L, 3).
    """
    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def calculate_self_distances_vectors(
    coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate intra-residue distances and vectors.

    Uses atoms N(0), CA(1), C(2), O(3) and sidechain centroid(-1).

    Args:
        coords: Residue coordinates of shape (L, MAX_ATOMS, 3).

    Returns:
        Tuple of (distances, vectors):
            - distances: (L, 10) upper-triangle pairwise distances
            - vectors: (L, 20, 3) selected pairwise vectors
    """
    coords_subset = torch.cat([coords[:, :4, :], coords[:, -1:, :]], dim=1)

    distance = torch.cdist(coords_subset, coords_subset)
    mask_sca = torch.triu(torch.ones_like(distance), diagonal=1).bool()
    distance = torch.masked_select(distance, mask_sca).view(distance.shape[0], -1)

    vectors = coords_subset[:, None] - coords_subset[:, :, None]
    vectors = vectors.view(coords.shape[0], 25, 3)
    vectors = torch.index_select(
        vectors, 1,
        torch.tensor([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]),
    )

    return torch.nan_to_num(distance), torch.nan_to_num(vectors)


def rbf_encode(
    distances: torch.Tensor,
    d_min: float = 0.0,
    d_max: float = 20.0,
    num_rbf: int = 16,
) -> torch.Tensor:
    """Gaussian Radial Basis Function encoding of distances.

    Standard encoding used by ProteinMPNN, GVP, PiFold, etc.

    Args:
        distances: Arbitrary-shape distance tensor.
        d_min: Minimum center value.
        d_max: Maximum center value.
        num_rbf: Number of Gaussian basis functions.

    Returns:
        Encoded tensor with shape (*distances.shape, num_rbf).
    """
    mu = torch.linspace(d_min, d_max, num_rbf, device=distances.device)
    sigma = (d_max - d_min) / num_rbf
    return torch.exp(-((distances.unsqueeze(-1) - mu) ** 2) / (2 * sigma ** 2))
