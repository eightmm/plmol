"""Stateless geometric computation functions for protein structures.

Pure functions with no class dependencies. Can be imported and reused
by any featurizer module (residue, backbone, interaction, etc.).
"""

from typing import Tuple

import numpy as np

from ..arrays import FLOAT, normalize, pad_last, pairwise_distances

#: Pairwise vector slots kept out of the 5x5 intra-residue grid: everything but
#: the diagonal, in row-major order.
_OFF_DIAGONAL = np.array(
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23],
    dtype=np.int64,
)


def calculate_dihedral(coords: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calculate dihedral angles from coordinates.

    Args:
        coords: Coordinate array of shape (N, M, 3) where N is the number
            of residues and M is the number of atoms per residue.
        eps: Small value for numerical stability.

    Returns:
        Dihedral angles array of shape (N, M).
    """
    shape = coords.shape
    coords_flat = coords.reshape(shape[0] * shape[1], shape[2])

    U = normalize(coords_flat[1:, :] - coords_flat[:-1, :], axis=-1)
    u_2 = U[:-2, :]
    u_1 = U[1:-1, :]
    u_0 = U[2:, :]

    n_2 = normalize(np.cross(u_2, u_1, axis=1), axis=-1)
    n_1 = normalize(np.cross(u_1, u_0, axis=1), axis=-1)

    cosD = (n_2 * n_1).sum(-1)
    cosD = np.clip(cosD, -1 + eps, 1 - eps)

    D = np.sign((u_2 * n_1).sum(-1)) * np.arccos(cosD)
    D = pad_last(D, 1, 2)

    return D.reshape((D.shape[0] // shape[1], shape[1]))


def calculate_local_frames(coords: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calculate local N-CA-C coordinate frames for each residue.

    Args:
        coords: Residue coordinates of shape (L, MAX_ATOMS, 3).
            Index 0=N, 1=CA, 2=C.
        eps: Small value for numerical stability.

    Returns:
        Local frames array of shape (L, 3, 3).
    """
    p_N, p_Ca, p_C = coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]

    u = p_N - p_Ca
    v = p_C - p_Ca

    x_axis = normalize(u, axis=-1, eps=eps)
    z_axis = normalize(np.cross(u, v, axis=-1), axis=-1, eps=eps)
    y_axis = np.cross(z_axis, x_axis, axis=-1)

    return np.stack([x_axis, y_axis, z_axis], axis=2)


def calculate_backbone_curvature(
    coords: np.ndarray,
    terminal_flags: Tuple[np.ndarray, np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """Calculate backbone curvature from CA coordinates.

    Args:
        coords: Residue coordinates of shape (L, MAX_ATOMS, 3). Index 1=CA.
        terminal_flags: Tuple of (n_terminal, c_terminal) boolean arrays.
        eps: Small value for numerical stability.

    Returns:
        Backbone curvature array of shape (L,).
    """
    ca_coords = coords[:, 1, :]

    p_im1 = ca_coords[:-2]
    p_i = ca_coords[1:-1]
    p_ip1 = ca_coords[2:]

    v1 = p_im1 - p_i
    v2 = p_ip1 - p_i

    cos_theta = (normalize(v1, axis=-1, eps=eps) * normalize(v2, axis=-1, eps=eps)).sum(axis=-1)
    curvature_rad = np.arccos(np.clip(cos_theta, -1.0 + eps, 1.0 - eps))

    curvature_rad = pad_last(curvature_rad, 1, 1)
    n_terminal, c_terminal = terminal_flags
    curvature_rad = curvature_rad * ~n_terminal
    curvature_rad = curvature_rad * ~c_terminal

    return curvature_rad


def calculate_backbone_torsion(
    coords: np.ndarray,
    terminal_flags: Tuple[np.ndarray, np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """Calculate backbone torsion from CA coordinates.

    Args:
        coords: Residue coordinates of shape (L, MAX_ATOMS, 3). Index 1=CA.
        terminal_flags: Tuple of (n_terminal, c_terminal) boolean arrays.
        eps: Small value for numerical stability.

    Returns:
        Backbone torsion array of shape (L,).
    """
    ca_coords = coords[:, 1, :]

    p0 = ca_coords[:-3]
    p1 = ca_coords[1:-2]
    p2 = ca_coords[2:-1]
    p3 = ca_coords[3:]

    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = normalize(np.cross(b1, b2, axis=-1), axis=-1, eps=eps)
    n2 = normalize(np.cross(b2, b3, axis=-1), axis=-1, eps=eps)

    x = (n1 * n2).sum(axis=-1)
    y = (np.cross(n1, n2, axis=-1) * normalize(b2, axis=-1, eps=eps)).sum(axis=-1)
    torsion_rad = np.arctan2(y, x)

    torsion_rad = pad_last(torsion_rad, 1, 2)
    n_terminal, c_terminal = terminal_flags
    torsion_rad = torsion_rad * ~n_terminal
    torsion_rad = torsion_rad * ~c_terminal

    return torsion_rad


def calculate_virtual_cb(coords: np.ndarray) -> np.ndarray:
    """Virtual CB from N-CA-C geometry (ProteinMPNN coefficients).

    Works uniformly for all residues including GLY.

    Args:
        coords: Array of shape (L, MAX_ATOMS, 3). Index 0=N, 1=CA, 2=C.

    Returns:
        Virtual CB positions of shape (L, 3).
    """
    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]
    b = CA - N
    c = C - CA
    a = np.cross(b, c, axis=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def calculate_self_distances_vectors(
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate intra-residue distances and vectors.

    Uses atoms N(0), CA(1), C(2), O(3) and sidechain centroid(-1).

    Args:
        coords: Residue coordinates of shape (L, MAX_ATOMS, 3).

    Returns:
        Tuple of (distances, vectors):
            - distances: (L, 10) upper-triangle pairwise distances
            - vectors: (L, 20, 3) selected pairwise vectors
    """
    coords_subset = np.concatenate([coords[:, :4, :], coords[:, -1:, :]], axis=1)

    distance = pairwise_distances(coords_subset, coords_subset)
    mask_sca = np.triu(np.ones_like(distance), k=1).astype(bool)
    distance = distance[mask_sca].reshape(distance.shape[0], -1)

    vectors = coords_subset[:, None] - coords_subset[:, :, None]
    vectors = vectors.reshape(coords.shape[0], 25, 3)
    vectors = vectors[:, _OFF_DIAGONAL]

    return np.nan_to_num(distance), np.nan_to_num(vectors)


def rbf_encode(
    distances: np.ndarray,
    d_min: float = 0.0,
    d_max: float = 20.0,
    num_rbf: int = 16,
) -> np.ndarray:
    """Gaussian Radial Basis Function encoding of distances.

    Standard encoding used by ProteinMPNN, GVP, PiFold, etc.

    Args:
        distances: Arbitrary-shape distance array.
        d_min: Minimum center value.
        d_max: Maximum center value.
        num_rbf: Number of Gaussian basis functions.

    Returns:
        Encoded array with shape (*distances.shape, num_rbf).
    """
    mu = np.linspace(d_min, d_max, num_rbf, dtype=FLOAT)
    sigma = (d_max - d_min) / num_rbf
    return np.exp(-((distances[..., None] - mu) ** 2) / (2 * sigma ** 2))
