"""Stateless geometric computation functions for protein structures.

Pure functions with no class dependencies. Can be imported and reused
by any featurizer module (residue, backbone, interaction, etc.).
"""

from typing import Tuple

import numpy as np

from ..arrays import FLOAT, normalize, pad_last, pairwise_distances
from ..utils import dihedral_angles

#: Pairwise vector slots kept out of the 5x5 intra-residue grid: everything but
#: the diagonal, in row-major order.
_OFF_DIAGONAL = np.array(
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23],
    dtype=np.int64,
)


def calculate_dihedral(
    coords: np.ndarray,
    eps: float = 1e-8,
    breaks: np.ndarray = None,
) -> np.ndarray:
    """Dihedral angles along a chain of residue coordinates.

    The chain is the coordinates read in order, so consecutive quadruples of
    atoms give one angle each; the first and last two slots have no quadruple
    and come back as zero.

    Args:
        coords: Coordinate array of shape (N, M, 3) where N is the number
            of residues and M is the number of atoms per residue.
        eps: Accepted for callers written against 0.4.x. The angle no longer
            comes from an arc cosine, so there is nothing to clamp.
        breaks: Optional ``(N - 1,)`` bool marking residue pairs that are *not*
            bonded -- a chain boundary, or the two sides of a missing loop.
            Three angles read across each such pair and come back as zero: the
            two at the end of row ``i`` and the first of row ``i + 1``. For the
            backbone, with M = 3, that is psi and omega of ``i`` and phi of
            ``i + 1``. Without this the reader treats the rows as one
            continuous chain, which for a deposited structure they are not.

    Returns:
        Dihedral angles array of shape (N, M).
    """
    shape = coords.shape
    chain = coords.reshape(shape[0] * shape[1], shape[2])

    # One dihedral, shared with the atom and nucleic paths. This used to be a
    # second implementation reading sign(u_2 . n_1) * arccos(n_2 . n_1), which
    # agrees in convention but loses the angles near a plane: float32 spacing
    # beside cos = +-1 is 6e-8, so arccos there resolves no finer than about
    # 3.5e-4 rad and collapses anything smaller to zero. Every omega sits in
    # exactly that blind spot. Measured against float64 on 400 random chains,
    # the arc cosine was off by up to 3.0e-5 rad against 2.7e-6 here.
    angles = dihedral_angles(chain[:-3], chain[1:-2], chain[2:-1], chain[3:])
    angles = pad_last(angles, 1, 2)
    angles = angles.reshape((angles.shape[0] // shape[1], shape[1]))

    if breaks is not None:
        broken = np.flatnonzero(np.asarray(breaks, dtype=bool))
        if broken.size:
            n_atoms = shape[1]
            angles[broken, n_atoms - 2] = 0.0
            angles[broken, n_atoms - 1] = 0.0
            angles[broken + 1, 0] = 0.0

    return angles


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

    # The window needs three CA atoms, so a chain shorter than that yields
    # none and the padding alone would be longer than the chain. Trimming to
    # the residue count keeps the promised (L,) shape: a one-residue chain has
    # no curvature and says so with a zero rather than a broadcast error.
    curvature_rad = pad_last(curvature_rad, 1, 1)[:ca_coords.shape[0]]
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

    # Four CA atoms are needed for a torsion, so a chain of three or fewer
    # yields none and the padding alone would be longer than the chain. Same
    # trim as the curvature: the shape is (L,) whatever L is.
    torsion_rad = pad_last(torsion_rad, 1, 2)[:ca_coords.shape[0]]
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
