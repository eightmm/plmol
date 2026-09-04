"""Shared utility functions for plmol."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def sasa_structure_result(pdb_file: str):
    """A ``(structure, result)`` SASA pair for a PDB path.

    See :mod:`plmol.sasa`. The areas are cached on the atom coordinates, so
    several featurizers over one structure compute them once.
    """
    from .sasa import native_structure_result

    return native_structure_result(pdb_file)


def atom_sasa_features(
    atom_positions: np.ndarray,
    res_names: list,
    atom_names: list,
    max_sasa: Optional[dict] = None,
    default_max_sasa: float = 200.0,
) -> dict:
    """Per-atom SASA, its relative form, burial index and polarity.

    The element comes from the leading letters of the PDB atom name, the same
    rule the parser uses. *max_sasa* is the per-residue reference the relative
    form divides by -- ``RESIDUE_MAX_SASA`` for a protein, ``NUCLEOTIDE_MAX_SASA``
    for a nucleic acid -- so both molecule types get the same four quantities
    from one computation rather than two.

    Returns:
        ``sasa`` (A^2), ``relative_sasa`` and ``burial_index`` in [0, 1], and
        ``is_polar_sasa``, 1.0 where the element is N, O or S.
    """
    from .constants import RESIDUE_MAX_SASA
    from .sasa import element_radius, is_polar_element, shrake_rupley

    if max_sasa is None:
        max_sasa = RESIDUE_MAX_SASA
    n_atoms = len(atom_names)
    elements = [_element_from_atom_name(name) for name in atom_names]
    radii = np.array([element_radius(e) for e in elements], dtype=np.float32)
    areas = shrake_rupley(np.asarray(atom_positions, dtype=np.float32), radii)
    reference = np.array(
        [max_sasa.get(res_names[i], default_max_sasa) or default_max_sasa
         for i in range(n_atoms)],
        dtype=np.float64,
    )
    relative = np.clip(areas / reference, 0.0, 1.0)
    return {
        "sasa": areas.astype(np.float32),
        "relative_sasa": relative.astype(np.float32),
        "burial_index": (1.0 - relative).astype(np.float32),
        "is_polar_sasa": np.array([float(is_polar_element(e)) for e in elements],
                                  dtype=np.float32),
    }


def _burial_index_native(
    atom_positions: np.ndarray,
    res_names: list,
    atom_names: list,
    n_atoms: int,
) -> np.ndarray:
    """Burial index from Shrake-Rupley over the coordinates given."""
    return atom_sasa_features(atom_positions, res_names, atom_names)["burial_index"]


def _element_from_atom_name(atom_name: str) -> str:
    """First alphabetic character of a PDB atom name, its element symbol."""
    for character in (atom_name or "").strip():
        if character.isalpha():
            return character.upper()
    return ""


def compute_burial_index(
    atom_positions: np.ndarray,
    res_names: list,
    atom_names: list,
    n_atoms: int,
    pdb_file: Optional[str] = None,
) -> np.ndarray:
    """Per-atom burial index from SASA.

    ``1 - sasa / RESIDUE_MAX_SASA``, clamped to [0, 1]: 1 is fully buried.

    Args:
        atom_positions: Atom coordinates (N, 3). May be None.
        res_names: Residue name per atom.
        atom_names: Atom name per atom.
        n_atoms: Number of atoms.
        pdb_file: Unused, kept so callers written against 0.3.x still work.
            The areas are cached on the coordinates instead, which does not
            need the atoms to have come from a file at all.

    Returns:
        Per-atom burial index array (N,) in [0, 1]. All 0.5 when there are no
        coordinates to compute from.
    """
    if atom_positions is None or n_atoms == 0:
        return np.full(n_atoms, 0.5, dtype=np.float32)
    return _burial_index_native(atom_positions, res_names, atom_names, n_atoms)


def dihedral_angles(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> np.ndarray:
    """Dihedral angles in radians for batches of four points, each ``(M, 3)``.

    Degenerate quadruples, where the central bond has near-zero length, yield
    0.0. Single points may be passed as ``(1, 3)``.

    ``protein.geometry.calculate_dihedral`` is deliberately separate: it walks a
    ``(N, M, 3)`` chain of residue coordinates with padding rather than taking
    four explicit points.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1, axis=-1)
    valid = b1_norm >= 1e-8
    b1_unit = b1 / np.where(valid, b1_norm, 1.0)[:, None]

    v = b0 - np.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - np.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_unit, v) * w, axis=-1)
    return np.where(valid, np.arctan2(y, x), 0.0)


def dense_to_edges(adjacency):
    """Split a dense ``(N, N, C)`` adjacency into edge indices and edge values.

    An edge exists where the ``C``-vector for that pair is not all zero, which
    is the same rule ``Tensor.to_sparse(sparse_dim=2)`` applies -- but that call
    is pathologically slow on hybrid tensors (25 ms against 0.2 ms on a
    416-residue graph) while producing identical indices and values.

    Args:
        adjacency: Dense ``(N, N, C)`` or ``(N, N)`` array.

    Returns:
        ``(src, dst, values)`` in row-major order.
    """
    mask = (adjacency != 0).any(axis=-1) if adjacency.ndim > 2 else adjacency != 0
    src, dst = np.nonzero(mask)
    return src, dst, adjacency[src, dst]


def knn_mask(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    """Square distance matrix -> kNN boolean mask, excluding the diagonal.

    Exact ties at the k-th place are broken arbitrarily, as they were before.
    """
    working = np.array(dist_matrix, copy=True)
    np.fill_diagonal(working, np.inf)
    k = min(k, working.shape[0] - 1)
    mask = np.zeros(dist_matrix.shape, dtype=bool)
    if k > 0:
        nearest = np.argpartition(working, k - 1, axis=1)[:, :k]
        np.put_along_axis(mask, nearest, True, axis=1)
    return mask


def knn_mask_bipartite_numpy(dm: np.ndarray, k: int) -> np.ndarray:
    """Bipartite (M, N) distance matrix -> kNN boolean mask.

    Each row's k nearest + each col's k nearest.
    """
    if k <= 0 or dm.size == 0:
        return np.zeros_like(dm, dtype=bool)

    k_col = min(k, dm.shape[1])
    topk_col = np.argpartition(dm, k_col - 1, axis=1)[:, :k_col]
    mask_row = np.zeros_like(dm, dtype=bool)
    np.put_along_axis(mask_row, topk_col, True, axis=1)

    k_row = min(k, dm.shape[0])
    topk_row = np.argpartition(dm.T, k_row - 1, axis=1)[:, :k_row]
    mask_col_t = np.zeros_like(dm.T, dtype=bool)
    np.put_along_axis(mask_col_t, topk_row, True, axis=1)
    mask_col = mask_col_t.T

    return mask_row | mask_col
