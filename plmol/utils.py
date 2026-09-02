"""Shared utility functions for plmol."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections import OrderedDict
from typing import Optional
from io import StringIO

import numpy as np

logger = logging.getLogger(__name__)

try:
    import freesasa as _freesasa
    _freesasa.setVerbosity(_freesasa.nowarnings)
except ImportError:
    _freesasa = None


@contextlib.contextmanager
def suppress_freesasa_warnings():
    """Suppress noisy FreeSASA atom-typing warnings emitted to stdio."""
    with warnings_suppressed():
        verbosity_api = None
        old_verbosity = None
        try:
            import freesasa as verbosity_api

            old_verbosity = verbosity_api.getVerbosity()
            verbosity_api.setVerbosity(verbosity_api.nowarnings)
        except Exception:
            verbosity_api = None

        old_stdout_fd = os.dup(1)
        old_fd = os.dup(2)
        old_stderr = sys.stderr
        devnull = open(os.devnull, "w")
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            sys.stderr = StringIO()
            yield
        finally:
            sys.stderr = old_stderr
            os.dup2(old_fd, 2)
            os.close(old_fd)
            os.dup2(old_stdout_fd, 1)
            os.close(old_stdout_fd)
            devnull.close()
            if verbosity_api is not None and old_verbosity is not None:
                verbosity_api.setVerbosity(old_verbosity)


@contextlib.contextmanager
def warnings_suppressed():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


_SASA_RESULT_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_SASA_RESULT_CACHE_MAX = 4


def sasa_structure_result(pdb_file: str):
    """Return a ``(structure, result)`` SASA pair from the configured backend.

    freesasa when it is installed and not overridden, otherwise plmol's own
    Shrake-Rupley. Both objects expose the same accessors the featurizers use.
    See :mod:`plmol.sasa` for how the two differ.
    """
    from .sasa import native_structure_result, resolve_sasa_backend

    if resolve_sasa_backend() == "native":
        return native_structure_result(pdb_file)
    return freesasa_structure_result(pdb_file)


def freesasa_structure_result(pdb_file: str):
    """Return a cached ``(structure, result)`` FreeSASA pair for a PDB path.

    ``freesasa.calc`` dominates protein featurization (~65 ms on a 3k-atom
    structure), and residue-level and atom-level featurizers run it on the same
    standardized file. The cache is keyed on identity plus mtime and size so a
    rewritten or recycled temporary path never returns a stale result.

    Raises:
        DependencyError: If freesasa is not installed.
    """
    from .errors import DependencyError

    if _freesasa is None:
        raise DependencyError("freesasa is required for SASA computation.")

    stat = os.stat(pdb_file)
    key = (os.path.abspath(pdb_file), stat.st_ino, stat.st_mtime_ns, stat.st_size)
    cached = _SASA_RESULT_CACHE.get(key)
    if cached is not None:
        _SASA_RESULT_CACHE.move_to_end(key)
        return cached

    with suppress_freesasa_warnings():
        structure = _freesasa.Structure(pdb_file)
        result = _freesasa.calc(structure)

    _SASA_RESULT_CACHE[key] = (structure, result)
    if len(_SASA_RESULT_CACHE) > _SASA_RESULT_CACHE_MAX:
        _SASA_RESULT_CACHE.popitem(last=False)
    return structure, result


def _burial_index_native(
    atom_positions: np.ndarray,
    res_names: list,
    atom_names: list,
    n_atoms: int,
) -> np.ndarray:
    """Burial index from plmol's own Shrake-Rupley, no freesasa involved.

    The element comes from the leading letters of the PDB atom name, the same
    rule the parser uses.
    """
    from .constants import RESIDUE_MAX_SASA
    from .sasa import element_radius, shrake_rupley

    radii = np.array(
        [element_radius(_element_from_atom_name(atom_names[i])) for i in range(n_atoms)],
        dtype=np.float32,
    )
    areas = shrake_rupley(np.asarray(atom_positions, dtype=np.float32), radii)
    max_sasa = np.array(
        [RESIDUE_MAX_SASA.get(res_names[i], 200.0) or 200.0 for i in range(n_atoms)],
        dtype=np.float64,
    )
    return np.clip(1.0 - areas / max_sasa, 0.0, 1.0).astype(np.float32)


def _element_from_atom_name(atom_name: str) -> str:
    """First alphabetic character of a PDB atom name, its element symbol."""
    for character in (atom_name or "").strip():
        if character.isalpha():
            return character.upper()
    return ""


def _burial_index_from_file(
    pdb_file: str,
    res_names: list,
    atom_names: list,
    n_atoms: int,
) -> "Optional[np.ndarray]":
    """Burial index from the cached file-based SASA, or None if it does not fit.

    The caller's atom list is only interchangeable with the file's when both
    describe the same atoms in the same order, so that is checked rather than
    assumed.
    """
    from .constants import RESIDUE_MAX_SASA

    try:
        structure, result = freesasa_structure_result(pdb_file)
        if result.nAtoms() != n_atoms:
            return None
        for i in range(n_atoms):
            if (structure.atomName(i).strip() != atom_names[i]
                    or structure.residueName(i).strip() != res_names[i]):
                return None

        burial = np.empty(n_atoms, dtype=np.float32)
        for i in range(n_atoms):
            max_sasa = RESIDUE_MAX_SASA.get(res_names[i], 200.0)
            relative = result.atomArea(i) / max_sasa if max_sasa > 0 else 0.5
            burial[i] = np.clip(1.0 - relative, 0.0, 1.0)
        return burial
    except Exception:
        return None


def compute_burial_index(
    atom_positions: np.ndarray,
    res_names: list,
    atom_names: list,
    n_atoms: int,
    pdb_file: Optional[str] = None,
) -> np.ndarray:
    """Compute per-atom burial index from SASA.

    Uses freesasa to compute per-atom solvent-accessible surface area,
    then normalizes by RESIDUE_MAX_SASA to get relative SASA.
    burial_index = 1.0 - relative_sasa, clamped to [0, 1].

    Falls back to 0.5 if freesasa is unavailable or computation fails.

    Args:
        atom_positions: Atom coordinates (N, 3). May be None.
        res_names: Residue name per atom.
        atom_names: Atom name per atom.
        n_atoms: Number of atoms.
        pdb_file: Optional path the same atoms came from. When the cached
            file-based FreeSASA result describes exactly these atoms, it is
            reused instead of running a second ~65 ms calculation; any
            mismatch falls back to building the structure atom by atom.

    Returns:
        Per-atom burial index array (N,) in [0, 1].
    """
    from .constants import RESIDUE_MAX_SASA

    if atom_positions is None or n_atoms == 0:
        return np.full(n_atoms, 0.5, dtype=np.float32)

    from .sasa import resolve_sasa_backend

    if resolve_sasa_backend() == "native":
        return _burial_index_native(atom_positions, res_names, atom_names, n_atoms)

    if pdb_file is not None:
        reused = _burial_index_from_file(pdb_file, res_names, atom_names, n_atoms)
        if reused is not None:
            return reused

    try:
        structure = _freesasa.Structure()
        for i in range(n_atoms):
            rn = res_names[i] if res_names[i] else "UNK"
            an = atom_names[i] if atom_names[i] else "X"
            x, y, z = float(atom_positions[i, 0]), float(atom_positions[i, 1]), float(atom_positions[i, 2])
            structure.addAtom(an, rn, "1", "A", x, y, z)

        with suppress_freesasa_warnings():
            result = _freesasa.calc(structure)

        burial = np.full(n_atoms, 0.5, dtype=np.float32)
        for i in range(n_atoms):
            atom_sasa = result.atomArea(i)
            max_sasa = RESIDUE_MAX_SASA.get(res_names[i], 200.0)
            relative_sasa = atom_sasa / max_sasa if max_sasa > 0 else 0.5
            burial[i] = np.clip(1.0 - relative_sasa, 0.0, 1.0)
        return burial
    except Exception:
        logger.warning("freesasa computation failed, using default burial_index=0.5")
        return np.full(n_atoms, 0.5, dtype=np.float32)


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


def dense_to_edges(adjacency: "torch.Tensor"):
    """Split a dense ``(N, N, C)`` adjacency into edge indices and edge values.

    torch is imported inside, not at module scope: the SASA and burial helpers
    next to it are pure numpy and are what the surface and voxel paths import,
    so a module-level import would drag torch into the geometry stack.

    An edge exists where the ``C``-vector for that pair is not all zero, which
    is the same rule ``Tensor.to_sparse(sparse_dim=2)`` applies -- but that call
    is pathologically slow on hybrid tensors (25 ms against 0.2 ms on a
    416-residue graph) while producing identical indices and values.

    Args:
        adjacency: Dense ``(N, N, C)`` or ``(N, N)`` tensor.

    Returns:
        ``(src, dst, values)`` in row-major order.
    """
    import torch

    mask = (adjacency != 0).any(dim=-1) if adjacency.dim() > 2 else adjacency != 0
    src, dst = torch.nonzero(mask, as_tuple=True)
    return src, dst, adjacency[src, dst]


def knn_mask_torch(dist_matrix: "torch.Tensor", k: int) -> "torch.Tensor":
    """Square distance matrix -> kNN boolean mask."""
    import torch

    dm = dist_matrix.clone()
    dm.fill_diagonal_(float('inf'))
    k = min(k, dm.size(0) - 1)
    _, topk_idx = torch.topk(dm, k, dim=1, largest=False)
    mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
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
