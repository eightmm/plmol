"""Shared utility functions for plmol."""

import contextlib
import logging
import os
import sys
from io import StringIO

import numpy as np
import torch

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


def compute_burial_index(
    atom_positions: np.ndarray,
    res_names: list,
    atom_names: list,
    n_atoms: int,
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

    Returns:
        Per-atom burial index array (N,) in [0, 1].
    """
    from .constants import RESIDUE_MAX_SASA

    if _freesasa is None or atom_positions is None or n_atoms == 0:
        return np.full(n_atoms, 0.5, dtype=np.float32)

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


def knn_mask_torch(dist_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """Square distance matrix -> kNN boolean mask."""
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
