"""Shared voxel featurization helpers for ligand/protein pipelines."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..constants import (
    VOXEL_DEFAULT_RESOLUTION,
    VOXEL_DEFAULT_BOX_SIZE,
    VOXEL_DEFAULT_PADDING,
    VOXEL_DEFAULT_SIGMA_SCALE,
    VOXEL_DEFAULT_CUTOFF_SIGMA,
)


def build_ligand_voxel(
    mol: Any,
    center: Optional[np.ndarray] = None,
    resolution: float = VOXEL_DEFAULT_RESOLUTION,
    box_size: Optional[int] = VOXEL_DEFAULT_BOX_SIZE,
    padding: float = VOXEL_DEFAULT_PADDING,
    sigma_scale: float = VOXEL_DEFAULT_SIGMA_SCALE,
    cutoff_sigma: float = VOXEL_DEFAULT_CUTOFF_SIGMA,
    charge_method: str = "gasteiger",
) -> Optional[Dict[str, np.ndarray]]:
    """Create ligand voxel representation.

    Args:
        mol: RDKit molecule with 3D conformer.
        center: Grid center (3,). None = ligand centroid.
        resolution: Angstrom per voxel.
        box_size: Fixed grid dimension. None for adaptive.
        padding: Padding when box_size is None.
        sigma_scale: VdW radius multiplier for Gaussian sigma.
        cutoff_sigma: Gaussian cutoff in sigma units.
        charge_method: "gasteiger" or "mmff94".

    Returns:
        Dict with "voxel", "channel_names", "grid_origin", etc.
    """
    from ..voxel.voxel_features import voxelize_ligand

    return voxelize_ligand(
        mol=mol,
        center=center,
        resolution=resolution,
        box_size=box_size,
        padding=padding,
        sigma_scale=sigma_scale,
        cutoff_sigma=cutoff_sigma,
        charge_method=charge_method,
    )


def build_protein_voxel(
    coords: np.ndarray,
    radii: np.ndarray,
    atom_metadata: list[dict],
    center: Optional[np.ndarray] = None,
    resolution: float = VOXEL_DEFAULT_RESOLUTION,
    box_size: Optional[int] = VOXEL_DEFAULT_BOX_SIZE,
    padding: float = VOXEL_DEFAULT_PADDING,
    sigma_scale: float = VOXEL_DEFAULT_SIGMA_SCALE,
    cutoff_sigma: float = VOXEL_DEFAULT_CUTOFF_SIGMA,
) -> Optional[Dict[str, np.ndarray]]:
    """Create protein voxel representation.

    Args:
        coords: Atom positions (N, 3).
        radii: VdW radii (N,).
        atom_metadata: Per-atom dicts with 'res_name', 'atom_name', etc.
        center: Grid center (3,). None = protein centroid.
        resolution: Angstrom per voxel.
        box_size: Fixed grid dimension. None for adaptive.
        padding: Padding when box_size is None.
        sigma_scale: VdW radius multiplier for Gaussian sigma.
        cutoff_sigma: Gaussian cutoff in sigma units.

    Returns:
        Dict with "voxel", "channel_names", "grid_origin", etc.
    """
    from ..voxel.voxel_features import voxelize_protein

    return voxelize_protein(
        positions=coords,
        radii=radii,
        atom_metadata=atom_metadata,
        center=center,
        resolution=resolution,
        box_size=box_size,
        padding=padding,
        sigma_scale=sigma_scale,
        cutoff_sigma=cutoff_sigma,
    )


__all__ = ["build_ligand_voxel", "build_protein_voxel"]
