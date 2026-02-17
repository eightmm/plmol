"""Surface feature extraction package for plmol (dMaSIF point cloud only)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .features import (
    create_surface_points,
    build_surface_dict,
    compute_all_vertex_features,
    compute_pointcloud_geometry,
    compute_chemical_features,
    compute_ligand_type_features,
    compute_protein_type_features,
    compute_extra_features,
    compute_ligand_surface_features,
    compute_protein_surface_features,
    _build_simple_protein_mol,
)
from ..constants import (
    SURFACE_DEFAULT_CURVATURE_SCALES,
    SURFACE_DEFAULT_KNN_ATOMS,
    SURFACE_DEFAULT_POINTS_PER_ATOM,
    SURFACE_DEFAULT_PROBE_RADIUS,
)


def build_ligand_surface(
    coords: np.ndarray,
    radii: np.ndarray,
    mol,
    include_features: bool = True,
    curvature_scales: Tuple[float, ...] = SURFACE_DEFAULT_CURVATURE_SCALES,
    knn_atoms: int = SURFACE_DEFAULT_KNN_ATOMS,
    verbose: bool = False,
    charge_method: str = "gasteiger",
    extra_atom_features: Optional[Dict[str, np.ndarray]] = None,
    n_points_per_atom: int = SURFACE_DEFAULT_POINTS_PER_ATOM,
    probe_radius: float = SURFACE_DEFAULT_PROBE_RADIUS,
) -> Optional[Dict[str, np.ndarray]]:
    """Create ligand surface point cloud and optional vertex features.

    Args:
        coords: Atom positions (N, 3).
        radii: VdW radii (N,).
        mol: RDKit molecule.
        include_features: Compute dMaSIF-style features.
        charge_method: "gasteiger" or "mmff94" for partial charge computation.
        extra_atom_features: User-provided per-atom features to map to vertices.
        n_points_per_atom: Points per atom (default: 100).
        probe_radius: Solvent probe radius (default: 1.4).
    """
    verts, normals = create_surface_points(
        coords,
        radii,
        n_points_per_atom=n_points_per_atom,
        probe_radius=probe_radius,
    )
    if len(verts) == 0:
        return None

    surface = build_surface_dict(verts, None, normals)
    if include_features:
        surface.update(
            compute_ligand_surface_features(
                verts=verts,
                atom_positions=coords,
                mol=mol,
                curvature_scales=curvature_scales,
                knn_atoms=knn_atoms,
                verbose=verbose,
                normals=normals,
                extra_atom_features=extra_atom_features,
                charge_method=charge_method,
            )
        )
    return surface


def build_protein_surface(
    coords: np.ndarray,
    radii: np.ndarray,
    atom_metadata: list[dict],
    include_features: bool = True,
    curvature_scales: Tuple[float, ...] = SURFACE_DEFAULT_CURVATURE_SCALES,
    knn_atoms: int = SURFACE_DEFAULT_KNN_ATOMS,
    verbose: bool = False,
    n_points_per_atom: int = SURFACE_DEFAULT_POINTS_PER_ATOM,
    probe_radius: float = SURFACE_DEFAULT_PROBE_RADIUS,
    extra_atom_features: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Create protein surface point cloud and optional vertex features.

    Args:
        coords: Atom positions (N, 3).
        radii: VdW radii (N,).
        atom_metadata: Per-atom dicts with res_name, atom_name, element, b_factor.
        include_features: Compute dMaSIF-style features.
        n_points_per_atom: Points per atom (default: 100).
        probe_radius: Solvent probe radius (default: 1.4).
        extra_atom_features: User-provided per-atom features to map to vertices.
    """
    verts, normals = create_surface_points(
        coords,
        radii,
        n_points_per_atom=n_points_per_atom,
        probe_radius=probe_radius,
    )
    if len(verts) == 0:
        return None

    surface = build_surface_dict(verts, None, normals)
    if include_features:
        surface.update(
            compute_protein_surface_features(
                verts=verts,
                atom_positions=coords,
                mol=None,
                atom_metadata=atom_metadata,
                curvature_scales=curvature_scales,
                knn_atoms=knn_atoms,
                verbose=verbose,
                normals=normals,
                extra_atom_features=extra_atom_features,
            )
        )
    return surface


__all__ = [
    "create_surface_points",
    "build_surface_dict",
    "compute_all_vertex_features",
    "compute_pointcloud_geometry",
    "compute_chemical_features",
    "compute_ligand_type_features",
    "compute_protein_type_features",
    "compute_extra_features",
    "compute_ligand_surface_features",
    "compute_protein_surface_features",
    "build_ligand_surface",
    "build_protein_surface",
]
