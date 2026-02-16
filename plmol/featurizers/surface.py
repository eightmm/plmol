"""Shared surface featurization helpers for ligand/protein pipelines."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..constants import (
    SURFACE_DEFAULT_CURVATURE_SCALES,
    SURFACE_DEFAULT_KNN_ATOMS,
    SURFACE_DEFAULT_POINTS_PER_ATOM,
    SURFACE_DEFAULT_PROBE_RADIUS,
)


def build_ligand_surface(
    coords: np.ndarray,
    radii: np.ndarray,
    mol: Any,
    grid_density: float = 2.5,
    threshold: float = 0.5,
    sharpness: float = 1.5,
    include_features: bool = True,
    include_patches: bool = False,
    patch_radius: float = 6.0,
    max_patch_size: int = 128,
    max_patches: Optional[int] = None,
    patch_center_stride: int = 1,
    max_memory_gb: float = 1.0,
    device: Optional[str] = None,
    curvature_scales: Tuple[float, ...] = SURFACE_DEFAULT_CURVATURE_SCALES,
    knn_atoms: int = SURFACE_DEFAULT_KNN_ATOMS,
    verbose: bool = False,
    mode: str = "mesh",
    charge_method: str = "gasteiger",
    extra_atom_features: Optional[Dict[str, np.ndarray]] = None,
    n_points_per_atom: int = SURFACE_DEFAULT_POINTS_PER_ATOM,
    probe_radius: float = SURFACE_DEFAULT_PROBE_RADIUS,
) -> Optional[Dict[str, np.ndarray]]:
    """Create ligand surface and optional ligand-specific vertex features.

    Args:
        mode: "mesh" for marching cubes mesh, "point_cloud" for SAS point cloud.
        charge_method: "gasteiger" or "mmff94" for partial charge computation.
        extra_atom_features: User-provided per-atom features to map to vertices.
        n_points_per_atom: Points per atom for point cloud mode (default: 100).
        probe_radius: Solvent probe radius for point cloud mode (default: 1.4).
    """
    from ..kernels.surface import (
        create_surface_mesh,
        create_surface_points,
        build_surface_dict,
        compute_ligand_surface_features,
        compute_geodesic_patches,
    )

    if mode == "point_cloud":
        verts, normals = create_surface_points(
            coords,
            radii,
            n_points_per_atom=n_points_per_atom,
            probe_radius=probe_radius,
        )
        if len(verts) == 0:
            return None
        faces = None
    else:
        verts, faces, normals = create_surface_mesh(
            coords,
            radii,
            grid_density=grid_density,
            threshold=threshold,
            sharpness=sharpness,
            max_memory_gb=max_memory_gb,
            device=device,
        )
        if verts is None or faces is None or normals is None:
            return None

    surface = build_surface_dict(verts, faces, normals)
    if include_features:
        surface.update(
            compute_ligand_surface_features(
                verts=verts,
                faces=faces,
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
    if include_patches and faces is not None:
        surface.update(
            compute_geodesic_patches(
                verts=verts,
                faces=faces,
                vertex_features=surface.get("features"),
                patch_radius=patch_radius,
                max_patch_size=max_patch_size,
                max_patches=max_patches,
                center_stride=patch_center_stride,
            )
        )
    return surface


def build_protein_surface(
    coords: np.ndarray,
    radii: np.ndarray,
    atom_metadata: list[dict],
    grid_density: float = 2.5,
    threshold: float = 0.5,
    sharpness: float = 1.5,
    include_features: bool = True,
    max_memory_gb: float = 1.0,
    device: Optional[str] = None,
    curvature_scales: Tuple[float, ...] = SURFACE_DEFAULT_CURVATURE_SCALES,
    knn_atoms: int = SURFACE_DEFAULT_KNN_ATOMS,
    verbose: bool = False,
    mode: str = "mesh",
    n_points_per_atom: int = SURFACE_DEFAULT_POINTS_PER_ATOM,
    probe_radius: float = SURFACE_DEFAULT_PROBE_RADIUS,
    extra_atom_features: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Create protein surface and optional protein-specific vertex features.

    Args:
        mode: "mesh" for marching cubes mesh, "point_cloud" for SAS point cloud.
        n_points_per_atom: Points per atom for point cloud mode (default: 100).
        probe_radius: Solvent probe radius for point cloud mode (default: 1.4).
        extra_atom_features: User-provided per-atom features to map to vertices.
    """
    from ..kernels.surface import (
        create_surface_mesh,
        create_surface_points,
        build_surface_dict,
        compute_protein_surface_features,
    )

    if mode == "point_cloud":
        verts, normals = create_surface_points(
            coords,
            radii,
            n_points_per_atom=n_points_per_atom,
            probe_radius=probe_radius,
        )
        if len(verts) == 0:
            return None
        faces = None
    else:
        verts, faces, normals = create_surface_mesh(
            coords,
            radii,
            grid_density=grid_density,
            threshold=threshold,
            sharpness=sharpness,
            max_memory_gb=max_memory_gb,
            device=device,
        )
        if verts is None or faces is None or normals is None:
            return None

    surface = build_surface_dict(verts, faces, normals)
    if include_features:
        surface.update(
            compute_protein_surface_features(
                verts=verts,
                faces=faces,
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


__all__ = ["build_ligand_surface", "build_protein_surface"]
