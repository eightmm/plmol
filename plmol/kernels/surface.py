"""Surface kernel compatibility layer.

This module provides a stable import path for surface kernel functions used by
`plmol.featurizers.surface`, while delegating implementations to the canonical
surface feature module.
"""

from ..surface.featurizer.surface_features import (
    build_surface_dict,
    compute_all_vertex_features,
    compute_geodesic_patches,
    compute_ligand_surface_features,
    compute_protein_surface_features,
    compute_mesh_geometry,
    compute_pointcloud_geometry,
    compute_chemical_features,
    compute_ligand_type_features,
    compute_protein_type_features,
    compute_extra_features,
    create_surface_mesh,
    create_surface_points,
)

__all__ = [
    "create_surface_mesh",
    "create_surface_points",
    "build_surface_dict",
    "compute_all_vertex_features",
    "compute_geodesic_patches",
    "compute_mesh_geometry",
    "compute_pointcloud_geometry",
    "compute_chemical_features",
    "compute_ligand_type_features",
    "compute_protein_type_features",
    "compute_extra_features",
    "compute_ligand_surface_features",
    "compute_protein_surface_features",
]
