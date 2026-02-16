"""
Surface-based Molecular Featurizer

MaSIF-style vertex features for protein-ligand interactions.
"""

from .surface_features import (
    create_surface_mesh,
    create_surface_points,
    build_surface_dict,
    compute_all_vertex_features,
    compute_mesh_geometry,
    compute_pointcloud_geometry,
    compute_chemical_features,
    compute_ligand_type_features,
    compute_protein_type_features,
    compute_extra_features,
    compute_ligand_surface_features,
    compute_protein_surface_features,
)

__version__ = "0.1.0"
