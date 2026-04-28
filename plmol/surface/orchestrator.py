"""High-level wrappers for surface feature computation."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..constants import AMINO_ACID_LETTERS, ATOM_TYPE_LABELS
from ..errors import InputError, FeatureError
from ._protein_adapter import _build_simple_protein_mol
from .chemical import compute_chemical_features
from .geometry import compute_pointcloud_geometry
from .mapping import (
    CURVATURE_SCALES,
    SURFACE_KNN_ATOMS,
    _build_knn_weights,
    _knn_map_matrix,
    _knn_map_scalar,
    _normalize_to_range,
)
from .type_features import compute_ligand_type_features, compute_protein_type_features

logger = logging.getLogger(__name__)


def compute_extra_features(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    extra_atom_features: dict[str, np.ndarray],
    knn_atoms: int = SURFACE_KNN_ATOMS,
    verbose: bool = False,
    _knn_data: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> dict:
    """Map user-provided per-atom features to surface vertices via KNN.

    Supports both scalar (1D) and vector (2D) per-atom features.
    Scalar features are normalized to [-1, 1]; vector features are
    mapped as-is via weighted interpolation.

    Can be used independently for custom feature mapping.

    Args:
        verts: Surface points (N, 3)
        atom_positions: Atom positions (M, 3)
        extra_atom_features: Dict of name -> per-atom feature array (1D or 2D)
        knn_atoms: Number of nearest atoms per vertex
        verbose: Whether to print progress messages
        _knn_data: Pre-built (knn_idx, knn_weights, knn_dists) to avoid recomputation

    Returns:
        Dict mapping each feature name to its vertex-mapped array.
    """
    if _knn_data is not None:
        knn_idx, knn_weights, _ = _knn_data
    else:
        knn_idx, knn_weights, _ = _build_knn_weights(verts, atom_positions, k=knn_atoms)

    if verbose:
        logger.debug("Mapping %d extra atom features to vertices", len(extra_atom_features))

    result: dict[str, np.ndarray] = {}
    for name, atom_feat in extra_atom_features.items():
        atom_feat = np.asarray(atom_feat, dtype=np.float32)
        if atom_feat.ndim == 1:
            mapped = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, atom_feat))
        elif atom_feat.ndim == 2:
            mapped = _knn_map_matrix(knn_idx, knn_weights, atom_feat)
        else:
            raise InputError(f"extra_atom_features['{name}'] must be 1D or 2D, got {atom_feat.ndim}D")
        result[name] = mapped

    return result


def compute_all_vertex_features(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    mol,
    is_ligand: bool = True,
    curvature_scales: tuple[float, ...] = CURVATURE_SCALES,
    knn_atoms: int = SURFACE_KNN_ATOMS,
    verbose: bool = False,
    normals: Optional[np.ndarray] = None,
    extra_atom_features: Optional[dict[str, np.ndarray]] = None,
    charge_method: str = "gasteiger",
) -> dict:
    """Compute dMaSIF-inspired surface features at each vertex.

    Wrapper that delegates to modular functions:
    - compute_pointcloud_geometry
    - compute_chemical_features
    - compute_ligand_type_features / compute_protein_type_features
    - compute_extra_features

    For fine-grained control, call the individual functions directly.

    Args:
        verts: Point cloud positions (N, 3)
        atom_positions: Atom positions (M, 3)
        mol: RDKit molecule or _SimpleMol for protein
        is_ligand: Whether this is a ligand (True) or protein (False)
        curvature_scales: Radii for multi-scale curvature computation
        knn_atoms: Number of nearest atoms per vertex for feature mapping
        verbose: Whether to print progress messages
        normals: Pre-computed normals (N, 3)
        extra_atom_features: User-provided per-atom features to map to vertices
        charge_method: "gasteiger" or "mmff94" (ligand only)

    Returns:
        Dictionary of feature arrays, all normalized to [-1, 1]
    """
    n_verts = len(verts)

    # === 1. Geometry features ===
    if normals is not None:
        geom = compute_pointcloud_geometry(verts, normals, curvature_scales, verbose)
    else:
        n_scales = len(curvature_scales)
        geom = {
            'mean_curvature': np.zeros((n_verts, n_scales), dtype=np.float32),
            'gaussian_curvature': np.zeros((n_verts, n_scales), dtype=np.float32),
            'vertex_normal': np.zeros((n_verts, 3), dtype=np.float32),
        }

    # === 2. Build shared KNN data ===
    if verbose:
        logger.debug("Building KNN atom-to-vertex mapping (K=%d)", knn_atoms)
    knn_data = _build_knn_weights(verts, atom_positions, k=knn_atoms)

    # === 3. Chemical features ===
    chem = compute_chemical_features(
        verts, atom_positions, mol, is_ligand, charge_method, knn_atoms, verbose,
        _knn_data=knn_data,
    )

    # === 4. Type-specific features ===
    if is_ligand:
        type_feat = compute_ligand_type_features(
            verts, atom_positions, mol, knn_atoms, verbose, _knn_data=knn_data,
        )
        type_feat['residue_type'] = np.zeros((n_verts, 20), dtype=np.float32)
        type_feat['is_backbone'] = np.zeros(n_verts, dtype=np.float32)
        type_feat['burial_index'] = np.zeros(n_verts, dtype=np.float32)
    else:
        type_feat = compute_protein_type_features(
            verts, atom_positions, mol, knn_atoms, verbose, _knn_data=knn_data,
        )
        type_feat['atom_type'] = np.zeros((n_verts, 6), dtype=np.float32)
        type_feat['hybridization'] = np.zeros(n_verts, dtype=np.float32)
        type_feat['in_ring'] = np.zeros(n_verts, dtype=np.float32)
        type_feat['ring_size'] = np.zeros(n_verts, dtype=np.float32)

    # === 5. Extra user features ===
    extra = {}
    if extra_atom_features:
        extra = compute_extra_features(
            verts, atom_positions, extra_atom_features, knn_atoms, verbose,
            _knn_data=knn_data,
        )

    return {
        **geom,
        **chem,
        **type_feat,
        **extra,
    }


def _stack_surface_features(
    feature_dict: dict,
    feature_keys: list[str],
    curvature_scales: tuple[float, ...] = CURVATURE_SCALES,
) -> tuple[np.ndarray, list[str]]:
    """Stack selected surface features into a single matrix and name list.

    Args:
        feature_dict: Output from compute_all_vertex_features.
        feature_keys: Ordered list of feature names to include.
        curvature_scales: Radii used for multi-scale curvature (for naming).

    Returns:
        Tuple of (features, feature_names) where features is (N, D).
    """
    arrays = []
    names: list[str] = []
    for key in feature_keys:
        values = feature_dict[key]
        if values.ndim == 1:
            arrays.append(values[:, None])
            names.append(key)
        elif values.ndim == 2:
            arrays.append(values)
            if key == "vertex_normal" and values.shape[1] == 3:
                names.extend([f"{key}_x", f"{key}_y", f"{key}_z"])
            elif key in ("mean_curvature", "gaussian_curvature"):
                names.extend([f"{key}_{r:.0f}A" for r in curvature_scales])
            elif key == "atom_type":
                names.extend([f"{key}_{l}" for l in ATOM_TYPE_LABELS])
            elif key == "residue_type":
                names.extend([f"{key}_{aa}" for aa in AMINO_ACID_LETTERS])
            else:
                names.extend([f"{key}_{i}" for i in range(values.shape[1])])
        else:
            raise FeatureError(f"Unsupported feature shape for {key}: {values.shape}")

    if not arrays:
        return np.zeros((0, 0), dtype=np.float32), []

    features = np.concatenate(arrays, axis=1).astype(np.float32)
    return features, names


def compute_ligand_surface_features(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    mol,
    curvature_scales: tuple[float, ...] = CURVATURE_SCALES,
    knn_atoms: int = SURFACE_KNN_ATOMS,
    verbose: bool = False,
    normals: Optional[np.ndarray] = None,
    extra_atom_features: Optional[dict[str, np.ndarray]] = None,
    charge_method: str = "gasteiger",
) -> dict:
    """Compute ligand-specific surface features (atomic/chemical scale).

    Features: multi-scale curvature (10D) + normals (3D) + chemical (8D)
    + atom type (6D) + hybridization/ring (3D) = 30D total.
    """
    all_features = compute_all_vertex_features(
        verts=verts,
        atom_positions=atom_positions,
        mol=mol,
        is_ligand=True,
        curvature_scales=curvature_scales,
        knn_atoms=knn_atoms,
        verbose=verbose,
        normals=normals,
        extra_atom_features=extra_atom_features,
        charge_method=charge_method,
    )

    feature_keys = [
        "mean_curvature",
        "gaussian_curvature",
        "vertex_normal",
        "electrostatic",
        "hydrophobicity",
        "hbd",
        "hba",
        "molar_refractivity",
        "aromaticity",
        "pos_ionizable",
        "neg_ionizable",
        "atom_type",
        "hybridization",
        "in_ring",
        "ring_size",
    ]
    if extra_atom_features:
        feature_keys.extend(extra_atom_features.keys())

    features, feature_names = _stack_surface_features(
        all_features, feature_keys, curvature_scales=curvature_scales,
    )

    return {
        "features": features,
        "feature_names": feature_names,
        "feature_level": "ligand",
        "feature_scope": "atomic",
        "feature_dict": all_features,
    }


def compute_protein_surface_features(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    mol=None,
    atom_metadata: Optional[list[dict]] = None,
    curvature_scales: tuple[float, ...] = CURVATURE_SCALES,
    knn_atoms: int = SURFACE_KNN_ATOMS,
    verbose: bool = False,
    normals: Optional[np.ndarray] = None,
    extra_atom_features: Optional[dict[str, np.ndarray]] = None,
) -> dict:
    """Compute protein-specific surface features (residue/patch scale).

    Features: multi-scale curvature (10D) + normals (3D) + chemical (4D)
    + residue type (20D) + backbone/burial (2D) = 39D total.
    """
    if mol is None and atom_metadata is not None:
        mol = _build_simple_protein_mol(atom_metadata)

    all_features = compute_all_vertex_features(
        verts=verts,
        atom_positions=atom_positions,
        mol=mol,
        is_ligand=False,
        curvature_scales=curvature_scales,
        knn_atoms=knn_atoms,
        verbose=verbose,
        normals=normals,
        extra_atom_features=extra_atom_features,
    )

    feature_keys = [
        "mean_curvature",
        "gaussian_curvature",
        "vertex_normal",
        "electrostatic",
        "hydrophobicity",
        "hbd",
        "hba",
        "residue_type",
        "is_backbone",
        "burial_index",
    ]
    if extra_atom_features:
        feature_keys.extend(extra_atom_features.keys())

    features, feature_names = _stack_surface_features(
        all_features, feature_keys, curvature_scales=curvature_scales,
    )

    return {
        "features": features,
        "feature_names": feature_names,
        "feature_level": "protein",
        "feature_scope": "residue_patch",
        "feature_dict": all_features,
    }
