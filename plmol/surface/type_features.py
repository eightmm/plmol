"""Type-specific surface features for ligands and proteins."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rdkit import Chem

from ..constants import (
    AMINO_ACID_LETTERS,
    ATOM_TYPE_MAP,
    BACKBONE_ATOM_SET,
)
from ..utils import DEFAULT_SASA_POINTS, compute_burial_index
from .mapping import (
    SURFACE_KNN_ATOMS,
    _build_knn_weights,
    _knn_map_matrix,
    _knn_map_scalar,
    _normalize_to_range,
)

logger = logging.getLogger(__name__)

def compute_ligand_type_features(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    mol,
    knn_atoms: int = SURFACE_KNN_ATOMS,
    verbose: bool = False,
    _knn_data: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> dict:
    """Compute ligand-specific type features mapped to surface vertices.

    Includes atom type one-hot encoding, hybridization state, ring
    membership, and ring size.

    Can be used independently for ligand surface analysis.

    Args:
        verts: Surface points (N, 3)
        atom_positions: Atom positions (M, 3)
        mol: RDKit molecule
        knn_atoms: Number of nearest atoms per vertex
        verbose: Whether to print progress messages
        _knn_data: Pre-built (knn_idx, knn_weights, knn_dists) to avoid recomputation

    Returns:
        Dict with keys: 'atom_type' (N, 6), 'hybridization' (N,),
        'in_ring' (N,), 'ring_size' (N,)
    """
    if _knn_data is not None:
        knn_idx, knn_weights, _ = _knn_data
    else:
        knn_idx, knn_weights, _ = _build_knn_weights(verts, atom_positions, k=knn_atoms)

    if verbose:
        logger.debug("Computing ligand type features")

    # Atom type one-hot (C, N, O, S, Halogen, Other)
    n_mol_atoms = mol.GetNumAtoms()
    atom_types = np.array(
        [ATOM_TYPE_MAP.get(a.GetAtomicNum(), 5) for a in mol.GetAtoms()],
        dtype=np.intp,
    )
    atom_type_onehot = np.zeros((n_mol_atoms, 6), dtype=np.float32)
    atom_type_onehot[np.arange(n_mol_atoms), atom_types] = 1.0
    vertex_atom_type = _knn_map_matrix(knn_idx, knn_weights, atom_type_onehot)

    # Hybridization
    hyb_map = {
        Chem.HybridizationType.SP: 1,
        Chem.HybridizationType.SP2: 2,
        Chem.HybridizationType.SP3: 3,
    }
    hybridization = np.array(
        [hyb_map.get(a.GetHybridization(), 0) for a in mol.GetAtoms()],
        dtype=np.float32,
    )

    # Ring membership
    ring_atoms = np.array(
        [1.0 if a.IsInRing() else 0.0 for a in mol.GetAtoms()],
        dtype=np.float32,
    )

    # Ring size
    ring_info = mol.GetRingInfo()
    ring_size = np.zeros(n_mol_atoms, dtype=np.float32)
    for ring in ring_info.AtomRings():
        ring_len = len(ring)
        for atom_idx in ring:
            if ring_len > ring_size[atom_idx]:
                ring_size[atom_idx] = ring_len

    vertex_hybridization = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, hybridization))
    vertex_ring = np.clip(_knn_map_scalar(knn_idx, knn_weights, ring_atoms), 0, 1)
    vertex_ring_size = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, ring_size))

    return {
        'atom_type': vertex_atom_type,
        'hybridization': vertex_hybridization,
        'in_ring': vertex_ring,
        'ring_size': vertex_ring_size,
    }


def compute_protein_type_features(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    mol,
    knn_atoms: int = SURFACE_KNN_ATOMS,
    verbose: bool = False,
    _knn_data: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    pdb_file: Optional[str] = None,
    sasa_points: int = DEFAULT_SASA_POINTS,
) -> dict:
    """Compute protein-specific type features mapped to surface vertices.

    Includes residue type one-hot encoding (20 standard amino acids),
    backbone vs sidechain classification, and burial index (solvent exposure).

    Can be used independently for protein surface analysis.

    Args:
        verts: Surface points (N, 3)
        atom_positions: Atom positions (M, 3)
        mol: RDKit molecule or _SimpleMol with PDB residue info
        knn_atoms: Number of nearest atoms per vertex
        verbose: Whether to print progress messages
        _knn_data: Pre-built (knn_idx, knn_weights, knn_dists) to avoid recomputation

    Returns:
        Dict with keys: 'residue_type' (N, 20), 'is_backbone' (N,),
        'burial_index' (N,)
    """
    if _knn_data is not None:
        knn_idx, knn_weights, _ = _knn_data
    else:
        knn_idx, knn_weights, _ = _build_knn_weights(verts, atom_positions, k=knn_atoms)

    if verbose:
        logger.debug("Computing protein type features")

    n_prot_atoms = mol.GetNumAtoms()
    res_names: list[str] = []
    atom_names: list[str] = []
    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res:
            res_names.append(res.GetResidueName().strip())
            atom_names.append(res.GetName().strip())
        else:
            res_names.append("")
            atom_names.append("")

    # Burial index: 1.0 - relative_sasa (clamped to [0, 1])
    burial = compute_burial_index(
        atom_positions, res_names, atom_names, n_prot_atoms, pdb_file=pdb_file,
        n_points=sasa_points,
    )

    # Residue type one-hot (20 amino acids)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACID_LETTERS)}
    res_indices = np.array(
        [aa_to_idx.get(rn, -1) for rn in res_names], dtype=np.intp,
    )
    residue_onehot = np.zeros((n_prot_atoms, 20), dtype=np.float32)
    valid = res_indices >= 0
    residue_onehot[np.where(valid)[0], res_indices[valid]] = 1.0
    vertex_residue_type = _knn_map_matrix(knn_idx, knn_weights, residue_onehot)

    # Backbone vs sidechain
    is_backbone = np.array(
        [1.0 if an in BACKBONE_ATOM_SET else 0.0 for an in atom_names],
        dtype=np.float32,
    )
    vertex_backbone = np.clip(_knn_map_scalar(knn_idx, knn_weights, is_backbone), 0, 1)

    # Burial index mapped to vertices, normalized to [-1, 1]
    vertex_burial = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, burial))

    return {
        'residue_type': vertex_residue_type,
        'is_backbone': vertex_backbone,
        'burial_index': vertex_burial,
    }
