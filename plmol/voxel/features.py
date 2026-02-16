"""Voxel-based 3D molecular featurization.

Atom features are mapped to a regular 3D grid via Gaussian smearing,
producing multi-channel volumetric representations suitable for 3D CNNs.

Channels:
    Ligand (16ch): occupancy, atom type (6), charge, hydrophobicity,
                   HBD, HBA, aromaticity, pos_ionizable, neg_ionizable,
                   hybridization, ring
    Protein (16ch): occupancy, atom type (6), charge, hydrophobicity,
                    HBD, HBA, aromaticity, pos_ionizable, neg_ionizable,
                    backbone, b_factor

Functions:
    gaussian_smear_to_grid: Core Gaussian smearing onto a 3D grid
    compute_ligand_channels: Extract per-atom feature matrix for ligand
    compute_protein_channels: Extract per-atom feature matrix for protein
    voxelize_ligand: Full ligand voxelization
    voxelize_protein: Full protein voxelization
    voxelize_complex: Combined protein + ligand voxelization
"""

from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Lipinski

from ..constants import (
    VDW_RADIUS,
    DEFAULT_VDW_RADIUS,
    VOXEL_DEFAULT_RESOLUTION,
    VOXEL_DEFAULT_BOX_SIZE,
    VOXEL_DEFAULT_PADDING,
    VOXEL_DEFAULT_SIGMA_SCALE,
    VOXEL_DEFAULT_CUTOFF_SIGMA,
)

# Channel name definitions
LIGAND_CHANNEL_NAMES = [
    "occupancy",
    "atom_C", "atom_N", "atom_O", "atom_S", "atom_Hal", "atom_Other",
    "charge",
    "hydrophobicity",
    "hbd", "hba",
    "aromaticity",
    "pos_ionizable", "neg_ionizable",
    "hybridization",
    "ring",
]

PROTEIN_CHANNEL_NAMES = [
    "occupancy",
    "atom_C", "atom_N", "atom_O", "atom_S", "atom_P", "atom_Other",
    "charge",
    "hydrophobicity",
    "hbd", "hba",
    "aromaticity",
    "pos_ionizable", "neg_ionizable",
    "backbone",
    "b_factor",
]


def _compute_grid_params(
    positions: np.ndarray,
    center: Optional[np.ndarray] = None,
    box_size: Optional[int] = VOXEL_DEFAULT_BOX_SIZE,
    resolution: float = VOXEL_DEFAULT_RESOLUTION,
    padding: float = VOXEL_DEFAULT_PADDING,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Compute grid origin and size.

    Args:
        positions: Atom positions (N, 3).
        center: Grid center (3,). If None, uses centroid of positions.
        box_size: Fixed grid dimension. If None, adapts to molecule size.
        resolution: Angstrom per voxel.
        padding: Padding in Angstrom (used when box_size is None).

    Returns:
        (grid_origin, grid_shape) where grid_origin is (3,) and grid_shape
        is (D, H, W).
    """
    if center is None:
        center = positions.mean(axis=0)

    if box_size is not None:
        half_extent = (box_size * resolution) / 2.0
        origin = center - half_extent
        return origin.astype(np.float32), (box_size, box_size, box_size)

    # Adaptive: fit to bounding box + padding
    min_bound = positions.min(axis=0) - padding
    max_bound = positions.max(axis=0) + padding
    extent = max_bound - min_bound
    grid_shape = tuple(max(4, int(np.ceil(s / resolution))) for s in extent)
    return min_bound.astype(np.float32), grid_shape


def gaussian_smear_to_grid(
    positions: np.ndarray,
    features: np.ndarray,
    sigmas: np.ndarray,
    grid_origin: np.ndarray,
    grid_shape: tuple[int, int, int],
    resolution: float = VOXEL_DEFAULT_RESOLUTION,
    cutoff_sigma: float = VOXEL_DEFAULT_CUTOFF_SIGMA,
) -> np.ndarray:
    """Smear per-atom features onto a 3D grid via Gaussian kernels.

    Each atom contributes to nearby voxels with weight:
        w = exp(-||voxel_pos - atom_pos||^2 / (2 * sigma^2))

    Only voxels within cutoff_sigma * sigma are updated (sparse update).

    Args:
        positions: Atom positions (N, 3).
        features: Per-atom feature matrix (N, C) where C is number of channels.
        sigmas: Per-atom Gaussian width (N,), typically VdW radius based.
        grid_origin: Origin (min corner) of the grid (3,).
        grid_shape: Grid dimensions (D, H, W).
        resolution: Angstrom per voxel.
        cutoff_sigma: Gaussian cutoff in units of sigma.

    Returns:
        Voxel grid (C, D, H, W) as float32.
    """
    n_atoms, n_channels = features.shape
    D, H, W = grid_shape
    grid = np.zeros((n_channels, D, H, W), dtype=np.float32)

    for i in range(n_atoms):
        sigma = sigmas[i]
        cutoff = cutoff_sigma * sigma

        # Atom position in grid index space (fractional)
        frac = (positions[i] - grid_origin) / resolution

        # Bounding box of affected voxels
        lo = np.maximum(0, np.floor(frac - cutoff / resolution).astype(np.intp))
        hi = np.minimum(
            np.array([D, H, W]),
            np.ceil(frac + cutoff / resolution).astype(np.intp) + 1,
        )

        if (lo >= hi).any():
            continue

        # Local grid coordinates (sub-grid only)
        ix = np.arange(lo[0], hi[0])
        iy = np.arange(lo[1], hi[1])
        iz = np.arange(lo[2], hi[2])
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")

        # Actual spatial positions of these voxels
        dx = gx * resolution + grid_origin[0] - positions[i, 0]
        dy = gy * resolution + grid_origin[1] - positions[i, 1]
        dz = gz * resolution + grid_origin[2] - positions[i, 2]
        dist_sq = dx * dx + dy * dy + dz * dz

        gauss = np.exp(-dist_sq / (2.0 * sigma * sigma))

        # Apply cutoff mask
        gauss[dist_sq > cutoff * cutoff] = 0.0

        # Add weighted features to grid channels
        # features[i] shape: (C,), gauss shape: (sx, sy, sz)
        grid[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += (
            features[i, :, None, None, None] * gauss[None, :, :, :]
        )

    return grid


def compute_ligand_channels(
    mol,
    charge_method: str = "gasteiger",
) -> tuple[np.ndarray, list[str]]:
    """Extract per-atom feature matrix for ligand voxelization.

    Args:
        mol: RDKit molecule with 3D conformer.
        charge_method: "gasteiger" or "mmff94".

    Returns:
        (features, channel_names) where features is (N, 16) float32.
    """
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, 16), dtype=np.float32)

    # 0: Occupancy (always 1)
    features[:, 0] = 1.0

    # 1-6: Atom type one-hot (C, N, O, S, Halogen, Other)
    atom_type_map = {6: 0, 7: 1, 8: 2, 16: 3, 9: 4, 17: 4, 35: 4, 53: 4}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        t = atom_type_map.get(atom.GetAtomicNum(), 5)
        features[idx, 1 + t] = 1.0

    # 7: Partial charge
    if charge_method == "mmff94":
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is not None:
            for i in range(n_atoms):
                features[i, 7] = props.GetMMFFPartialCharge(i)
        else:
            AllChem.ComputeGasteigerCharges(mol)
            for atom in mol.GetAtoms():
                c = atom.GetDoubleProp('_GasteigerCharge')
                features[atom.GetIdx(), 7] = c if np.isfinite(c) else 0.0
    else:
        AllChem.ComputeGasteigerCharges(mol)
        for atom in mol.GetAtoms():
            c = atom.GetDoubleProp('_GasteigerCharge')
            features[atom.GetIdx(), 7] = c if np.isfinite(c) else 0.0

    # 8: Hydrophobicity (Crippen LogP contributions)
    contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    for i, (logp, _) in enumerate(contribs):
        features[i, 8] = logp

    # 9: HBD
    for match in mol.GetSubstructMatches(Lipinski.HDonorSmarts):
        features[match[0], 9] = 1.0

    # 10: HBA
    for match in mol.GetSubstructMatches(Lipinski.HAcceptorSmarts):
        features[match[0], 10] = 1.0

    # 11: Aromaticity
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            features[atom.GetIdx(), 11] = 1.0

    # 12: Positive ionizable
    pos_smarts = Chem.MolFromSmarts(
        "[+1,+2,$([NH2]-C(=N)N),$([NH]=C(N)N),$([nH]1ccnc1)]"
    )
    for match in mol.GetSubstructMatches(pos_smarts):
        features[match[0], 12] = 1.0

    # 13: Negative ionizable
    neg_smarts = Chem.MolFromSmarts(
        "[-1,-2,$([CX3](=O)[OH]),$([CX3](=O)[O-]),$([SX4](=O)(=O)[OH])]"
    )
    for match in mol.GetSubstructMatches(neg_smarts):
        features[match[0], 13] = 1.0

    # 14: Hybridization (normalized: sp=0.33, sp2=0.67, sp3=1.0)
    hyb_map = {
        Chem.HybridizationType.SP: 0.33,
        Chem.HybridizationType.SP2: 0.67,
        Chem.HybridizationType.SP3: 1.0,
    }
    for atom in mol.GetAtoms():
        features[atom.GetIdx(), 14] = hyb_map.get(atom.GetHybridization(), 0.0)

    # 15: Ring membership
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            features[atom.GetIdx(), 15] = 1.0

    return features, list(LIGAND_CHANNEL_NAMES)


def compute_protein_channels(
    mol,
    atom_metadata: Optional[list[dict]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract per-atom feature matrix for protein voxelization.

    Accepts either an RDKit Mol with PDB residue info, or a _SimpleMol
    built from atom_metadata.

    Args:
        mol: RDKit molecule or _SimpleMol with PDB residue info.
        atom_metadata: If mol is None, used to build a _SimpleMol.

    Returns:
        (features, channel_names) where features is (N, 16) float32.
    """
    if mol is None and atom_metadata is not None:
        from ..surface.features import _build_simple_protein_mol
        mol = _build_simple_protein_mol(atom_metadata)

    if mol is None:
        raise ValueError(
            "compute_protein_channels requires either 'mol' or 'atom_metadata'."
        )

    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, 16), dtype=np.float32)

    # Kyte-Doolittle hydrophobicity scale
    _KD_SCALE = {
        'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
        'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
        'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
        'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
    }
    # Charged residue atoms
    _CHARGED_RESIDUES = {
        'ASP': {'OD1': -0.5, 'OD2': -0.5},
        'GLU': {'OE1': -0.5, 'OE2': -0.5},
        'LYS': {'NZ': 1.0},
        'ARG': {'NH1': 0.5, 'NH2': 0.5},
        'HIS': {'ND1': 0.5, 'NE2': 0.5},
    }
    # Ionizable residue names
    _POS_RESIDUES = {'LYS', 'ARG', 'HIS'}
    _NEG_RESIDUES = {'ASP', 'GLU'}

    backbone_names = {'N', 'CA', 'C', 'O'}
    # Atom type mapping for protein: C=0, N=1, O=2, S=3, P=4, Other=5
    prot_type_map = {6: 0, 7: 1, 8: 2, 16: 3, 15: 4}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        res = atom.GetPDBResidueInfo()
        res_name = ""
        atom_name = ""
        b_factor = 0.0

        if res:
            res_name = res.GetResidueName().strip()
            atom_name = res.GetName().strip()
            b_factor = res.GetTempFactor()

        # 0: Occupancy
        features[idx, 0] = 1.0

        # 1-6: Atom type one-hot (C, N, O, S, P, Other)
        t = prot_type_map.get(atom.GetAtomicNum(), 5)
        features[idx, 1 + t] = 1.0

        # 7: Partial charge (residue-based)
        if res_name in _CHARGED_RESIDUES:
            features[idx, 7] = _CHARGED_RESIDUES[res_name].get(atom_name, 0.0)

        # 8: Hydrophobicity (Kyte-Doolittle per residue)
        features[idx, 8] = _KD_SCALE.get(res_name, 0.0)

        # 9: HBD (nitrogen atoms)
        if atom.GetAtomicNum() == 7:
            features[idx, 9] = 1.0

        # 10: HBA (oxygen atoms)
        if atom.GetAtomicNum() == 8:
            features[idx, 10] = 1.0

        # 11: Aromaticity (PHE, TYR, TRP, HIS aromatic residues)
        if res_name in ('PHE', 'TYR', 'TRP', 'HIS'):
            if atom_name not in backbone_names:
                features[idx, 11] = 1.0

        # 12: Positive ionizable
        if res_name in _POS_RESIDUES:
            features[idx, 12] = 1.0

        # 13: Negative ionizable
        if res_name in _NEG_RESIDUES:
            features[idx, 13] = 1.0

        # 14: Backbone
        if atom_name in backbone_names:
            features[idx, 14] = 1.0

        # 15: B-factor (raw, will be normalized after)
        features[idx, 15] = b_factor

    # Normalize B-factor to [0, 1] range
    bf = features[:, 15]
    bf_min, bf_max = bf.min(), bf.max()
    if bf_max - bf_min > 1e-6:
        features[:, 15] = (bf - bf_min) / (bf_max - bf_min)

    # Normalize hydrophobicity to [-1, 1] using global Kyte-Doolittle range
    KD_MIN, KD_MAX = -4.5, 4.5
    features[:, 8] = 2.0 * (features[:, 8] - KD_MIN) / (KD_MAX - KD_MIN) - 1.0

    return features, list(PROTEIN_CHANNEL_NAMES)


def voxelize_ligand(
    mol,
    center: Optional[np.ndarray] = None,
    resolution: float = VOXEL_DEFAULT_RESOLUTION,
    box_size: Optional[int] = VOXEL_DEFAULT_BOX_SIZE,
    padding: float = VOXEL_DEFAULT_PADDING,
    sigma_scale: float = VOXEL_DEFAULT_SIGMA_SCALE,
    cutoff_sigma: float = VOXEL_DEFAULT_CUTOFF_SIGMA,
    charge_method: str = "gasteiger",
) -> dict:
    """Voxelize a ligand molecule into a multi-channel 3D grid.

    Args:
        mol: RDKit molecule with 3D conformer.
        center: Grid center (3,). If None, uses ligand centroid.
        resolution: Angstrom per voxel (default: 1.0).
        box_size: Fixed grid dimension (default: 24). None for adaptive.
        padding: Padding in Angstrom when box_size is None.
        sigma_scale: Multiplier on VdW radius for Gaussian sigma.
        cutoff_sigma: Gaussian cutoff in units of sigma.
        charge_method: "gasteiger" or "mmff94".

    Returns:
        Dict with keys:
            - "voxel": (16, D, H, W) float32
            - "channel_names": list of 16 channel names
            - "grid_origin": (3,) origin of grid
            - "grid_shape": (D, H, W)
            - "resolution": float
    """
    conf = mol.GetConformer()
    positions = np.array(conf.GetPositions(), dtype=np.float32)

    # Per-atom sigma from VdW radii
    sigmas = np.array(
        [
            VDW_RADIUS.get(a.GetAtomicNum(), DEFAULT_VDW_RADIUS) * sigma_scale
            for a in mol.GetAtoms()
        ],
        dtype=np.float32,
    )

    # Compute grid params
    grid_origin, grid_shape = _compute_grid_params(
        positions, center=center, box_size=box_size,
        resolution=resolution, padding=padding,
    )

    # Compute per-atom features
    features, channel_names = compute_ligand_channels(mol, charge_method=charge_method)

    # Gaussian smear to grid
    voxel = gaussian_smear_to_grid(
        positions, features, sigmas, grid_origin, grid_shape,
        resolution=resolution, cutoff_sigma=cutoff_sigma,
    )

    return {
        "voxel": voxel,
        "channel_names": channel_names,
        "grid_origin": grid_origin,
        "grid_shape": grid_shape,
        "resolution": resolution,
    }


def voxelize_protein(
    positions: np.ndarray,
    radii: np.ndarray,
    mol=None,
    atom_metadata: Optional[list[dict]] = None,
    center: Optional[np.ndarray] = None,
    resolution: float = VOXEL_DEFAULT_RESOLUTION,
    box_size: Optional[int] = VOXEL_DEFAULT_BOX_SIZE,
    padding: float = VOXEL_DEFAULT_PADDING,
    sigma_scale: float = VOXEL_DEFAULT_SIGMA_SCALE,
    cutoff_sigma: float = VOXEL_DEFAULT_CUTOFF_SIGMA,
) -> dict:
    """Voxelize protein atoms into a multi-channel 3D grid.

    Args:
        positions: Atom positions (N, 3).
        radii: VdW radii (N,) for sigma computation.
        mol: RDKit mol or _SimpleMol with PDB residue info.
        atom_metadata: Used to build _SimpleMol if mol is None.
        center: Grid center (3,). If None, uses centroid.
        resolution: Angstrom per voxel (default: 1.0).
        box_size: Fixed grid dimension (default: 24). None for adaptive.
        padding: Padding in Angstrom when box_size is None.
        sigma_scale: Multiplier on VdW radius for Gaussian sigma.
        cutoff_sigma: Gaussian cutoff in units of sigma.

    Returns:
        Dict with keys:
            - "voxel": (16, D, H, W) float32
            - "channel_names": list of 16 channel names
            - "grid_origin": (3,) origin of grid
            - "grid_shape": (D, H, W)
            - "resolution": float
    """
    positions = np.asarray(positions, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    sigmas = radii * sigma_scale

    grid_origin, grid_shape = _compute_grid_params(
        positions, center=center, box_size=box_size,
        resolution=resolution, padding=padding,
    )

    features, channel_names = compute_protein_channels(
        mol, atom_metadata=atom_metadata,
    )

    voxel = gaussian_smear_to_grid(
        positions, features, sigmas, grid_origin, grid_shape,
        resolution=resolution, cutoff_sigma=cutoff_sigma,
    )

    return {
        "voxel": voxel,
        "channel_names": channel_names,
        "grid_origin": grid_origin,
        "grid_shape": grid_shape,
        "resolution": resolution,
    }


def voxelize_complex(
    ligand_mol,
    protein_positions: np.ndarray,
    protein_radii: np.ndarray,
    protein_mol=None,
    protein_atom_metadata: Optional[list[dict]] = None,
    center: Optional[np.ndarray] = None,
    resolution: float = VOXEL_DEFAULT_RESOLUTION,
    box_size: Optional[int] = VOXEL_DEFAULT_BOX_SIZE,
    padding: float = VOXEL_DEFAULT_PADDING,
    sigma_scale: float = VOXEL_DEFAULT_SIGMA_SCALE,
    cutoff_sigma: float = VOXEL_DEFAULT_CUTOFF_SIGMA,
    charge_method: str = "gasteiger",
) -> dict:
    """Voxelize a protein-ligand complex into separate + combined grids.

    Both protein and ligand share the same grid origin and shape,
    enabling direct channel-wise concatenation.

    Args:
        ligand_mol: RDKit molecule with 3D conformer.
        protein_positions: Protein atom positions (N, 3).
        protein_radii: Protein atom VdW radii (N,).
        protein_mol: RDKit mol or _SimpleMol for protein.
        protein_atom_metadata: Metadata for building _SimpleMol.
        center: Grid center. If None, uses ligand centroid.
        resolution: Angstrom per voxel.
        box_size: Fixed grid dimension.
        padding: Padding when box_size is None.
        sigma_scale: VdW radius multiplier for Gaussian sigma.
        cutoff_sigma: Gaussian cutoff in sigma units.
        charge_method: "gasteiger" or "mmff94" (ligand only).

    Returns:
        Dict with keys:
            - "ligand_voxel": (16, D, H, W)
            - "protein_voxel": (16, D, H, W)
            - "combined_voxel": (32, D, H, W)
            - "ligand_channel_names": list[str]
            - "protein_channel_names": list[str]
            - "grid_origin": (3,)
            - "grid_shape": (D, H, W)
            - "resolution": float
    """
    # Compute ligand positions once
    lig_pos = np.array(
        ligand_mol.GetConformer().GetPositions(), dtype=np.float32
    )

    # Use ligand centroid as grid center by default
    if center is None:
        center = lig_pos.mean(axis=0)

    # Shared grid
    all_positions = np.concatenate(
        [lig_pos, np.asarray(protein_positions, dtype=np.float32)],
        axis=0,
    )
    grid_origin, grid_shape = _compute_grid_params(
        all_positions, center=center, box_size=box_size,
        resolution=resolution, padding=padding,
    )
    lig_sigmas = np.array(
        [VDW_RADIUS.get(a.GetAtomicNum(), DEFAULT_VDW_RADIUS) * sigma_scale
         for a in ligand_mol.GetAtoms()],
        dtype=np.float32,
    )
    lig_features, lig_names = compute_ligand_channels(
        ligand_mol, charge_method=charge_method,
    )
    lig_voxel = gaussian_smear_to_grid(
        lig_pos, lig_features, lig_sigmas, grid_origin, grid_shape,
        resolution=resolution, cutoff_sigma=cutoff_sigma,
    )

    # Protein voxel
    prot_pos = np.asarray(protein_positions, dtype=np.float32)
    prot_sigmas = np.asarray(protein_radii, dtype=np.float32) * sigma_scale
    prot_features, prot_names = compute_protein_channels(
        protein_mol, atom_metadata=protein_atom_metadata,
    )
    prot_voxel = gaussian_smear_to_grid(
        prot_pos, prot_features, prot_sigmas, grid_origin, grid_shape,
        resolution=resolution, cutoff_sigma=cutoff_sigma,
    )

    combined = np.concatenate([lig_voxel, prot_voxel], axis=0)

    return {
        "ligand_voxel": lig_voxel,
        "protein_voxel": prot_voxel,
        "combined_voxel": combined,
        "ligand_channel_names": lig_names,
        "protein_channel_names": prot_names,
        "grid_origin": grid_origin,
        "grid_shape": grid_shape,
        "resolution": resolution,
    }
