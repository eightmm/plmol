"""
Vertex Creation and Feature Extraction Module

This module provides functions for:
1. Creating SAS point cloud surfaces from molecular structures
2. Extracting dMaSIF-inspired features at each vertex

Features:
    - Multi-scale PCA curvature (mean + Gaussian at 5 radii)
    - KNN-based atom-to-vertex mapping (K=16, no global blur)
    - Consistent [-1, 1] normalization across all features

Functions:
    - create_surface_points: Create SAS point cloud from atom positions and radii
    - compute_all_vertex_features: Extract features at each vertex
"""

from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Lipinski
from scipy.spatial import cKDTree
from ..constants import (
    SURFACE_DEFAULT_CURVATURE_SCALES,
    SURFACE_DEFAULT_KNN_ATOMS,
    SURFACE_DEFAULT_POINTS_PER_ATOM,
    SURFACE_DEFAULT_PROBE_RADIUS,
)

# Re-export for convenience
CURVATURE_SCALES = SURFACE_DEFAULT_CURVATURE_SCALES
SURFACE_KNN_ATOMS = SURFACE_DEFAULT_KNN_ATOMS


def _normalize_to_range(arr: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Normalize array to [lo, hi] using robust min/max (1st/99th percentile).

    NaN/Inf values are replaced with 0.0 (midpoint of [-1, 1]).
    Constant arrays return all zeros.
    """
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)
    finite_vals = arr[finite_mask]
    p1, p99 = np.percentile(finite_vals, [1, 99])
    if p99 - p1 < 1e-8:
        return np.zeros_like(arr)
    clipped = np.clip(arr, p1, p99)
    scaled = (clipped - p1) / (p99 - p1)  # [0, 1]
    result = scaled * (hi - lo) + lo
    result[~finite_mask] = 0.0
    return result


def _build_knn_weights(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    k: int = SURFACE_KNN_ATOMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build KNN-based distance weights for atom-to-vertex mapping.

    Uses cKDTree for O(N*K) memory instead of O(N*M) full distance matrix.

    Args:
        verts: Mesh vertices (N, 3)
        atom_positions: Atom positions (M, 3)
        k: Number of nearest atoms per vertex

    Returns:
        knn_idx: (N, K) indices of K nearest atoms per vertex
        knn_weights: (N, K) normalized inverse-distance weights (rows sum to 1)
        knn_dists: (N, K) Euclidean distances to K nearest atoms
    """
    n_atoms = len(atom_positions)
    k = min(k, n_atoms)

    tree = cKDTree(atom_positions)
    knn_dists, knn_idx = tree.query(verts, k=k, workers=-1)

    # cKDTree.query returns 1D arrays when k=1; ensure 2D
    if k == 1:
        knn_dists = knn_dists[:, None]
        knn_idx = knn_idx[:, None]

    knn_dists = knn_dists.astype(np.float32)
    knn_idx = knn_idx.astype(np.intp)

    knn_dists_clamped = np.maximum(knn_dists, 0.5)
    knn_weights = 1.0 / knn_dists_clamped
    row_sums = knn_weights.sum(axis=1, keepdims=True)
    knn_weights = knn_weights / np.maximum(row_sums, 1e-8)

    return knn_idx, knn_weights, knn_dists


def _knn_map_scalar(
    knn_idx: np.ndarray,
    knn_weights: np.ndarray,
    atom_features: np.ndarray,
) -> np.ndarray:
    """Map per-atom scalar features to vertices via KNN weights.

    Args:
        knn_idx: (N, K) KNN atom indices
        knn_weights: (N, K) normalized weights
        atom_features: (M,) per-atom scalar feature

    Returns:
        (N,) per-vertex feature
    """
    return (knn_weights * atom_features[knn_idx]).sum(axis=1)


def _knn_map_matrix(
    knn_idx: np.ndarray,
    knn_weights: np.ndarray,
    atom_features: np.ndarray,
) -> np.ndarray:
    """Map per-atom vector/matrix features to vertices via KNN weights.

    Args:
        knn_idx: (N, K) KNN atom indices
        knn_weights: (N, K) normalized weights
        atom_features: (M, D) per-atom feature matrix

    Returns:
        (N, D) per-vertex feature matrix
    """
    # atom_features[knn_idx] -> (N, K, D)
    gathered = atom_features[knn_idx]
    # knn_weights[:, :, None] -> (N, K, 1) for broadcasting
    return (knn_weights[:, :, None] * gathered).sum(axis=1)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately uniform points on a unit sphere (Fibonacci lattice).

    Returns:
        (n, 3) array of unit vectors.
    """
    indices = np.arange(n, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / golden
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z]).astype(np.float32)


def create_surface_points(
    positions: np.ndarray,
    radii: np.ndarray,
    n_points_per_atom: int = SURFACE_DEFAULT_POINTS_PER_ATOM,
    probe_radius: float = SURFACE_DEFAULT_PROBE_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast SAS point cloud via Shrake-Slee sphere sampling.

    Generates solvent-accessible surface points without mesh construction.
    Much faster than marching cubes for applications that don't need faces.

    Algorithm:
        1. Fibonacci sphere generates uniform points on each atom sphere
        2. cKDTree identifies buried points (overlapping with neighbouring atoms)
        3. Surviving points form the SAS point cloud

    Args:
        positions: Atom positions (N, 3)
        radii: VdW radii for each atom (N,)
        n_points_per_atom: Number of sample points per atom sphere (default: 100)
        probe_radius: Solvent probe radius in Angstroms (default: 1.4)

    Returns:
        Tuple of (points, normals), each (P, 3). normals are radial outward vectors.
    """
    n_atoms = len(positions)
    if n_atoms == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    positions = np.asarray(positions, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    expanded_radii = radii + probe_radius

    unit_sphere = _fibonacci_sphere(n_points_per_atom)

    # Generate all candidate points: (N * n_points_per_atom, 3)
    # For each atom i, points = expanded_radii[i] * unit_sphere + positions[i]
    all_points = (
        expanded_radii[:, None, None] * unit_sphere[None, :, :]
        + positions[:, None, :]
    )  # (N, n_pts, 3)
    all_normals = np.broadcast_to(
        unit_sphere[None, :, :], (n_atoms, n_points_per_atom, 3)
    ).copy()

    # Atom index for each candidate point
    atom_ids = np.repeat(np.arange(n_atoms), n_points_per_atom)

    all_points_flat = all_points.reshape(-1, 3)
    all_normals_flat = all_normals.reshape(-1, 3)

    # Build tree of atom centres for neighbour lookup
    tree = cKDTree(positions)
    max_radius = float(expanded_radii.max())

    # For each point, check if it is buried inside any *other* atom's expanded sphere
    neighbours = tree.query_ball_point(all_points_flat, r=max_radius, workers=-1)

    exposed = np.ones(len(all_points_flat), dtype=bool)
    for pt_idx in range(len(all_points_flat)):
        owner = atom_ids[pt_idx]
        for atom_j in neighbours[pt_idx]:
            if atom_j == owner:
                continue
            dist = np.linalg.norm(all_points_flat[pt_idx] - positions[atom_j])
            if dist < expanded_radii[atom_j]:
                exposed[pt_idx] = False
                break

    points = all_points_flat[exposed]
    normals = all_normals_flat[exposed]

    return points.astype(np.float32), normals.astype(np.float32)


def build_surface_dict(
    verts: np.ndarray,
    faces: Optional[np.ndarray],
    normals: np.ndarray,
) -> dict:
    """
    Build a standardized surface dictionary.

    Returns:
        Dict with "points", "normals" (and legacy "verts").
        "faces" is included only when *faces* is not None.
    """
    d: dict = {
        "points": verts,
        "normals": normals,
        "verts": verts,
    }
    if faces is not None:
        d["faces"] = faces
    return d


def _compute_pca_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    tree: Optional[cKDTree] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate mean and Gaussian curvature from a point cloud via local PCA.

    Uses a **KNN-based** approach where K is adapted to the scale *radius*.
    This is more robust than a fixed-radius ball query because it guarantees
    a minimum number of neighbours even at small scales, and avoids the
    "everything inside" degeneracy at large scales.

    Heuristic: ``K = clamp(int(radius * 8), 6, N//2)``
    – smaller radius → fewer neighbours (fine detail)
    – larger radius → more neighbours (coarse curvature)

    Eigenvalue ratios of the local 3×3 covariance matrix:
        lambda_0 <= lambda_1 <= lambda_2

    * **mean curvature proxy** – ``lambda_0 / total``
    * **Gaussian curvature proxy** – ``lambda_0 * lambda_1 / total²``

    Both are normalised to [-1, 1] before return.

    Args:
        points: (N, 3) surface positions.
        normals: (N, 3) surface normals.
        radius: curvature scale (Å) – controls K.
        tree: optional pre-built cKDTree of *points*.

    Returns:
        (mean_curv, gauss_curv) each (N,), normalised to [-1, 1].
    """
    n = len(points)
    mean_curv = np.zeros(n, dtype=np.float32)
    gauss_curv = np.zeros(n, dtype=np.float32)

    if n < 6:
        return mean_curv, gauss_curv

    if tree is None:
        tree = cKDTree(points)

    # Adaptive K based on scale
    k = max(6, min(int(radius * 8), n // 2))
    _, knn_idx = tree.query(points, k=k, workers=-1)
    if knn_idx.ndim == 1:
        knn_idx = knn_idx[:, None]

    # Vectorised PCA: batch covariance for all points
    neighbours = points[knn_idx]                     # (N, K, 3)
    centroid = neighbours.mean(axis=1, keepdims=True)  # (N, 1, 3)
    centered = neighbours - centroid                    # (N, K, 3)

    # Covariance matrices: (N, 3, 3)
    covs = np.einsum('nki,nkj->nij', centered, centered) / k

    # Batch eigenvalues: (N, 3) ascending
    eigvals = np.linalg.eigvalsh(covs)
    eigvals = np.maximum(eigvals, 0.0)

    total = eigvals.sum(axis=1)  # (N,)
    valid = total > 1e-12

    # Mean curvature proxy
    mean_curv[valid] = eigvals[valid, 0] / total[valid]

    # Gaussian curvature proxy
    gauss_curv[valid] = (
        eigvals[valid, 0] * eigvals[valid, 1]
    ) / (total[valid] ** 2)

    return _normalize_to_range(mean_curv), _normalize_to_range(gauss_curv)


# =========================================================================
# Modular Feature Computation Functions
# =========================================================================


def compute_pointcloud_geometry(
    points: np.ndarray,
    normals: np.ndarray,
    curvature_scales: tuple[float, ...] = CURVATURE_SCALES,
    verbose: bool = False,
) -> dict:
    """Compute geometry features from a point cloud.

    Uses PCA-based curvature estimation (adaptive KNN per scale)
    instead of mesh-based discrete curvature.

    Can be used independently for point-cloud-based surface analysis.

    Args:
        points: Point cloud positions (N, 3)
        normals: Point normals (N, 3)
        curvature_scales: Radii for multi-scale curvature computation
        verbose: Whether to print progress messages

    Returns:
        Dict with keys:
            - 'mean_curvature': (N, n_scales) normalized to [-1, 1]
            - 'gaussian_curvature': (N, n_scales) normalized to [-1, 1]
            - 'vertex_normal': (N, 3) unit vectors
    """
    n_verts = len(points)
    n_scales = len(curvature_scales)
    mean_curvatures = np.zeros((n_verts, n_scales), dtype=np.float32)
    gauss_curvatures = np.zeros((n_verts, n_scales), dtype=np.float32)
    vertex_normals = np.asarray(normals, dtype=np.float32)

    if n_verts >= 4:
        if verbose:
            print("  Computing PCA curvature for point cloud...")
        pc_tree = cKDTree(points)
        with ThreadPoolExecutor(max_workers=n_scales) as executor:
            futures = {
                executor.submit(
                    _compute_pca_curvature, points, normals, radius, pc_tree
                ): i
                for i, radius in enumerate(curvature_scales)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    mc, gc = future.result()
                    mean_curvatures[:, i] = mc
                    gauss_curvatures[:, i] = gc
                except Exception as e:
                    if verbose:
                        print(f"  PCA curvature at radius={curvature_scales[i]} failed: {e}")

    return {
        'mean_curvature': mean_curvatures,
        'gaussian_curvature': gauss_curvatures,
        'vertex_normal': vertex_normals,
    }


def compute_chemical_features(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    mol,
    is_ligand: bool = True,
    charge_method: str = "gasteiger",
    knn_atoms: int = SURFACE_KNN_ATOMS,
    verbose: bool = False,
    _knn_data: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> dict:
    """Compute chemical features mapped from atoms to surface vertices.

    Features include electrostatic potential (Coulomb), hydrophobicity
    (Crippen LogP / Kyte-Doolittle), H-bond donors/acceptors, molar
    refractivity, aromaticity, and ionizability.

    Can be used independently for both ligand and protein surfaces.

    Args:
        verts: Surface points (N, 3)
        atom_positions: Atom positions (M, 3)
        mol: RDKit molecule or _SimpleMol for protein
        is_ligand: Whether this is a ligand (True) or protein (False)
        charge_method: "gasteiger" or "mmff94" (ligand only)
        knn_atoms: Number of nearest atoms per vertex
        verbose: Whether to print progress messages
        _knn_data: Pre-built (knn_idx, knn_weights, knn_dists) to avoid recomputation

    Returns:
        Dict with keys: 'electrostatic', 'hydrophobicity', 'hbd', 'hba',
        'molar_refractivity', 'aromaticity', 'pos_ionizable', 'neg_ionizable'
    """
    if _knn_data is not None:
        knn_idx, knn_weights, knn_dists = _knn_data
    else:
        knn_idx, knn_weights, knn_dists = _build_knn_weights(verts, atom_positions, k=knn_atoms)

    knn_dists_clamped = np.maximum(knn_dists, 0.5)

    # --- Electrostatic potential ---
    if verbose:
        print("  Computing electrostatic potential...")
    if is_ligand:
        if charge_method == "mmff94":
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                charges = np.array(
                    [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())],
                    dtype=np.float32,
                )
            else:
                if verbose:
                    print("  MMFF94 failed, falling back to Gasteiger")
                AllChem.ComputeGasteigerCharges(mol)
                charges = np.array(
                    [a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()],
                    dtype=np.float32,
                )
                np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        else:  # "gasteiger"
            AllChem.ComputeGasteigerCharges(mol)
            charges = np.array(
                [a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()],
                dtype=np.float32,
            )
            bad_mask = ~np.isfinite(charges)
            if verbose and bad_mask.any():
                print(f"  Warning: {bad_mask.sum()} atoms had NaN/Inf Gasteiger charges (zeroed)")
            np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    else:
        _CHARGED_RESIDUES = {
            'ASP': {'OD1': -0.5, 'OD2': -0.5},
            'GLU': {'OE1': -0.5, 'OE2': -0.5},
            'LYS': {'NZ': 1.0},
            'ARG': {'NH1': 0.5, 'NH2': 0.5},
            'HIS': {'ND1': 0.5, 'NE2': 0.5},
        }
        charges = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for atom in mol.GetAtoms():
            res = atom.GetPDBResidueInfo()
            if res:
                res_name = res.GetResidueName().strip()
                atom_name = res.GetName().strip()
                if res_name in _CHARGED_RESIDUES:
                    charges[atom.GetIdx()] = _CHARGED_RESIDUES[res_name].get(atom_name, 0.0)

    knn_charges = charges[knn_idx]
    electrostatic_raw = np.sum(knn_charges / knn_dists_clamped, axis=1)
    electrostatic = _normalize_to_range(electrostatic_raw)

    # --- Chemical features ---
    if verbose:
        print("  Computing chemical features...")
    if is_ligand:
        contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
        logp_contribs = np.array([c[0] for c in contribs], dtype=np.float32)
        mr_contribs = np.array([c[1] for c in contribs], dtype=np.float32)

        hbd_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for match in mol.GetSubstructMatches(Lipinski.HDonorSmarts):
            hbd_atoms[match[0]] = 1.0

        hba_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for match in mol.GetSubstructMatches(Lipinski.HAcceptorSmarts):
            hba_atoms[match[0]] = 1.0

        aromaticity_atoms = np.array(
            [1.0 if a.GetIsAromatic() else 0.0 for a in mol.GetAtoms()],
            dtype=np.float32,
        )

        pos_smarts = Chem.MolFromSmarts("[+1,+2,$([NH2]-C(=N)N),$([NH]=C(N)N),$([nH]1ccnc1)]")
        pos_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for match in mol.GetSubstructMatches(pos_smarts):
            pos_atoms[match[0]] = 1.0

        neg_smarts = Chem.MolFromSmarts("[-1,-2,$([CX3](=O)[OH]),$([CX3](=O)[O-]),$([SX4](=O)(=O)[OH])]")
        neg_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for match in mol.GetSubstructMatches(neg_smarts):
            neg_atoms[match[0]] = 1.0
    else:
        _KD_SCALE = {
            'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
            'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
            'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
            'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
        }
        logp_contribs = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        mr_contribs = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        hbd_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        hba_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        aromaticity_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        pos_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        neg_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)

        for atom in mol.GetAtoms():
            res = atom.GetPDBResidueInfo()
            if res:
                res_name = res.GetResidueName().strip()
                logp_contribs[atom.GetIdx()] = _KD_SCALE.get(res_name, 0.0)
                if atom.GetAtomicNum() == 7:
                    hbd_atoms[atom.GetIdx()] = 1.0
                if atom.GetAtomicNum() == 8:
                    hba_atoms[atom.GetIdx()] = 1.0

    hydrophobicity = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, logp_contribs))
    molar_refractivity = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, mr_contribs))
    hbd = np.clip(_knn_map_scalar(knn_idx, knn_weights, hbd_atoms), 0, 1)
    hba = np.clip(_knn_map_scalar(knn_idx, knn_weights, hba_atoms), 0, 1)
    aromaticity = np.clip(_knn_map_scalar(knn_idx, knn_weights, aromaticity_atoms), 0, 1)
    pos_ionizable = np.clip(_knn_map_scalar(knn_idx, knn_weights, pos_atoms), 0, 1)
    neg_ionizable = np.clip(_knn_map_scalar(knn_idx, knn_weights, neg_atoms), 0, 1)

    return {
        'electrostatic': electrostatic,
        'hydrophobicity': hydrophobicity,
        'hbd': hbd,
        'hba': hba,
        'molar_refractivity': molar_refractivity,
        'aromaticity': aromaticity,
        'pos_ionizable': pos_ionizable,
        'neg_ionizable': neg_ionizable,
    }


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
        print("  Computing ligand type features...")

    # Atom type one-hot (C, N, O, S, Halogen, Other)
    atom_type_map = {6: 0, 7: 1, 8: 2, 16: 3, 9: 4, 17: 4, 35: 4, 53: 4}
    n_mol_atoms = mol.GetNumAtoms()
    atom_types = np.array(
        [atom_type_map.get(a.GetAtomicNum(), 5) for a in mol.GetAtoms()],
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
) -> dict:
    """Compute protein-specific type features mapped to surface vertices.

    Includes residue type one-hot encoding (20 standard amino acids),
    backbone vs sidechain classification, and B-factor (flexibility).

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
        'b_factor' (N,)
    """
    if _knn_data is not None:
        knn_idx, knn_weights, _ = _knn_data
    else:
        knn_idx, knn_weights, _ = _build_knn_weights(verts, atom_positions, k=knn_atoms)

    if verbose:
        print("  Computing protein type features...")

    n_prot_atoms = mol.GetNumAtoms()
    res_names: list[str] = []
    atom_names: list[str] = []
    b_factors = np.zeros(n_prot_atoms, dtype=np.float32)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        res = atom.GetPDBResidueInfo()
        if res:
            res_names.append(res.GetResidueName().strip())
            atom_names.append(res.GetName().strip())
            b_factors[idx] = res.GetTempFactor()
        else:
            res_names.append("")
            atom_names.append("")

    # Residue type one-hot (20 amino acids)
    AA_LIST = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    ]
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}
    res_indices = np.array(
        [aa_to_idx.get(rn, -1) for rn in res_names], dtype=np.intp,
    )
    residue_onehot = np.zeros((n_prot_atoms, 20), dtype=np.float32)
    valid = res_indices >= 0
    residue_onehot[np.where(valid)[0], res_indices[valid]] = 1.0
    vertex_residue_type = _knn_map_matrix(knn_idx, knn_weights, residue_onehot)

    # Backbone vs sidechain
    backbone_names = {'N', 'CA', 'C', 'O'}
    is_backbone = np.array(
        [1.0 if an in backbone_names else 0.0 for an in atom_names],
        dtype=np.float32,
    )
    vertex_backbone = np.clip(_knn_map_scalar(knn_idx, knn_weights, is_backbone), 0, 1)

    # B-factor (flexibility)
    vertex_bfactor = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, b_factors))

    return {
        'residue_type': vertex_residue_type,
        'is_backbone': vertex_backbone,
        'b_factor': vertex_bfactor,
    }


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
        print(f"  Mapping {len(extra_atom_features)} extra atom features to vertices...")

    result: dict[str, np.ndarray] = {}
    for name, atom_feat in extra_atom_features.items():
        atom_feat = np.asarray(atom_feat, dtype=np.float32)
        if atom_feat.ndim == 1:
            mapped = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, atom_feat))
        elif atom_feat.ndim == 2:
            mapped = _knn_map_matrix(knn_idx, knn_weights, atom_feat)
        else:
            raise ValueError(f"extra_atom_features['{name}'] must be 1D or 2D, got {atom_feat.ndim}D")
        result[name] = mapped

    return result


# =========================================================================
# Backward-Compatible Wrapper
# =========================================================================


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
        print(f"  Building KNN atom-to-vertex mapping (K={knn_atoms})...")
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
        type_feat['b_factor'] = np.zeros(n_verts, dtype=np.float32)
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
                _atom_labels = ["C", "N", "O", "S", "Hal", "Other"]
                names.extend([f"{key}_{l}" for l in _atom_labels])
            elif key == "residue_type":
                _aa = [
                    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                ]
                names.extend([f"{key}_{aa}" for aa in _aa])
            else:
                names.extend([f"{key}_{i}" for i in range(values.shape[1])])
        else:
            raise ValueError(f"Unsupported feature shape for {key}: {values.shape}")

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

    Features: multi-scale curvature (10D) + normals (3D) + chemical (9D)
    + atom type (6D) + hybridization/ring (3D) = 31D total.
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


class _SimplePDBResidueInfo:
    def __init__(self, res_name: str, atom_name: str, b_factor: float = 0.0):
        self._res_name = res_name
        self._atom_name = atom_name
        self._b_factor = b_factor

    def GetResidueName(self) -> str:
        return self._res_name

    def GetName(self) -> str:
        return self._atom_name

    def GetTempFactor(self) -> float:
        return self._b_factor


class _SimpleAtom:
    def __init__(self, res_name: str, atom_name: str, element: str,
                 b_factor: float = 0.0, idx: int = 0):
        self._idx = idx
        self._res_info = _SimplePDBResidueInfo(res_name, atom_name, b_factor=b_factor)
        try:
            self._atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
        except Exception:
            self._atomic_num = 0

    def GetIdx(self) -> int:
        return self._idx

    def GetPDBResidueInfo(self) -> _SimplePDBResidueInfo:
        return self._res_info

    def GetAtomicNum(self) -> int:
        return self._atomic_num


class _SimpleMol:
    def __init__(self, atoms: list[_SimpleAtom]):
        self._atoms = atoms

    def GetAtoms(self):
        return self._atoms

    def GetNumAtoms(self) -> int:
        return len(self._atoms)


def _build_simple_protein_mol(atom_metadata: list[dict]) -> _SimpleMol:
    atoms = [
        _SimpleAtom(
            res_name=meta.get("res_name", "UNK"),
            atom_name=meta.get("atom_name", ""),
            element=meta.get("element", ""),
            b_factor=meta.get("b_factor", 0.0),
            idx=i,
        )
        for i, meta in enumerate(atom_metadata)
    ]
    return _SimpleMol(atoms)


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

    Features: multi-scale curvature (10D) + normals (3D) + chemical (5D)
    + residue type (20D) + backbone/bfactor (2D) = 40D total.
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
        "b_factor",
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
