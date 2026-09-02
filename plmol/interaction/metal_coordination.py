"""
Metal Coordination Support.

Provides detection, geometry classification, and feature encoding for
metal coordination sites in protein structures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from ..constants.interactions import (
    METAL_COORDINATION_CUTOFF,
    COMMON_COORDINATION_GEOMETRIES,
    METAL_PREFERRED_DONORS,
    SQUARE_PLANAR_LINEAR_ANGLE,
    TRIGONAL_BIPYRAMIDAL_LINEAR_ANGLE,
)


@dataclass
class MetalSite:
    metal_element: str
    metal_coords: np.ndarray
    coordinating_atoms: List[Dict]  # [{atom_name, res_name, chain_id, coords, distance}]
    coordination_number: int
    geometry: str  # "linear", "trigonal_planar", "tetrahedral", etc.


def detect_metal_sites(
    atom_coords: np.ndarray,
    atom_metadata: List[Dict],
    metal_indices: List[int],
    distance_cutoff: float = METAL_COORDINATION_CUTOFF,
) -> List[MetalSite]:
    """Detect metal coordination sites.

    Args:
        atom_coords: (N, 3) array of atom coordinates.
        atom_metadata: List of dicts with keys: atom_name, res_name, chain_id, element.
        metal_indices: Indices of metal atoms in atom_coords/atom_metadata.
        distance_cutoff: Maximum coordination distance in Angstroms.

    Returns:
        List of MetalSite instances.
    """
    if len(atom_coords) == 0 or len(metal_indices) == 0:
        return []

    metal_set = set(metal_indices)
    non_metal_indices = [i for i in range(len(atom_coords)) if i not in metal_set]

    if not non_metal_indices:
        return []

    non_metal_coords = atom_coords[non_metal_indices]

    sites = []
    for metal_idx in metal_indices:
        metal_coord = atom_coords[metal_idx]
        metal_meta = atom_metadata[metal_idx]
        metal_element = metal_meta.get('element', metal_meta.get('atom_name', 'UNK')).upper().strip()

        # A structure holds a handful of metals against a few thousand other
        # atoms, so one pass over the distances beats building an index.
        offsets = non_metal_coords - metal_coord
        squared = np.einsum("ij,ij->i", offsets, offsets)
        neighbor_positions = np.flatnonzero(squared <= distance_cutoff * distance_cutoff)
        preferred_donors = METAL_PREFERRED_DONORS.get(metal_element, {'N', 'O', 'S'})

        coordinating_atoms = []
        for pos in neighbor_positions:
            actual_idx = non_metal_indices[pos]
            neighbor_meta = atom_metadata[actual_idx]
            neighbor_element = neighbor_meta.get('element', '').upper().strip()

            # Accept atoms with preferred donor elements, or with no element info
            if neighbor_element and neighbor_element not in preferred_donors:
                continue

            dist = float(np.linalg.norm(atom_coords[actual_idx] - metal_coord))
            coordinating_atoms.append({
                'atom_name': neighbor_meta.get('atom_name', ''),
                'res_name': neighbor_meta.get('res_name', ''),
                'chain_id': neighbor_meta.get('chain_id', ''),
                'coords': atom_coords[actual_idx].copy(),
                'distance': dist,
                'atom_index': actual_idx,
            })

        coordination_number = len(coordinating_atoms)
        if coordination_number == 0:
            continue

        donor_coords = np.array([a['coords'] for a in coordinating_atoms])
        geometry = classify_coordination_geometry(donor_coords, metal_coord)

        sites.append(MetalSite(
            metal_element=metal_element,
            metal_coords=metal_coord.copy(),
            coordinating_atoms=coordinating_atoms,
            coordination_number=coordination_number,
            geometry=geometry,
        ))

    return sites


def classify_coordination_geometry(
    donor_coords: np.ndarray,
    metal_coord: np.ndarray,
) -> str:
    """Classify coordination geometry from donor positions.

    Args:
        donor_coords: (N, 3) array of donor atom coordinates.
        metal_coord: (3,) metal center coordinate.

    Returns:
        Geometry string from COMMON_COORDINATION_GEOMETRIES.
    """
    n = len(donor_coords)

    if n <= 1:
        return "unknown"
    if n == 2:
        return "linear"
    if n == 3:
        return "trigonal_planar"

    # Compute bond vectors from metal to donors
    vectors = donor_coords - metal_coord[np.newaxis, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    unit_vectors = vectors / norms

    if n == 4:
        return _classify_4coord(unit_vectors)
    if n == 5:
        return _classify_5coord(unit_vectors)
    if n == 6:
        return "octahedral"

    return "unknown"


def _all_angles(unit_vectors: np.ndarray) -> np.ndarray:
    """Compute all pairwise angles (degrees) between unit bond vectors."""
    n = len(unit_vectors)
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_a = np.clip(np.dot(unit_vectors[i], unit_vectors[j]), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_a)))
    return np.array(angles)


def _classify_4coord(unit_vectors: np.ndarray) -> str:
    """Tetrahedral (~109.5°) vs square_planar (~90° and ~180°).

    Square planar has two pairs of trans (near-180°) angles;
    tetrahedral has no near-linear angles.
    """
    angles = _all_angles(unit_vectors)
    n_linear = np.sum(angles > SQUARE_PLANAR_LINEAR_ANGLE)
    if n_linear >= 1:
        return "square_planar"
    return "tetrahedral"


def _classify_5coord(unit_vectors: np.ndarray) -> str:
    """Trigonal bipyramidal vs square pyramidal."""
    angles = _all_angles(unit_vectors)
    # Trigonal bipyramidal has two ~180° angles (axial-axial)
    n_linear = np.sum(angles > TRIGONAL_BIPYRAMIDAL_LINEAR_ANGLE)
    if n_linear >= 1:
        return "trigonal_bipyramidal"
    return "square_pyramidal"


def encode_metal_features(
    metal_sites: List[MetalSite],
    n_residues: int,
) -> Dict[str, np.ndarray]:
    """Encode metal coordination sites as feature arrays.

    Args:
        metal_sites: List of detected MetalSite objects.
        n_residues: Number of residues (used for per-residue arrays).

    Returns:
        Dict with:
          - metal_type_onehot: (M, len(METAL_PREFERRED_DONORS)) one-hot metal type
          - coordination_number: (M,) int coordination numbers
          - geometry_encoding: (M, len(COMMON_COORDINATION_GEOMETRIES)) one-hot geometry
          - distance_stats: (M, 3) [min_dist, max_dist, mean_dist] per site
          - n_metal_sites: scalar int
    """
    metal_types = list(METAL_PREFERRED_DONORS.keys())
    n_metal_types = len(metal_types)
    metal_type_idx = {m: i for i, m in enumerate(metal_types)}

    n_geoms = len(COMMON_COORDINATION_GEOMETRIES)
    geom_idx = {g: i for i, g in enumerate(COMMON_COORDINATION_GEOMETRIES)}

    m = len(metal_sites)
    if m == 0:
        return {
            'metal_type_onehot': np.zeros((0, n_metal_types), dtype=np.float32),
            'coordination_number': np.zeros((0,), dtype=np.int32),
            'geometry_encoding': np.zeros((0, n_geoms), dtype=np.float32),
            'distance_stats': np.zeros((0, 3), dtype=np.float32),
            'n_metal_sites': np.int32(0),
        }

    metal_type_onehot = np.zeros((m, n_metal_types), dtype=np.float32)
    coordination_number = np.zeros((m,), dtype=np.int32)
    geometry_encoding = np.zeros((m, n_geoms), dtype=np.float32)
    distance_stats = np.zeros((m, 3), dtype=np.float32)

    for i, site in enumerate(metal_sites):
        elem = site.metal_element.upper()
        if elem in metal_type_idx:
            metal_type_onehot[i, metal_type_idx[elem]] = 1.0

        coordination_number[i] = site.coordination_number

        geom = site.geometry
        if geom in geom_idx:
            geometry_encoding[i, geom_idx[geom]] = 1.0

        if site.coordinating_atoms:
            dists = np.array([a['distance'] for a in site.coordinating_atoms], dtype=np.float32)
            distance_stats[i, 0] = dists.min()
            distance_stats[i, 1] = dists.max()
            distance_stats[i, 2] = dists.mean()

    return {
        'metal_type_onehot': metal_type_onehot,
        'coordination_number': coordination_number,
        'geometry_encoding': geometry_encoding,
        'distance_stats': distance_stats,
        'n_metal_sites': np.int32(m),
    }
