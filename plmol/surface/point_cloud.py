"""Surface point generation via Shrake-Slee sphere sampling."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..constants import (
    SURFACE_BURIAL_KNN,
    SURFACE_DEFAULT_POINTS_PER_ATOM,
    SURFACE_DEFAULT_PROBE_RADIUS,
    SURFACE_MAX_POINTS_RATIO,
    SURFACE_MIN_POINTS_PER_ATOM,
)


@lru_cache(maxsize=None)
def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately uniform points on a unit sphere (Fibonacci lattice).

    Cached and returned read-only: every atom of a given point count shares the
    same lattice, so this is called once per distinct count rather than per atom.

    Returns:
        (n, 3) read-only array of unit vectors.
    """
    indices = np.arange(n, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / golden
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    sphere = np.column_stack([x, y, z]).astype(np.float32)
    sphere.setflags(write=False)
    return sphere


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

    # Allocate points proportional to surface area (4πr² ∝ r²)
    areas = expanded_radii ** 2
    area_mean = areas.mean()
    raw_counts = np.round(n_points_per_atom * areas / area_mean).astype(np.int32)
    min_pts = SURFACE_MIN_POINTS_PER_ATOM
    max_pts = n_points_per_atom * SURFACE_MAX_POINTS_RATIO
    per_atom_counts = np.clip(raw_counts, min_pts, max_pts)

    # Generate per-atom Fibonacci spheres with varying n_points (Option A)
    all_points_list = []
    all_normals_list = []
    atom_ids_list = []

    for i in range(n_atoms):
        n_pts = int(per_atom_counts[i])
        unit_sphere = _fibonacci_sphere(n_pts)
        pts = expanded_radii[i] * unit_sphere + positions[i]
        all_points_list.append(pts)
        all_normals_list.append(unit_sphere)
        atom_ids_list.append(np.full(n_pts, i, dtype=np.int32))

    all_points_flat = np.concatenate(all_points_list, axis=0)
    all_normals_flat = np.concatenate(all_normals_list, axis=0)
    atom_ids = np.concatenate(atom_ids_list, axis=0)

    # Build tree of atom centres for neighbour lookup
    tree = cKDTree(positions)

    # For each point, check if it is buried inside any *other* atom's expanded sphere
    K = min(SURFACE_BURIAL_KNN, n_atoms)
    dists, idx = tree.query(all_points_flat, k=K, workers=-1)
    if K == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    is_owner = idx == atom_ids[:, None]  # (P, K) bool
    is_inside = dists < expanded_radii[idx]  # (P, K) bool
    buried = (is_inside & ~is_owner).any(axis=1)  # (P,) bool
    exposed = ~buried

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
