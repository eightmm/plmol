"""Surface point generation via Shrake-Slee sphere sampling."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from ..constants import (
    SURFACE_DEFAULT_POINTS_PER_ATOM,
    SURFACE_DEFAULT_PROBE_RADIUS,
    SURFACE_MAX_POINTS_RATIO,
    SURFACE_MIN_POINTS_PER_ATOM,
)
from ..spatial import sphere_point_exposure


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
        2. Points covered by a neighbouring atom's expanded sphere are dropped
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

    # Points are laid out atom by atom, which is the layout the occlusion test
    # expects, so the exposure mask indexes straight into these arrays.
    all_points_list = []
    all_normals_list = []
    for i in range(n_atoms):
        unit_sphere = _fibonacci_sphere(int(per_atom_counts[i]))
        all_points_list.append(expanded_radii[i] * unit_sphere + positions[i])
        all_normals_list.append(unit_sphere)

    all_points_flat = np.concatenate(all_points_list, axis=0)
    all_normals_flat = np.concatenate(all_normals_list, axis=0)

    exposed = sphere_point_exposure(
        positions, expanded_radii, per_atom_counts, _fibonacci_sphere
    )

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
