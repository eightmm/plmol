"""Finding the cavities in a structure, without being told where to look.

``extract_pocket`` answers "which residues line this ligand". This answers the
question you have before there is a ligand: where are the enclosed spaces at
all. Apo structures, docking boxes, and comparing sites across a family all
start here.

The method is LIGSITE's. Put the structure on a grid and mark every point that
falls inside an atom. From each remaining point, look along seven axes -- the
three grid axes and the four body diagonals. An axis is *enclosed* when there
is protein in both directions along it, which means the point sits between two
walls rather than out in bulk solvent. Points enclosed on enough axes are
cavity; adjacent cavity points are one cavity.

Nothing here needs a dependency plmol does not already have. The alternative
family of methods -- fpocket's alpha spheres -- wants a Delaunay triangulation,
and that would put scipy back in the hard requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .arrays import FLOAT, INT
from .constants import (
    CAVITY_GRID_RESOLUTION,
    CAVITY_LINING_MARGIN,
    CAVITY_MIN_POINTS,
    CAVITY_PADDING,
    CAVITY_PROBE_RADIUS,
    CAVITY_PSP_THRESHOLD,
    CAVITY_SCAN_LENGTH,
    DEFAULT_VDW_RADIUS,
    ELEMENT_SYMBOL_TO_ATOMIC_NUMBER,
    VDW_RADIUS,
)
from .errors import InputError

#: The seven axes a point is tested along: three grid axes and four body
#: diagonals. Only one direction of each is listed; both are scanned.
SCAN_AXES = np.array(
    [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    ],
    dtype=np.int64,
)

#: The 13 forward steps that reach all 26 neighbours of a grid point when each
#: is also taken backwards. Used to join adjacent cavity points.
_FORWARD_NEIGHBOURS = np.array(
    [offset for offset in
     (tuple(index - 1 for index in cell) for cell in np.ndindex(3, 3, 3))
     if offset > (0, 0, 0)],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Cavity:
    """One enclosed space, and what lines it.

    Attributes:
        center: Centroid of the cavity's grid points ``(3,)``.
        volume: Cubic Angstrom, the point count times the grid cell volume.
        points: The grid points themselves ``(M, 3)``, in Angstrom.
        buriedness: Mean enclosed-axis count over the points, 0 to 7. A deep
            pocket sits near 7; a groove near the threshold.
        lining_atom_indices: Atoms within reach of any cavity point.
        lining_residues: ``(chain, residue number, residue name)`` for those
            atoms, in structure order. Empty when no residue labels were given.
        extent: Side lengths of the cavity's bounding box ``(3,)``.
    """

    center: np.ndarray
    volume: float
    points: np.ndarray
    buriedness: float
    lining_atom_indices: np.ndarray
    lining_residues: List[Tuple[str, int, str]]
    extent: np.ndarray

    @property
    def num_points(self) -> int:
        return len(self.points)


def element_vdw_radii(elements: Sequence[Optional[str]]) -> np.ndarray:
    """Van der Waals radius per atom from element symbols."""
    return np.array(
        [
            VDW_RADIUS.get(
                ELEMENT_SYMBOL_TO_ATOMIC_NUMBER.get((element or "").upper(), 0),
                DEFAULT_VDW_RADIUS,
            )
            for element in elements
        ],
        dtype=FLOAT,
    )


def detect_cavities(
    coords: np.ndarray,
    radii: np.ndarray,
    *,
    resolution: float = CAVITY_GRID_RESOLUTION,
    probe_radius: float = CAVITY_PROBE_RADIUS,
    psp_threshold: int = CAVITY_PSP_THRESHOLD,
    scan_length: float = CAVITY_SCAN_LENGTH,
    min_points: int = CAVITY_MIN_POINTS,
    padding: float = CAVITY_PADDING,
    lining_margin: float = CAVITY_LINING_MARGIN,
    residues: Optional[Sequence[Tuple[str, int, str]]] = None,
) -> List[Cavity]:
    """Every cavity in a structure, largest first.

    Args:
        coords: Atom positions ``(N, 3)``.
        radii: Van der Waals radii ``(N,)``, without the probe.
        resolution: Grid spacing in Angstrom. Finer resolves smaller pockets
            and costs the cube of the ratio.
        probe_radius: A grid point within ``vdw + probe`` of an atom is inside
            the structure, not in a cavity.
        psp_threshold: How many of the seven axes must be enclosed. 7 finds
            only fully buried voids; the default 5 finds pockets; 3 also
            catches open grooves.
        scan_length: How far along an axis to look for the far wall, in
            Angstrom. Beyond this the point counts as open in that direction.
        min_points: Clusters smaller than this are dropped as noise.
        padding: Empty grid kept around the structure's bounding box.
        lining_margin: Extra distance past ``vdw + probe`` for an atom to
            count as lining a cavity.
        residues: Optional ``(chain, number, name)`` per atom, used to report
            which residues line each cavity.

    Returns:
        Cavities sorted by volume, largest first.

    Raises:
        InputError: If the inputs disagree in length, or a parameter is
            outside its meaningful range.
    """
    coords = np.ascontiguousarray(coords, dtype=FLOAT)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise InputError(f"coords must be (N, 3), got {coords.shape}.")
    radii = np.asarray(radii, dtype=FLOAT).reshape(-1)
    if len(radii) != len(coords):
        raise InputError(f"coords has {len(coords)} rows but radii has {len(radii)}.")
    if resolution <= 0:
        raise InputError(f"resolution must be positive, got {resolution}.")
    if not 1 <= psp_threshold <= len(SCAN_AXES):
        raise InputError(
            f"psp_threshold must be between 1 and {len(SCAN_AXES)}, got {psp_threshold}."
        )
    if len(coords) == 0:
        return []
    if residues is not None and len(residues) != len(coords):
        raise InputError(
            f"residues has {len(residues)} entries for {len(coords)} atoms."
        )

    reach = radii + probe_radius
    origin, shape = _grid_bounds(coords, reach, resolution, padding)
    inside = _mark_inside(coords, reach, origin, shape, resolution)

    enclosed = _enclosed_axis_count(inside, resolution, scan_length)
    cavity_mask = (~inside) & (enclosed >= psp_threshold)
    if not cavity_mask.any():
        return []

    cells, labels, count = _connected_components(cavity_mask)
    # Grouped by label in one pass. Reading a component out with
    # `labels == label` swept the whole grid once per component, and a protein
    # has a hundred-odd of them, nearly all too small to survive min_points --
    # a hundred sweeps of a million cells to find a few thousand.
    #
    # The sort must be stable. Within a component the points then keep the C
    # order the cells were found in, which is the order the centre, the extent
    # and the reported point cloud were all built from.
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    cavities = []
    for label in range(count):
        indices = cells[order[bounds[label]:bounds[label + 1]]]
        if len(indices) < min_points:
            continue
        points = origin + indices.astype(FLOAT) * resolution
        cavities.append(
            _describe(points, indices, enclosed, coords, reach,
                      resolution, lining_margin, residues)
        )

    cavities.sort(key=lambda cavity: cavity.volume, reverse=True)
    return cavities


def _grid_bounds(coords, reach, resolution, padding):
    """Grid origin and shape covering the structure plus padding."""
    margin = float(reach.max()) + padding
    origin = (coords.min(axis=0) - margin).astype(FLOAT)
    extent = (coords.max(axis=0) + margin) - origin
    shape = tuple(max(3, int(np.ceil(size / resolution)) + 1) for size in extent)
    return origin, shape


def _mark_inside(coords, reach, origin, shape, resolution):
    """Boolean grid: True where a point falls inside an atom's expanded sphere.

    Each atom marks the block of grid points around it rather than every point
    being tested against every atom. The stencil is sized for the largest atom
    and the exact distance decides, so a small atom does not over-mark.
    """
    inside = np.zeros(shape, dtype=bool)
    span = int(np.ceil(float(reach.max()) / resolution))
    axis = np.arange(-span, span + 1, dtype=INT)
    stencil = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    offsets = stencil[(np.linalg.norm(stencil * resolution, axis=1) <= reach.max())]

    base = np.rint((coords - origin) / resolution).astype(INT)
    dims = np.array(shape, dtype=INT)
    for start in range(0, len(coords), _atom_block(len(offsets))):
        stop = min(start + _atom_block(len(offsets)), len(coords))
        cells = base[start:stop, None, :] + offsets[None]           # (block, S, 3)
        valid = ((cells >= 0) & (cells < dims)).all(axis=-1)
        position = origin + cells.astype(FLOAT) * resolution
        gap = position - coords[start:stop, None, :]
        squared = np.einsum("bsi,bsi->bs", gap, gap)
        limit = reach[start:stop, None] ** 2
        hit = valid & (squared <= limit)
        chosen = cells[hit]
        inside[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
    return inside


def _atom_block(stencil_size: int) -> int:
    """Atoms per block, so the (block, stencil, 3) array stays a few MB."""
    return max(1, (1 << 20) // max(stencil_size, 1))


def _enclosed_axis_count(inside, resolution, scan_length):
    """Per grid point, how many of the seven axes have structure on both sides.

    A ray is walked one step at a time and the wall it finds is recorded by
    shifting the whole occupancy grid, so the scan is seven shifts per step
    rather than one walk per point.
    """
    steps = max(1, int(round(scan_length / resolution)))
    count = np.zeros(inside.shape, dtype=np.int8)
    for axis in SCAN_AXES:
        forward = np.zeros(inside.shape, dtype=bool)
        backward = np.zeros(inside.shape, dtype=bool)
        for step in range(1, steps + 1):
            forward |= _shift(inside, axis * step)
            backward |= _shift(inside, -axis * step)
        count += (forward & backward).view(np.int8)
    return count


def _shift(grid: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """*grid* seen from a point *offset* cells away, padded with False.

    ``result[p]`` is ``grid[p + offset]``, so a True says there is structure
    that far along the direction.
    """
    out = np.zeros_like(grid)
    source = [slice(None)] * 3
    target = [slice(None)] * 3
    for axis, step in enumerate(offset):
        step = int(step)
        size = grid.shape[axis]
        if abs(step) >= size:
            return out
        if step > 0:
            source[axis] = slice(step, size)
            target[axis] = slice(0, size - step)
        elif step < 0:
            source[axis] = slice(0, size + step)
            target[axis] = slice(-step, size)
    out[tuple(target)] = grid[tuple(source)]
    return out


def _connected_components(mask: np.ndarray):
    """Label 26-connected groups of True cells.

    Returns the True cells in C order, a 0-based component label for each of
    them, and the number of components. Cells rather than a labelled grid:
    what the caller wants is each component's cells, and a grid would have to
    be searched to get them back.

    Union-find over the True cells only. A flood fill on the full grid would
    touch far more cells than a cavity ever occupies, and the labelling in
    scipy.ndimage is the dependency this module exists to avoid.
    """
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return cells, np.zeros(0, dtype=INT), 0

    dims = np.array(mask.shape, dtype=INT)
    flat = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    position = {int(key): i for i, key in enumerate(flat)}

    parent = np.arange(len(cells), dtype=INT)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:          # path compression
            parent[node], node = root, parent[node]
        return root

    for step in _FORWARD_NEIGHBOURS:
        shifted = cells + step
        legal = ((shifted >= 0) & (shifted < dims)).all(axis=1)
        keys = (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]
        for source, key in zip(np.flatnonzero(legal), keys[legal]):
            target = position.get(int(key))
            if target is not None:
                left, right = find(int(source)), find(target)
                if left != right:
                    parent[right] = left

    roots = np.array([find(i) for i in range(len(cells))], dtype=INT)
    _, labels = np.unique(roots, return_inverse=True)
    labels = labels.reshape(-1).astype(INT)
    return cells, labels, int(labels.max()) + 1


def _describe(points, indices, enclosed, coords, reach,
              resolution, lining_margin, residues):
    """Build the Cavity record for one cluster of grid points."""
    # spatial's cell grid, used privately: the query here is "atoms near these
    # points", which no public function in that module spells. If spatial ever
    # grows one, this should use it.
    from .spatial import _Grid

    # Accumulated in float64. The points are float32 and a cavity has
    # thousands of them, so a float32 sum drifts by tens of ULP -- and by a
    # different amount depending on how the array happens to be laid out in
    # memory, since that decides whether numpy sums pairwise or straight
    # through. The centroid is a reported number; it should not depend on that.
    center = points.mean(axis=0, dtype=np.float64).astype(FLOAT)
    buriedness = float(enclosed[indices[:, 0], indices[:, 1], indices[:, 2]].mean())

    # Lining atoms: any atom whose expanded sphere plus a margin reaches a
    # cavity point. The grid over the atoms keeps this local to the cavity.
    limit = float(reach.max()) + lining_margin
    grid = _Grid(coords, max(limit, 1e-6))
    row, index = grid.candidates(points, ring=1)
    if row.size:
        gap = coords[index] - points[row]
        squared = np.einsum("ij,ij->i", gap, gap)
        allowed = reach[index] + lining_margin
        lining = np.unique(index[squared <= allowed * allowed])
    else:
        lining = np.zeros(0, dtype=INT)

    lining_residues: List[Tuple[str, int, str]] = []
    if residues is not None and lining.size:
        seen = set()
        for atom in lining:
            residue = tuple(residues[int(atom)])
            if residue not in seen:
                seen.add(residue)
                lining_residues.append(residue)

    return Cavity(
        center=center,
        volume=float(len(points) * resolution ** 3),
        points=points.astype(FLOAT),
        buriedness=buriedness,
        lining_atom_indices=lining.astype(INT),
        lining_residues=lining_residues,
        extent=(points.max(axis=0) - points.min(axis=0)).astype(FLOAT),
    )
