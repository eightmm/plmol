"""Neighbour search for the geometry paths, with or without scipy.

Every geometric feature plmol extracts -- SASA, the surface point cloud, its
curvature, the atom-to-point mapping -- reduces to one of two neighbour
questions, and both were answered by ``scipy.spatial.cKDTree``. This module
answers them in numpy so scipy becomes optional rather than required.

Two different things live here, and they are optional for different reasons:

**Sphere occlusion** (:func:`sphere_point_exposure`) has no backend switch. The
tree formulation asked, for each of a protein's ~300k sampled sphere points,
which atoms are nearby -- and capped the answer at *k* neighbours, so it was an
approximation that could silently under-count. The formulation here inverts it:
find the atom *pairs* whose spheres can overlap at all, then test one atom's
sample points against one neighbour with a single matrix product. That is exact
-- no *k* to saturate -- and measurably faster, so it simply replaces the tree.

**k nearest neighbours** (:func:`knn`) does have a backend switch. A KD-tree is
the right structure for this and scipy's is threaded C; the uniform grid here
is slower on a point cloud. ``auto`` therefore prefers scipy when it is
installed and falls back to the grid when it is not.

    from plmol import set_spatial_backend
    set_spatial_backend("native")   # or "scipy", or "auto" (the default)
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np

from .errors import DependencyError, InputError

#: Selectable backends for :func:`knn`. ``"auto"`` prefers scipy.
SPATIAL_BACKENDS = ("auto", "scipy", "native")

#: Candidate pairs are generated in blocks of this many query points, so the
#: intermediate index arrays stay a few MB whatever the structure's size.
_QUERY_BLOCK = 1 << 14

#: Atom pairs are occlusion-tested in blocks of this many, for the same reason.
_PAIR_BLOCK = 1 << 16

#: Queries sampled to size the k-nearest-neighbour grid.
_CELL_SIZE_SAMPLE = 128

_BACKEND = "auto"


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def set_spatial_backend(name: str) -> None:
    """Choose which implementation :func:`knn` uses.

    Args:
        name: One of :data:`SPATIAL_BACKENDS`.

    Raises:
        InputError: If *name* is not a known backend.
    """
    global _BACKEND
    if name not in SPATIAL_BACKENDS:
        raise InputError(
            f"Unknown spatial backend {name!r}. Choose from {SPATIAL_BACKENDS}."
        )
    _BACKEND = name


def get_spatial_backend() -> str:
    """The configured backend, which may still be ``"auto"``."""
    return _BACKEND


def resolve_spatial_backend() -> str:
    """The backend that would actually run: ``"scipy"`` or ``"native"``.

    Raises:
        DependencyError: If scipy was asked for by name and is not installed.
    """
    if _BACKEND == "native":
        return "native"
    if _BACKEND == "scipy":
        if _import_ckdtree() is None:
            raise DependencyError(
                "The scipy spatial backend was requested but scipy is not "
                "installed. Install it with `pip install plmol[spatial]`, or "
                "call set_spatial_backend('native')."
            )
        return "scipy"
    return "scipy" if _import_ckdtree() is not None else "native"


def _import_ckdtree():
    """``scipy.spatial.cKDTree``, or None when scipy is not installed."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    return cKDTree


# ---------------------------------------------------------------------------
# Uniform grid
# ---------------------------------------------------------------------------


class _Grid:
    """Points bucketed into a uniform grid of cubic cells.

    Cells are addressed by their integer coordinate, flattened row-major.
    ``order`` lists point indices sorted by cell, and ``start`` marks where each
    cell's run begins, so a cell's members are a contiguous slice of ``order``.
    """

    __slots__ = ("origin", "dims", "strides", "inv", "order", "start", "count", "size")

    def __init__(self, points: np.ndarray, cell_size: float):
        self.size = float(cell_size)
        self.origin = points.min(axis=0)
        self.inv = np.float64(1.0 / cell_size)
        cell = self._cell_of(points)
        self.dims = cell.max(axis=0) + 1
        self.strides = np.array(
            [self.dims[1] * self.dims[2], self.dims[2], 1], dtype=np.int64
        )
        flat = cell @ self.strides
        self.order = np.argsort(flat, kind="stable")
        n_cells = int(self.dims.prod())
        self.count = np.bincount(flat, minlength=n_cells)
        self.start = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(self.count, out=self.start[1:])

    def _cell_of(self, points: np.ndarray) -> np.ndarray:
        """Integer cell coordinate of each point. May be out of range."""
        return np.floor((points - self.origin) * self.inv).astype(np.int64)

    def candidates(self, queries: np.ndarray, ring: int) -> Tuple[np.ndarray, np.ndarray]:
        """Points in the cells within Chebyshev distance *ring* of each query.

        Returns ``(row, index)`` pairs: ``row`` indexes *queries*, ``index``
        indexes the points the grid was built from. Every point closer than
        ``ring * cell_size`` is guaranteed to appear: two coordinates less than
        ``ring * cell_size`` apart divide by the cell size into values less than
        ``ring`` apart, so their cell indices differ by at most ``ring``.
        """
        cell = self._cell_of(queries)
        offsets = _ring_offsets(ring)
        neighbour_cells = cell[:, None, :] + offsets[None]
        inside = ((neighbour_cells >= 0) & (neighbour_cells < self.dims)).all(-1)
        flat = np.where(inside, neighbour_cells @ self.strides, 0)
        count = np.where(inside, self.count[flat], 0).ravel()
        total = int(count.sum())
        if total == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        # Expand the per-cell runs into one flat list: each cell contributes
        # order[start : start + count], and the ramp walks that offset.
        run_start = np.zeros(count.size, dtype=np.int64)
        np.cumsum(count[:-1], out=run_start[1:])
        ramp = np.arange(total, dtype=np.int64)
        ramp -= np.repeat(run_start, count)
        index = self.order[np.repeat(self.start[flat].ravel(), count) + ramp]
        row = np.repeat(
            np.arange(len(queries), dtype=np.int64),
            count.reshape(len(queries), -1).sum(1),
        )
        return row, index


@lru_cache(maxsize=8)
def _ring_offsets(ring: int) -> np.ndarray:
    """The ``(2*ring + 1)**3`` cell offsets forming a cube around the centre."""
    span = np.arange(-ring, ring + 1)
    return np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)


def _as_points(array: np.ndarray, name: str) -> np.ndarray:
    points = np.ascontiguousarray(array, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise InputError(f"{name} must be (N, 3), got {points.shape}.")
    return points


# ---------------------------------------------------------------------------
# Overlapping-sphere pairs
# ---------------------------------------------------------------------------


def _grid_pairs(coords: np.ndarray, cell_size: float, keep):
    """Ordered index pairs a *keep* rule accepts, from one pass over the grid.

    *keep* is called with ``(i, j, delta, squared_distance)`` for each candidate
    and returns a boolean mask. Candidates come from one ring of cells, so
    *cell_size* has to be at least the largest distance the rule can accept.
    """
    n = len(coords)
    grid = _Grid(coords, max(float(cell_size), 1e-6))
    parts = []
    for start in range(0, n, _QUERY_BLOCK):
        stop = min(start + _QUERY_BLOCK, n)
        row, index = grid.candidates(coords[start:stop], ring=1)
        if row.size == 0:
            continue
        row += start
        delta = coords[index] - coords[row]
        squared = np.einsum("ij,ij->i", delta, delta)
        mask = keep(row, index, delta, squared)
        mask &= row != index
        parts.append((row[mask], index[mask], delta[mask], squared[mask]))
    if not parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    return tuple(np.concatenate(column) for column in zip(*parts))


def overlapping_sphere_pairs(
    coords: np.ndarray, radii: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ordered index pairs whose spheres intersect.

    Args:
        coords: Sphere centres ``(N, 3)``.
        radii: Sphere radii ``(N,)``, probe already included if wanted.

    Returns:
        ``(i, j, delta, squared_distance)`` for every ordered pair with
        ``|c_i - c_j| < r_i + r_j`` and ``i != j``, where ``delta`` is
        ``c_j - c_i``. ``i`` is sorted ascending, so a pair's owner runs are
        contiguous.
    """
    coords = _as_points(coords, "coords")
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    n = len(coords)
    if n != len(radii):
        raise InputError(f"coords has {n} rows but radii has {len(radii)}.")
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros((0, 3), np.float32), np.zeros(0, np.float32)

    def touching(row, index, delta, squared):
        limit = radii[row] + radii[index]
        return squared < limit * limit

    # Two spheres can only touch within the largest possible sum of radii, so
    # that is the cell size: one ring of cells then covers every candidate.
    return _grid_pairs(coords, 2.0 * radii.max(), touching)


def pairs_within(
    coords: np.ndarray, cutoff: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordered index pairs closer than *cutoff*, and how far apart they are.

    The alternative is a full ``(N, N)`` distance matrix, which for a protein's
    atoms computes ten million distances to keep the seventy thousand under a
    few Angstrom. This visits only the cells that can contain one.

    Args:
        coords: Positions ``(N, 3)``.
        cutoff: Maximum distance, exclusive.

    Returns:
        ``(i, j, distance)`` for every ordered pair with ``|c_i - c_j| < cutoff``
        and ``i != j``, in row-major order -- the order a dense mask would give,
        so the pairs can stand in for one.
    """
    coords = _as_points(coords, "coords")
    if cutoff <= 0:
        raise InputError(f"cutoff must be positive, got {cutoff}.")
    if len(coords) < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros(0, np.float32)

    limit = float(cutoff) ** 2

    def near(row, index, delta, squared):
        return squared < limit

    row, index, _, squared = _grid_pairs(coords, cutoff, near)
    # The grid emits a row's neighbours in cell order; a dense mask would emit
    # them by column. Downstream edge features index by position, so sort.
    order = np.lexsort((index, row))
    return row[order], index[order], np.sqrt(squared[order]).astype(np.float32)


# ---------------------------------------------------------------------------
# Sphere point occlusion
# ---------------------------------------------------------------------------


def sphere_point_exposure(
    coords: np.ndarray,
    radii: np.ndarray,
    counts: Sequence[int],
    sphere_of,
) -> np.ndarray:
    """Which sampled sphere points lie outside every *other* sphere.

    This is the shared core of solvent accessible surface area and of the
    surface point cloud: both sample each atom's expanded sphere and keep the
    points no neighbouring atom covers.

    A point on atom ``i`` in direction ``s`` is covered by atom ``j`` when
    ``|c_i + r_i s - c_j| < r_j``. Writing ``d = c_j - c_i`` and expanding, that
    is ``s . d > (|d|^2 + r_i^2 - r_j^2) / (2 r_i)`` -- one dot product per
    (direction, pair). Every direction of an atom against every neighbour is
    therefore a single matrix product, and only pairs of spheres that actually
    intersect take part.

    Args:
        coords: Atom positions ``(N, 3)``.
        radii: Sphere radii ``(N,)``, probe already included if wanted.
        counts: Sample points per atom ``(N,)``. May vary between atoms.
        sphere_of: ``count -> (count, 3)`` unit direction table. Called once per
            distinct count.

    Returns:
        ``(sum(counts),)`` bool. Points are laid out atom by atom in order, so
        atom ``i`` owns the slice starting at ``cumsum(counts)[i] - counts[i]``.
    """
    coords = _as_points(coords, "coords")
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    n = len(coords)
    if not (n == len(radii) == len(counts)):
        raise InputError(
            f"coords, radii and counts must agree in length, got "
            f"{n}, {len(radii)}, {len(counts)}."
        )

    offset = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offset[1:])
    total_points = int(offset[-1])
    occluded = np.zeros(total_points, dtype=bool)
    if n < 2 or total_points == 0:
        return ~occluded

    owner, neighbour, delta, squared = overlapping_sphere_pairs(coords, radii)
    if owner.size == 0:
        return ~occluded

    # The dot products are a matrix product against a fixed direction table, so
    # pairs are grouped by their owner's sample count. Sorting is stable, which
    # keeps each group's owners ascending and its runs contiguous.
    group = counts[owner]
    if group.min() != group.max():
        regroup = np.argsort(group, kind="stable")
        owner, neighbour = owner[regroup], neighbour[regroup]
        delta, squared, group = delta[regroup], squared[regroup], group[regroup]
    bounds = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    bounds = np.r_[bounds, len(group)]

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        count = int(group[lo])
        if count == 0:
            continue
        directions = np.ascontiguousarray(sphere_of(count), dtype=np.float32)
        for start in range(lo, hi, _PAIR_BLOCK):
            stop = min(start + _PAIR_BLOCK, hi)
            own = owner[start:stop]
            radius = radii[own]
            threshold = (
                squared[start:stop] + radius * radius - radii[neighbour[start:stop]] ** 2
            ) / (2.0 * radius)
            covered = (directions @ delta[start:stop].T) > threshold[None, :]
            # One column per pair; fold each owner's pairs together, then OR the
            # result into that atom's slice of the output.
            runs = np.flatnonzero(np.r_[True, own[1:] != own[:-1]])
            folded = np.logical_or.reduceat(covered, runs, axis=1)
            atoms = own[runs]
            slots = offset[atoms][None, :] + np.arange(count, dtype=np.int64)[:, None]
            np.logical_or.at(occluded, slots, folded)

    return ~occluded


# ---------------------------------------------------------------------------
# k nearest neighbours
# ---------------------------------------------------------------------------


class NeighbourIndex:
    """A reusable k-nearest-neighbour index over a fixed point set.

    Build it once and query it repeatedly -- the surface curvature pass queries
    the same cloud block by block from several threads. Queries are thread safe;
    the underlying structure is built once, on the first query that needs it.

    Args:
        points: The searchable points ``(M, 3)``.
        backend: Override the configured backend for this index.
    """

    def __init__(self, points: np.ndarray, backend: Optional[str] = None):
        self.points = _as_points(points, "points")
        self.backend = backend or resolve_spatial_backend()
        self._lock = threading.Lock()
        self._tree = None
        self._grids: dict = {}

    def query(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """The *k* nearest points to each query, nearest first. See :func:`knn`."""
        queries = _as_points(queries, "queries")
        k = min(int(k), len(self.points))
        if k < 1:
            raise InputError(f"k must be at least 1 and points non-empty, got k={k}.")
        if self.backend == "scipy":
            distances, indices = self._scipy_tree().query(queries, k=k, workers=-1)
            if k == 1:
                distances, indices = distances[:, None], indices[:, None]
            return distances, indices
        return _knn_native(self.points, queries, k, grid=self._grid_for(queries, k))

    def _scipy_tree(self):
        with self._lock:
            if self._tree is None:
                self._tree = _import_ckdtree()(self.points)
        return self._tree

    def _grid_for(self, queries: np.ndarray, k: int) -> "_Grid":
        """The grid for this *k*, built once and kept.

        The cell size is read off the queries, so a first call with only a
        handful of them would size the grid from too little and every later
        query would pay for it. Below the sample size the points themselves
        stand in, which for the usual self-query is the same thing.
        """
        with self._lock:
            grid = self._grids.get(k)
            if grid is None:
                sample = queries if len(queries) >= _CELL_SIZE_SAMPLE else self.points
                grid = _Grid(self.points, _knn_cell_size(self.points, sample, k))
                self._grids[k] = grid
        return grid


def knn(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    backend: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """The *k* nearest points of *data* to each of *queries*, nearest first.

    Args:
        data: Points to search ``(M, 3)``.
        queries: Query positions ``(Q, 3)``.
        k: Neighbours per query. Clamped to ``M``.
        backend: Override the configured backend for this call.

    Returns:
        ``(distances, indices)``, both ``(Q, k)``, sorted by distance. Queries
        coinciding with a data point get that point as their first neighbour --
        nothing is excluded.
    """
    data = _as_points(data, "data")
    queries = _as_points(queries, "queries")
    k = min(int(k), len(data))
    if k < 1:
        raise InputError(f"k must be at least 1 and data non-empty, got k={k}.")

    return NeighbourIndex(data, backend=backend).query(queries, k)


def _knn_native(data: np.ndarray, queries: np.ndarray, k: int, grid=None):
    """Grid-backed k nearest neighbours.

    Queries are grouped by the cell they land in, so every query in a cell
    shares one candidate list and the distances become a dense block instead of
    a ragged gather. A ring of cells around a query only guarantees the points
    closer than ``ring * cell_size``; a group whose k-th neighbour lies at or
    beyond that is redone with a wider ring, which keeps the result exact rather
    than approximate.
    """
    if grid is None:
        grid = _Grid(data, _knn_cell_size(data, queries, k))
    cell = grid._cell_of(queries)
    # Queries may sit outside the data's own box, so group them by a key built
    # from their own extent. The grid's strides would alias negative cells.
    corner = cell.min(axis=0)
    span = cell.max(axis=0) - corner + 1
    local = cell - corner
    key = (local[:, 0] * span[1] + local[:, 1]) * span[2] + local[:, 2]
    group = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.r_[True, key[group][1:] != key[group][:-1]])
    bounds = np.r_[bounds, len(group)]

    distances = np.empty((len(queries), k), dtype=np.float64)
    indices = np.empty((len(queries), k), dtype=np.int64)

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        members = group[lo:hi]
        here = cell[members[0]]
        ring = 1
        while True:
            candidates = _cell_block(grid, here, ring)
            if len(candidates) >= k:
                near_d, near_i = _dense_topk(queries[members], data, candidates, k)
                if near_d[:, -1].max() < ring * grid.size:
                    break
            if len(candidates) >= len(data):
                near_d, near_i = _dense_topk(queries[members], data, candidates, k)
                break
            ring += 1
        distances[members] = near_d
        indices[members] = near_i

    return distances, indices


def _cell_block(grid: "_Grid", centre: np.ndarray, ring: int) -> np.ndarray:
    """Indices of every point in the cells within *ring* of *centre*."""
    cells = centre[None] + _ring_offsets(ring)
    inside = ((cells >= 0) & (cells < grid.dims)).all(-1)
    flat = (cells[inside] @ grid.strides)
    count = grid.count[flat]
    flat = flat[count > 0]
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64)
    count = grid.count[flat]
    total = int(count.sum())
    run_start = np.zeros(count.size, dtype=np.int64)
    np.cumsum(count[:-1], out=run_start[1:])
    ramp = np.arange(total, dtype=np.int64)
    ramp -= np.repeat(run_start, count)
    return grid.order[np.repeat(grid.start[flat], count) + ramp]


def _dense_topk(queries, data, candidates, k):
    """Nearest *k* candidates for each query, sorted by distance."""
    # Both sides are shifted to the block's own origin before the matrix
    # product. The expanded form |a|^2 + |b|^2 - 2ab cancels catastrophically in
    # float32 on raw crystallographic coordinates; on centred ones it does not.
    origin = queries[0]
    points = (data[candidates] - origin).astype(np.float64)
    local = (queries - origin).astype(np.float64)
    squared = (
        np.einsum("ij,ij->i", points, points)[None, :]
        + np.einsum("ij,ij->i", local, local)[:, None]
        - 2.0 * (local @ points.T)
    )
    picked = np.argpartition(squared, k - 1, axis=1)[:, :k]
    rows = np.arange(len(queries))[:, None]
    chosen = squared[rows, picked]
    picked = picked[rows, np.argsort(chosen, axis=1, kind="stable")]
    return np.sqrt(np.maximum(squared[rows, picked], 0.0)), candidates[picked]


def _knn_cell_size(data: np.ndarray, queries: np.ndarray, k: int) -> float:
    """Cell size about as wide as the k-th neighbour is far.

    Estimated by brute force from a small sample of queries rather than from the
    bounding box, because the point clouds this runs on sit on a surface: a
    density read off the enclosing volume is wrong by an order of magnitude.
    The estimate is used as the cell size directly: one ring of cells then
    covers everything closer than that distance, which is what makes a typical
    query exact without widening. Scaling it by 0.7 to 1.3 was measured on a
    15k-point cloud and none of those beat leaving it alone.

    The bounding box still sets a floor, for the degenerate cases where the
    sampled distance comes out at zero.
    """
    extent = data.max(axis=0) - data.min(axis=0)
    volume = float(np.prod(np.maximum(extent, 1e-3)))
    # One point per cell on average, as a floor. Duplicate points and a query
    # set that includes its own data both make the k-th distance zero, and
    # without this the grid would then be asked for billions of empty cells.
    floor = (volume / max(len(data), 1)) ** (1.0 / 3.0)

    sample = queries
    if len(queries) > _CELL_SIZE_SAMPLE:
        step = len(queries) // _CELL_SIZE_SAMPLE
        sample = queries[::step][:_CELL_SIZE_SAMPLE]
    delta = data[None, :, :] - sample[:, None, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    kth = np.partition(squared, k - 1, axis=1)[:, k - 1]
    typical = float(np.sqrt(np.percentile(kth, 90)))
    return max(typical, floor, 1e-3)
