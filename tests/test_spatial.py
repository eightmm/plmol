"""Tests for the native neighbour search that replaced scipy's cKDTree."""

import numpy as np
import pytest

from plmol import InputError
from plmol.spatial import (
    SPATIAL_BACKENDS,
    NeighbourIndex,
    get_spatial_backend,
    knn,
    overlapping_sphere_pairs,
    pairs_within,
    resolve_spatial_backend,
    set_spatial_backend,
    sphere_point_exposure,
)
from plmol.surface.point_cloud import _fibonacci_sphere


@pytest.fixture(autouse=True)
def restore_backend():
    yield
    set_spatial_backend("auto")


def brute_force_pairs(coords, radii):
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    limit = radii[:, None] + radii[None, :]
    touching = distance < limit
    np.fill_diagonal(touching, False)
    return set(map(tuple, np.argwhere(touching)))


def brute_force_exposure(coords, radii, counts):
    out = []
    for i, count in enumerate(counts):
        points = coords[i] + radii[i] * _fibonacci_sphere(int(count))
        covered = np.zeros(int(count), dtype=bool)
        for j in range(len(coords)):
            if j != i:
                covered |= np.linalg.norm(points - coords[j], axis=1) < radii[j]
        out.append(~covered)
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)


def brute_force_knn(data, queries, k):
    squared = ((data[None] - queries[:, None]) ** 2).sum(-1)
    order = np.argsort(squared, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(queries))[:, None]
    return np.sqrt(squared[rows, order]), order


# -- Overlapping sphere pairs -------------------------------------------------


class TestOverlappingSpherePairs:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        coords = rng.normal(scale=3.0, size=(120, 3)).astype(np.float32)
        radii = rng.uniform(1.0, 2.5, size=120).astype(np.float32)
        i, j, _, _ = overlapping_sphere_pairs(coords, radii)
        assert set(zip(i.tolist(), j.tolist())) == brute_force_pairs(coords, radii)

    def test_owner_runs_are_contiguous(self):
        rng = np.random.default_rng(3)
        coords = rng.normal(scale=3.0, size=(80, 3)).astype(np.float32)
        radii = np.full(80, 2.0, dtype=np.float32)
        i, _, _, _ = overlapping_sphere_pairs(coords, radii)
        assert np.all(np.diff(i) >= 0)

    def test_delta_and_distance_are_consistent(self):
        rng = np.random.default_rng(4)
        coords = rng.normal(scale=3.0, size=(60, 3)).astype(np.float32)
        radii = np.full(60, 2.0, dtype=np.float32)
        i, j, delta, squared = overlapping_sphere_pairs(coords, radii)
        assert np.allclose(delta, coords[j] - coords[i], atol=1e-5)
        assert np.allclose(squared, (delta ** 2).sum(-1), atol=1e-4)

    def test_far_apart_spheres_make_no_pairs(self):
        coords = np.array([[0.0, 0, 0], [100.0, 0, 0]], dtype=np.float32)
        i, _, _, _ = overlapping_sphere_pairs(coords, np.float32([1.0, 1.0]))
        assert i.size == 0

    def test_single_sphere(self):
        i, _, _, _ = overlapping_sphere_pairs(np.zeros((1, 3), np.float32), np.float32([2.0]))
        assert i.size == 0

    def test_bad_shape_is_rejected(self):
        with pytest.raises(InputError, match=r"\(N, 3\)"):
            overlapping_sphere_pairs(np.zeros((4, 2)), np.zeros(4))

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(InputError, match="radii"):
            overlapping_sphere_pairs(np.zeros((4, 3)), np.zeros(3))


# -- Pairs within a cutoff ----------------------------------------------------


class TestPairsWithin:
    """This stands in for a dense distance matrix, so it has to match one
    exactly -- same pairs, same order, same distances."""

    @pytest.mark.parametrize("cutoff", [1.0, 3.0, 8.0])
    def test_matches_a_dense_mask(self, cutoff):
        rng = np.random.default_rng(30)
        points = (rng.normal(size=(400, 3)) * 4).astype(np.float32)
        dense = np.linalg.norm(points[:, None] - points[None], axis=-1)
        expected_i, expected_j = np.nonzero((dense < cutoff) & (dense > 0))
        got_i, got_j, distances = pairs_within(points, cutoff)
        assert np.array_equal(got_i, expected_i)
        assert np.array_equal(got_j, expected_j)
        assert np.allclose(distances, dense[expected_i, expected_j], atol=1e-5)

    def test_the_order_is_row_major(self):
        rng = np.random.default_rng(31)
        points = (rng.normal(size=(60, 3)) * 3).astype(np.float32)
        i, j, _ = pairs_within(points, 5.0)
        assert np.array_equal(np.lexsort((j, i)), np.arange(len(i)))

    def test_no_pair_is_a_point_with_itself(self):
        points = np.zeros((20, 3), dtype=np.float32)
        i, j, _ = pairs_within(points, 1.0)
        assert not (i == j).any()
        assert len(i) == 20 * 19

    def test_far_apart_points_make_no_pairs(self):
        points = np.array([[0.0, 0, 0], [500.0, 0, 0]], dtype=np.float32)
        assert pairs_within(points, 1.0)[0].size == 0

    def test_a_single_point(self):
        assert pairs_within(np.zeros((1, 3), np.float32), 5.0)[0].size == 0

    def test_a_nonpositive_cutoff_is_rejected(self):
        with pytest.raises(InputError, match="cutoff must be positive"):
            pairs_within(np.zeros((4, 3), np.float32), 0.0)

    def test_awkward_geometries(self):
        """A rod fills one row of cells; two clusters leave most of them empty."""
        rng = np.random.default_rng(32)
        blob = rng.normal(size=(300, 3))
        for points in (
            rng.uniform([0, 0, 0], [400, 1, 1], size=(600, 3)),
            np.vstack([blob, blob + 300.0]),
            np.c_[rng.uniform(0, 30, size=(500, 2)), np.zeros(500)],
        ):
            points = points.astype(np.float32)
            dense = np.linalg.norm(points[:, None] - points[None], axis=-1)
            expected_i, expected_j = np.nonzero((dense < 4.0) & (dense > 0))
            got_i, got_j, _ = pairs_within(points, 4.0)
            assert np.array_equal(got_i, expected_i)
            assert np.array_equal(got_j, expected_j)


# -- Sphere point occlusion ---------------------------------------------------


class TestSpherePointExposure:
    @pytest.mark.parametrize("seed", [0, 5])
    def test_uniform_counts_match_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        coords = rng.normal(scale=2.5, size=(50, 3)).astype(np.float32)
        radii = rng.uniform(1.5, 3.0, size=50).astype(np.float32)
        counts = np.full(50, 60, dtype=np.int64)
        assert np.array_equal(
            sphere_point_exposure(coords, radii, counts, _fibonacci_sphere),
            brute_force_exposure(coords, radii, counts),
        )

    def test_variable_counts_match_brute_force(self):
        """Atoms are sampled in proportion to area, so counts differ per atom."""
        rng = np.random.default_rng(6)
        coords = rng.normal(scale=2.5, size=(40, 3)).astype(np.float32)
        radii = rng.uniform(1.5, 3.0, size=40).astype(np.float32)
        counts = rng.integers(20, 90, size=40).astype(np.int64)
        assert np.array_equal(
            sphere_point_exposure(coords, radii, counts, _fibonacci_sphere),
            brute_force_exposure(coords, radii, counts),
        )

    def test_lone_sphere_is_fully_exposed(self):
        exposed = sphere_point_exposure(
            np.zeros((1, 3), np.float32), np.float32([2.0]),
            np.array([50], np.int64), _fibonacci_sphere,
        )
        assert exposed.all()

    def test_engulfed_sphere_is_fully_buried(self):
        coords = np.zeros((2, 3), dtype=np.float32)
        exposed = sphere_point_exposure(
            coords, np.float32([0.4, 5.0]), np.array([40, 40], np.int64), _fibonacci_sphere
        )
        assert not exposed[:40].any()
        assert exposed[40:].all()

    def test_empty_input(self):
        assert sphere_point_exposure(
            np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
            np.zeros(0, np.int64), _fibonacci_sphere,
        ).shape == (0,)

    def test_zero_count_atoms_contribute_nothing(self):
        coords = np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32)
        counts = np.array([0, 30], dtype=np.int64)
        exposed = sphere_point_exposure(coords, np.float32([2.0, 2.0]), counts, _fibonacci_sphere)
        assert exposed.shape == (30,)

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(InputError, match="counts"):
            sphere_point_exposure(
                np.zeros((3, 3), np.float32), np.zeros(3, np.float32),
                np.zeros(2, np.int64), _fibonacci_sphere,
            )


# -- k nearest neighbours -----------------------------------------------------


class TestKnn:
    @pytest.mark.parametrize("k", [1, 4, 25])
    def test_native_matches_brute_force(self, k):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(300, 3)).astype(np.float32) * 8.0
        queries = rng.normal(size=(120, 3)).astype(np.float32) * 8.0
        expected_d, expected_i = brute_force_knn(data, queries, k)
        got_d, got_i = knn(data, queries, k, backend="native")
        assert np.array_equal(got_i, expected_i)
        assert np.allclose(got_d, expected_d, atol=1e-4)

    def test_native_matches_scipy_on_a_surface_like_shell(self):
        """The clouds this runs on are 2D shells, where a volume-based density
        estimate is badly wrong; the grid has to size itself from the data."""
        rng = np.random.default_rng(12)
        directions = rng.normal(size=(4000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        shell = (directions * (20.0 + rng.normal(scale=0.3, size=(4000, 1)))).astype(np.float32)
        native_d, native_i = knn(shell, shell, 40, backend="native")
        scipy_d, scipy_i = knn(shell, shell, 40, backend="scipy")
        assert np.array_equal(native_i, scipy_i)
        assert np.allclose(native_d, scipy_d, atol=1e-4)

    @pytest.mark.parametrize(
        "name",
        ["rod", "two distant clusters", "plane", "collinear", "coincident"],
    )
    def test_awkward_geometries_still_match_scipy(self, name):
        """A uniform grid is where degenerate shapes go wrong.

        Each of these breaks a different assumption a grid could make: the
        points fill one cell, or none of the cells between two clusters, or only
        a slab of them.
        """
        rng = np.random.default_rng(21)
        shapes = {
            "rod": rng.uniform([0, 0, 0], [900, 1, 1], size=(3000, 3)),
            "two distant clusters": np.vstack(
                [blob := rng.normal(size=(1500, 3)), blob + 900.0]
            ),
            "plane": np.c_[rng.uniform(0, 40, size=(3000, 2)), np.zeros(3000)],
            "collinear": np.c_[np.arange(2000), np.zeros(2000), np.zeros(2000)],
            "coincident": np.zeros((400, 3)),
        }
        points = shapes[name].astype(np.float32)
        native_d, _ = knn(points, points, 12, backend="native")
        scipy_d, _ = knn(points, points, 12, backend="scipy")
        assert np.allclose(native_d, scipy_d, atol=1e-4)

    def test_queries_outside_the_data_box(self):
        rng = np.random.default_rng(13)
        data = rng.uniform(0.0, 5.0, size=(200, 3)).astype(np.float32)
        queries = rng.uniform(-40.0, 45.0, size=(50, 3)).astype(np.float32)
        expected_d, expected_i = brute_force_knn(data, queries, 8)
        got_d, got_i = knn(data, queries, 8, backend="native")
        assert np.array_equal(got_i, expected_i)
        assert np.allclose(got_d, expected_d, atol=1e-3)

    def test_duplicate_points(self):
        data = np.repeat(np.arange(6, dtype=np.float32)[:, None], 3, axis=1)
        data = np.repeat(data, 4, axis=0)
        got_d, _ = knn(data, data, 4, backend="native")
        assert np.allclose(got_d, 0.0)

    def test_k_larger_than_the_data_is_clamped(self):
        data = np.zeros((3, 3), dtype=np.float32)
        data[:, 0] = [0.0, 1.0, 2.0]
        distances, indices = knn(data, data, 10, backend="native")
        assert distances.shape == indices.shape == (3, 3)

    def test_k_of_one_returns_two_dimensional_output(self):
        rng = np.random.default_rng(14)
        data = rng.normal(size=(50, 3)).astype(np.float32)
        for backend in ("native", "scipy"):
            distances, indices = knn(data, data, 1, backend=backend)
            assert distances.shape == indices.shape == (50, 1)

    def test_bad_shape_is_rejected(self):
        with pytest.raises(InputError, match=r"\(N, 3\)"):
            knn(np.zeros((4, 2)), np.zeros((2, 3)), 1)

    def test_empty_data_is_rejected(self):
        with pytest.raises(InputError, match="k must be"):
            knn(np.zeros((0, 3)), np.zeros((2, 3)), 1)


class TestNeighbourIndex:
    def test_reuse_gives_the_same_answer_as_a_fresh_query(self):
        rng = np.random.default_rng(15)
        data = rng.normal(size=(500, 3)).astype(np.float32) * 5
        index = NeighbourIndex(data, backend="native")
        first = index.query(data[:100], 12)
        second = index.query(data[:100], 12)
        assert np.array_equal(first[1], second[1])
        assert np.array_equal(first[1], knn(data, data[:100], 12, backend="native")[1])

    def test_queried_from_several_threads(self):
        """The curvature pass queries one index from a thread pool."""
        from concurrent.futures import ThreadPoolExecutor

        rng = np.random.default_rng(16)
        data = rng.normal(size=(2000, 3)).astype(np.float32) * 5
        index = NeighbourIndex(data, backend="native")
        blocks = [data[start:start + 250] for start in range(0, 2000, 250)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda block: index.query(block, 10)[1], blocks))
        assert np.array_equal(np.concatenate(results), knn(data, data, 10, backend="native")[1])


# -- Backend selection --------------------------------------------------------


class TestBackendSelection:
    def test_default_is_auto_and_resolves_to_scipy_here(self):
        assert get_spatial_backend() in SPATIAL_BACKENDS
        set_spatial_backend("auto")
        assert resolve_spatial_backend() == "scipy"

    def test_explicit_choices(self):
        for name in ("native", "scipy"):
            set_spatial_backend(name)
            assert get_spatial_backend() == name
            assert resolve_spatial_backend() == name

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(InputError, match="Unknown spatial backend"):
            set_spatial_backend("kdtree")


# -- The features themselves --------------------------------------------------


class TestFeaturesAgreeAcrossBackends:
    def test_surface_features_match(self, example_pdb):
        """Only float32 rounding may separate the two backends."""
        from plmol import Protein

        values = []
        for backend in ("scipy", "native"):
            set_spatial_backend(backend)
            surface = Protein.from_pdb(example_pdb).featurize(mode="surface")["surface"]
            values.append(np.asarray(surface["features"]))
        assert values[0].shape == values[1].shape
        assert np.abs(values[0] - values[1]).max() < 1e-4

    def test_sasa_does_not_depend_on_the_spatial_backend(self, example_pdb):
        """SASA no longer goes through a neighbour index at all."""
        from plmol.parsers import PDBParser
        from plmol.sasa import element_radius, shrake_rupley

        atoms = PDBParser(example_pdb).protein_atoms
        coords = np.array([a.coords for a in atoms], dtype=np.float32)
        radii = np.array([element_radius(a.element) for a in atoms], dtype=np.float32)
        set_spatial_backend("native")
        native = shrake_rupley(coords, radii)
        set_spatial_backend("scipy")
        assert np.array_equal(native, shrake_rupley(coords, radii))
