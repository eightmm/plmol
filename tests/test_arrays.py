"""Tests for the numpy spellings of the torch operations plmol used.

Each of these has an obvious numpy translation that is subtly wrong, which is
why the helpers exist at all; the tests pin the corner that differs.
"""

import numpy as np
import pytest

from plmol.arrays import (
    FLOAT,
    INT,
    normalize,
    one_hot,
    pad_last,
    pairwise_distances,
    sanitized,
    to_numpy,
    to_torch,
)
from plmol.errors import DependencyError

rng = np.random.default_rng(0)


class TestNormalize:
    def test_rows_become_unit_length(self):
        vectors = rng.normal(size=(30, 3)).astype(FLOAT)
        assert np.allclose(np.linalg.norm(normalize(vectors), axis=-1), 1.0, atol=1e-6)

    def test_a_zero_vector_stays_zero(self):
        """Dividing by ``norm + eps`` would too, but it also shrinks every other
        row; dividing by ``max(norm, eps)`` is what torch does."""
        vectors = np.zeros((3, 3), dtype=FLOAT)
        assert np.array_equal(normalize(vectors), vectors)

    def test_short_vectors_are_not_shrunk(self):
        tiny = np.full((1, 3), 1e-6, dtype=FLOAT)
        assert np.allclose(np.linalg.norm(normalize(tiny), axis=-1), 1.0, atol=1e-5)

    def test_float_width_is_kept(self):
        assert normalize(rng.normal(size=(4, 3)).astype(FLOAT)).dtype == FLOAT


class TestPadLast:
    def test_only_the_last_axis_grows(self):
        array = np.ones((2, 3), dtype=FLOAT)
        assert pad_last(array, 1, 2).shape == (2, 6)

    def test_padding_is_where_it_was_asked_for(self):
        assert np.array_equal(
            pad_last(np.array([1.0, 2.0], dtype=FLOAT), 1, 2),
            np.array([0.0, 1.0, 2.0, 0.0, 0.0], dtype=FLOAT),
        )

    def test_value_is_honoured(self):
        assert pad_last(np.zeros(2, dtype=FLOAT), 1, 0, value=5.0)[0] == 5.0


class TestPairwiseDistances:
    def test_matches_a_direct_loop(self):
        left = rng.normal(size=(6, 3)).astype(FLOAT) * 30
        right = rng.normal(size=(4, 3)).astype(FLOAT) * 30
        expected = np.array([[np.linalg.norm(a - b) for b in right] for a in left])
        assert np.allclose(pairwise_distances(left, right), expected, atol=1e-4)

    def test_leading_axes_broadcast(self):
        batch = rng.normal(size=(5, 7, 3)).astype(FLOAT)
        assert pairwise_distances(batch, batch).shape == (5, 7, 7)

    def test_the_diagonal_is_zero(self):
        points = rng.normal(size=(9, 3)).astype(FLOAT) * 40
        assert np.allclose(np.diag(pairwise_distances(points, points)), 0.0, atol=1e-4)

    def test_beats_the_expansion_far_from_the_origin(self):
        """torch.cdist switches to |a|^2 + |b|^2 - 2ab above 25 rows, and that
        form cancels away most of its digits on crystallographic coordinates."""
        points = (rng.normal(size=(40, 3)) * 2.0 + 4000.0).astype(FLOAT)
        truth = np.linalg.norm(
            points.astype(np.float64)[:, None] - points.astype(np.float64)[None], axis=-1
        )
        squared = np.float32(
            np.einsum("ij,ij->i", points, points)[:, None]
            + np.einsum("ij,ij->i", points, points)[None, :]
            - 2.0 * (points @ points.T)
        )
        expansion = np.sqrt(np.maximum(squared, 0.0))
        direct = pairwise_distances(points, points)
        assert np.abs(direct - truth).max() < np.abs(expansion - truth).max()


class TestPairwiseDistancesTranspose:
    def test_swapping_the_arguments_transposes_the_result_exactly(self):
        """The residue featurizer builds three of its four distance matrices
        and transposes the third for the fourth. That is only sound if the two
        agree bit for bit, which they do: a - b negates b - a exactly, and a
        norm squares away the sign."""
        for n in (3, 40, 430):
            left = (rng.normal(scale=25, size=(n, 3)) + 60).astype(FLOAT)
            right = (rng.normal(scale=25, size=(n, 3)) + 60).astype(FLOAT)
            assert np.array_equal(
                pairwise_distances(right, left), pairwise_distances(left, right).T
            ), n


class TestSanitized:
    """It answers ``nan_to_num``'s question by looking at the coordinates the
    array was derived from, which are thousands of values rather than millions."""

    def test_a_finite_array_comes_back_untouched(self):
        source = rng.normal(size=(20, 3)).astype(FLOAT)
        derived = pairwise_distances(source, source)
        assert sanitized(derived, source) is derived

    def test_a_non_finite_source_falls_back_to_the_scrub(self):
        source = rng.normal(size=(20, 3)).astype(FLOAT)
        source[3, 1] = np.nan
        derived = pairwise_distances(source, source)
        cleaned = sanitized(derived, source)
        assert np.isfinite(cleaned).all()
        assert np.array_equal(cleaned, np.nan_to_num(derived))

    def test_every_source_is_examined(self):
        good = rng.normal(size=(5, 3)).astype(FLOAT)
        bad = good.copy()
        bad[0, 0] = np.inf
        derived = np.full((5, 5), np.nan, dtype=FLOAT)
        assert not np.isnan(sanitized(derived, good, bad)).any()
        assert np.isnan(sanitized(derived, good, good)).all()


class TestOneHot:
    def test_rows_are_indicators(self):
        encoded = one_hot(np.array([0, 2, 1]), 3)
        assert np.array_equal(encoded, np.eye(3, dtype=FLOAT)[[0, 2, 1]])

    def test_shape_and_width(self):
        assert one_hot(np.zeros((4, 5), dtype=INT), 7).shape == (4, 5, 7)


class TestConversion:
    def test_round_trip_through_torch(self):
        torch = pytest.importorskip("torch")
        original = {
            "a": rng.normal(size=(3, 2)).astype(FLOAT),
            "b": [np.arange(4, dtype=INT), "not an array"],
            "c": (np.ones(2, dtype=bool),),
        }
        tensors = to_torch(original)
        assert isinstance(tensors["a"], torch.Tensor)
        assert tensors["b"][1] == "not an array"
        back = to_numpy(tensors)
        assert np.array_equal(back["a"], original["a"])
        assert np.array_equal(back["c"][0], original["c"][0])

    def test_dtypes_survive_the_round_trip(self):
        pytest.importorskip("torch")
        original = {"f": np.ones(2, FLOAT), "i": np.ones(2, INT), "b": np.ones(2, bool)}
        back = to_numpy(to_torch(original))
        assert [back[k].dtype for k in "fib"] == [np.dtype(FLOAT), np.dtype(INT), np.dtype(bool)]

    def test_to_numpy_leaves_arrays_alone(self):
        array = np.zeros(3)
        assert to_numpy({"x": array})["x"] is array

    def test_without_torch_the_error_names_the_fix(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
        with pytest.raises(DependencyError, match="pip install torch"):
            to_torch({"x": np.zeros(2)})
