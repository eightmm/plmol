"""Tests for plmol/base.py — BaseMolecule abstract class."""

import numpy as np
import pytest

from plmol.base import BaseMolecule


class ConcreteMolecule(BaseMolecule):
    """Minimal concrete subclass for testing."""

    def featurize(self, mode="all"):
        return {"mode": mode}


class TestBaseMolecule:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseMolecule()

    def test_concrete_init(self):
        mol = ConcreteMolecule()
        assert mol.sequence is None
        assert mol.coords is None
        assert mol.has_3d is False
        assert mol.has_surface is False
        assert mol.metadata == {}

    def test_set_surface_without_faces(self):
        mol = ConcreteMolecule()
        points = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)
        mol.set_surface(points, normals)
        assert mol.has_surface is True
        surface = mol.get_surface()
        assert surface is not None
        assert "points" in surface
        assert "faces" not in surface
        assert "normals" in surface
        assert "verts" in surface
        np.testing.assert_array_equal(surface["points"], points)
        np.testing.assert_array_equal(surface["verts"], points)

    def test_set_surface_with_faces(self):
        mol = ConcreteMolecule()
        points = np.random.randn(100, 3)
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        normals = np.random.randn(100, 3)
        mol.set_surface(points, normals, faces=faces)
        assert mol.has_surface is True
        surface = mol.get_surface()
        assert "faces" in surface
        np.testing.assert_array_equal(surface["faces"], faces)

    def test_get_surface_none(self):
        mol = ConcreteMolecule()
        assert mol.get_surface() is None

    def test_featurize(self):
        mol = ConcreteMolecule()
        result = mol.featurize("graph")
        assert result == {"mode": "graph"}

    def test_metadata(self):
        mol = ConcreteMolecule()
        mol.metadata["key"] = "value"
        assert mol.metadata["key"] == "value"

    def test_coords_set(self):
        mol = ConcreteMolecule()
        mol._coords = np.array([[1.0, 2.0, 3.0]])
        assert mol.has_3d is True
        np.testing.assert_array_equal(mol.coords, [[1.0, 2.0, 3.0]])

    def test_sequence_set(self):
        mol = ConcreteMolecule()
        mol._sequence = "ACDEFG"
        assert mol.sequence == "ACDEFG"
