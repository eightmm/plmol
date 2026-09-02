"""Tests for plmol/base.py — BaseMolecule abstract class."""

import os

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


class TestTempFileOwnership:
    """Temporary files are released deterministically, not only via __del__."""

    @staticmethod
    def _temp_pdbs():
        import glob
        import tempfile

        return set(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.pdb")))

    def test_molecules_and_complex_share_the_mixin(self):
        from plmol import Ligand, NucleicAcid, Protein, TempFileOwner
        from plmol.complex import MolecularComplex

        for cls in (Protein, Ligand, NucleicAcid, MolecularComplex):
            assert issubclass(cls, TempFileOwner)

    def test_context_manager_releases_the_standardized_pdb(self, example_pdb):
        from plmol import Protein

        before = self._temp_pdbs()
        with Protein.from_pdb(example_pdb) as protein:
            protein.featurize(mode="graph")
            assert self._temp_pdbs() - before, "featurizer should hold a temp file"
        assert not self._temp_pdbs() - before

    def test_cleanup_is_repeatable_and_does_not_break_reuse(self, example_pdb):
        from plmol import Protein

        protein = Protein.from_pdb(example_pdb)
        assert protein.featurize(mode="sequence")["sequence"]
        protein.cleanup()
        protein.cleanup()
        assert protein.featurize(mode="sequence")["sequence"]

    def test_complex_cleanup_reaches_its_molecules(self):
        from plmol import Ligand
        from plmol.complex import MolecularComplex

        ligand = Ligand.from_smiles("CCO")
        ligand._owned_temp_paths.append("/nonexistent/path.sdf")
        complex_ = MolecularComplex(molecules={"ligand": ligand})
        with complex_:
            pass
        assert ligand._owned_temp_paths == []

    def test_owned_paths_exist_without_calling_init(self):
        """The list is created on demand, so a subclass skipping __init__ works."""
        from plmol import TempFileOwner

        owner = TempFileOwner.__new__(TempFileOwner)
        assert owner._owned_temp_paths == []
        owner.cleanup()
