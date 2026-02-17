"""Tests for plmol/surface/ — point cloud surface extraction."""

import numpy as np
import pytest

from plmol.surface.features import (
    create_surface_points,
    compute_pointcloud_geometry,
)
from plmol.surface import build_ligand_surface, build_protein_surface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def water_atoms():
    """3-atom water-like geometry for minimal surface tests."""
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float32)
    radii = np.array([1.52, 1.20, 1.20], dtype=np.float32)  # O, H, H VdW
    return positions, radii


@pytest.fixture
def ethanol_mol():
    """RDKit ethanol molecule with 3D conformer."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol


@pytest.fixture
def protein_atoms():
    """Minimal protein-like atoms for surface tests (5 CA atoms)."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
        [11.4, 0.0, 0.0],
        [15.2, 0.0, 0.0],
    ], dtype=np.float32)
    radii = np.array([1.7, 1.7, 1.7, 1.7, 1.7], dtype=np.float32)
    metadata = [
        {"res_name": "ALA", "atom_name": "CA", "element": "C", "b_factor": 10.0},
        {"res_name": "GLY", "atom_name": "CA", "element": "C", "b_factor": 15.0},
        {"res_name": "VAL", "atom_name": "CA", "element": "C", "b_factor": 12.0},
        {"res_name": "LEU", "atom_name": "CA", "element": "C", "b_factor": 8.0},
        {"res_name": "ILE", "atom_name": "CA", "element": "C", "b_factor": 20.0},
    ]
    return positions, radii, metadata


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestCreateSurfacePoints:
    def test_shape_and_type(self, water_atoms):
        positions, radii = water_atoms
        points, normals = create_surface_points(positions, radii, n_points_per_atom=50)
        assert points.dtype == np.float32
        assert normals.dtype == np.float32
        assert points.shape[1] == 3
        assert normals.shape[1] == 3
        assert points.shape[0] == normals.shape[0]
        assert points.shape[0] > 0

    def test_empty_input(self):
        points, normals = create_surface_points(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
        assert points.shape == (0, 3)
        assert normals.shape == (0, 3)

    def test_normals_are_unit_vectors(self, water_atoms):
        positions, radii = water_atoms
        _, normals = create_surface_points(positions, radii, n_points_per_atom=50)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestComputePointcloudGeometry:
    def test_dict_keys_and_shapes(self, water_atoms):
        positions, radii = water_atoms
        points, normals = create_surface_points(positions, radii, n_points_per_atom=50)
        geom = compute_pointcloud_geometry(points, normals)
        assert "mean_curvature" in geom
        assert "gaussian_curvature" in geom
        assert "vertex_normal" in geom
        n = points.shape[0]
        assert geom["mean_curvature"].shape == (n, 5)
        assert geom["gaussian_curvature"].shape == (n, 5)
        assert geom["vertex_normal"].shape == (n, 3)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestBuildLigandSurface:
    def test_end_to_end(self, ethanol_mol):
        from plmol.constants import VDW_RADIUS, DEFAULT_VDW_RADIUS

        mol = ethanol_mol
        coords = mol.GetConformer().GetPositions().astype(np.float32)
        radii = np.array(
            [VDW_RADIUS.get(a.GetAtomicNum(), DEFAULT_VDW_RADIUS) for a in mol.GetAtoms()],
            dtype=np.float32,
        )
        surface = build_ligand_surface(coords, radii, mol, n_points_per_atom=30)
        assert surface is not None
        assert "points" in surface
        assert "normals" in surface
        assert "faces" not in surface  # point cloud only
        assert "features" in surface
        assert surface["features"].shape[0] == surface["points"].shape[0]


class TestBuildProteinSurface:
    def test_end_to_end(self, protein_atoms):
        positions, radii, metadata = protein_atoms
        surface = build_protein_surface(
            positions, radii, metadata, n_points_per_atom=30,
        )
        assert surface is not None
        assert "points" in surface
        assert "normals" in surface
        assert "faces" not in surface
        assert "features" in surface
        assert surface["features"].shape[0] == surface["points"].shape[0]


# ---------------------------------------------------------------------------
# Featurizer integration tests
# ---------------------------------------------------------------------------

class TestLigandSurfaceViaFeaturizer:
    def test_featurize_surface(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from plmol import Ligand

        # Build mol with conformer manually (avoid generate_conformer bug)
        mol = Chem.MolFromSmiles("CCO")
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        mol_noh = Chem.RemoveHs(mol_h)

        ligand = Ligand.from_smiles("CCO")
        ligand._rdmol = mol_noh  # inject conformer

        result = ligand.featurize(mode="surface")
        surface = result["surface"]
        assert surface is not None
        assert "points" in surface
        assert "normals" in surface
        assert "features" in surface


class TestProteinSurfaceViaFeaturizer:
    def test_featurize_surface(self, tmp_path):
        from plmol import Protein

        # Create a minimal PDB file
        pdb_content = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 10.00           O
ATOM      5  CB  ALA A   1       1.986  -0.764   1.199  1.00 10.00           C
ATOM      6  N   GLY A   2       3.322   1.490   0.000  1.00 12.00           N
ATOM      7  CA  GLY A   2       3.954   2.806   0.000  1.00 12.00           C
ATOM      8  C   GLY A   2       5.470   2.710   0.000  1.00 12.00           C
ATOM      9  O   GLY A   2       6.080   1.640   0.000  1.00 12.00           O
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content)

        protein = Protein.from_pdb(str(pdb_file), standardize=False)
        result = protein.featurize(mode="surface")
        surface = result["surface"]
        assert surface is not None
        assert "points" in surface
        assert "features" in surface
