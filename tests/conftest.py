"""Shared test fixtures for plmol."""

import os

import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


@pytest.fixture
def example_pdb() -> str:
    """Path to real PDB file (10gs)."""
    path = os.path.join(EXAMPLES_DIR, "10gs_protein.pdb")
    assert os.path.exists(path), f"Example PDB not found: {path}"
    return path


@pytest.fixture
def example_sdf() -> str:
    """Path to real SDF ligand file (10gs)."""
    path = os.path.join(EXAMPLES_DIR, "10gs_ligand.sdf")
    assert os.path.exists(path), f"Example SDF not found: {path}"
    return path


@pytest.fixture
def aspirin_smiles() -> str:
    return "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def ethanol_smiles() -> str:
    return "CCO"


@pytest.fixture
def mini_pdb(tmp_path) -> str:
    """10-residue 2-chain minimal PDB for fast pipeline tests."""
    lines = []
    atom_num = 1
    # Chain A: 5 residues (ALA x5)
    residues_a = [("ALA", i + 1) for i in range(5)]
    for res_name, res_num in residues_a:
        z = float(res_num - 1) * 3.8  # ~3.8A CA-CA distance
        atoms = [
            ("N", -0.5, 0.0, z - 1.0, "N"),
            ("CA", 0.0, 0.0, z, "C"),
            ("C", 0.5, 0.0, z + 0.5, "C"),
            ("O", 0.5, 1.0, z + 0.5, "O"),
            ("CB", 0.0, 1.5, z, "C"),
        ]
        for aname, x, y, zc, elem in atoms:
            lines.append(
                f"ATOM  {atom_num:5d}  {aname:<4s}{res_name:3s} A{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{zc:8.3f}  1.00  0.00          {elem:>2s}\n"
            )
            atom_num += 1

    # Chain B: 5 residues (GLY x5)
    residues_b = [("GLY", i + 1) for i in range(5)]
    for res_name, res_num in residues_b:
        z = float(res_num - 1) * 3.8
        atoms = [
            ("N", 10.0, 0.0, z - 1.0, "N"),
            ("CA", 10.5, 0.0, z, "C"),
            ("C", 11.0, 0.0, z + 0.5, "C"),
            ("O", 11.0, 1.0, z + 0.5, "O"),
        ]
        for aname, x, y, zc, elem in atoms:
            lines.append(
                f"ATOM  {atom_num:5d}  {aname:<4s}{res_name:3s} B{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{zc:8.3f}  1.00  0.00          {elem:>2s}\n"
            )
            atom_num += 1

    lines.append("END\n")
    pdb_path = str(tmp_path / "mini.pdb")
    with open(pdb_path, "w") as f:
        f.writelines(lines)
    return pdb_path
