"""Shared test fixtures for plmol."""

import os
import textwrap

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

DNA_PDB = textwrap.dedent("""\
ATOM      1  P    DA A   1       1.000   0.000   0.000  1.00  0.00           P
ATOM      2  O5'  DA A   1       2.500   0.000   0.000  1.00  0.00           O
ATOM      3  C5'  DA A   1       3.500   1.000   0.000  1.00  0.00           C
ATOM      4  C4'  DA A   1       4.500   1.000   1.000  1.00  0.00           C
ATOM      5  O4'  DA A   1       5.000   2.000   1.000  1.00  0.00           O
ATOM      6  C3'  DA A   1       5.500   0.500   1.500  1.00  0.00           C
ATOM      7  O3'  DA A   1       6.500   0.500   2.000  1.00  0.00           O
ATOM      8  C2'  DA A   1       5.500   1.500   2.500  1.00  0.00           C
ATOM      9  C1'  DA A   1       5.000   2.500   1.500  1.00  0.00           C
ATOM     10  N9   DA A   1       4.000   3.000   1.500  1.00  0.00           N
ATOM     11  C4   DA A   1       3.000   3.500   1.500  1.00  0.00           C
ATOM     12  P    DT A   2       7.000   0.000   2.000  1.00  0.00           P
ATOM     13  O5'  DT A   2       8.500   0.000   2.000  1.00  0.00           O
ATOM     14  C5'  DT A   2       9.500   1.000   2.000  1.00  0.00           C
ATOM     15  C4'  DT A   2      10.500   1.000   3.000  1.00  0.00           C
ATOM     16  O4'  DT A   2      11.000   2.000   3.000  1.00  0.00           O
ATOM     17  C3'  DT A   2      11.500   0.500   3.500  1.00  0.00           C
ATOM     18  O3'  DT A   2      12.500   0.500   4.000  1.00  0.00           O
ATOM     19  C2'  DT A   2      11.500   1.500   4.500  1.00  0.00           C
ATOM     20  C1'  DT A   2      11.000   2.500   3.500  1.00  0.00           C
ATOM     21  N1   DT A   2      10.000   3.000   3.500  1.00  0.00           N
ATOM     22  C2   DT A   2       9.000   3.500   3.500  1.00  0.00           C
ATOM     23  P    DG A   3      13.000   0.000   4.000  1.00  0.00           P
ATOM     24  O5'  DG A   3      14.500   0.000   4.000  1.00  0.00           O
ATOM     25  C5'  DG A   3      15.500   1.000   4.000  1.00  0.00           C
ATOM     26  C4'  DG A   3      16.500   1.000   5.000  1.00  0.00           C
ATOM     27  O4'  DG A   3      17.000   2.000   5.000  1.00  0.00           O
ATOM     28  C3'  DG A   3      17.500   0.500   5.500  1.00  0.00           C
ATOM     29  O3'  DG A   3      18.500   0.500   6.000  1.00  0.00           O
ATOM     30  C2'  DG A   3      17.500   1.500   6.500  1.00  0.00           C
ATOM     31  C1'  DG A   3      17.000   2.500   5.500  1.00  0.00           C
ATOM     32  N9   DG A   3      16.000   3.000   5.500  1.00  0.00           N
ATOM     33  C4   DG A   3      15.000   3.500   5.500  1.00  0.00           C
ATOM     34  P    DC A   4      19.000   0.000   6.000  1.00  0.00           P
ATOM     35  O5'  DC A   4      20.500   0.000   6.000  1.00  0.00           O
ATOM     36  C5'  DC A   4      21.500   1.000   6.000  1.00  0.00           C
ATOM     37  C4'  DC A   4      22.500   1.000   7.000  1.00  0.00           C
ATOM     38  O4'  DC A   4      23.000   2.000   7.000  1.00  0.00           O
ATOM     39  C3'  DC A   4      23.500   0.500   7.500  1.00  0.00           C
ATOM     40  O3'  DC A   4      24.500   0.500   8.000  1.00  0.00           O
ATOM     41  C2'  DC A   4      23.500   1.500   8.500  1.00  0.00           C
ATOM     42  C1'  DC A   4      23.000   2.500   7.500  1.00  0.00           C
ATOM     43  N1   DC A   4      22.000   3.000   7.500  1.00  0.00           N
ATOM     44  C2   DC A   4      21.000   3.500   7.500  1.00  0.00           C
END
""")


@pytest.fixture
def dna_pdb(tmp_path) -> str:
    """Minimal 4-residue DNA strand (DA, DT, DG, DC)."""
    path = tmp_path / "dna.pdb"
    path.write_text(DNA_PDB)
    return str(path)

MINI_COMPLEX_CIF = """data_TEST
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
BNZ C1 C2 arom Y N 1
BNZ C2 C3 arom Y N 2
BNZ C3 C4 arom Y N 3
BNZ C4 C5 arom Y N 4
BNZ C5 C6 arom Y N 5
BNZ C6 C1 arom Y N 6
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . GLY A 1 1 ? 0.000 0.000 0.000 1.00 20.00 ? 1 GLY A N 1
ATOM 2 C CA . GLY A 1 1 ? 1.450 0.000 0.000 1.00 20.00 ? 1 GLY A CA 1
ATOM 3 C C . GLY A 1 1 ? 2.400 1.000 0.000 1.00 20.00 ? 1 GLY A C 1
ATOM 4 O O . GLY A 1 1 ? 2.400 2.200 0.000 1.00 20.00 ? 1 GLY A O 1
ATOM 5 N N . GLY A 1 2 ? 0.000 3.800 0.000 1.00 20.00 ? 2 GLY A N 1
ATOM 6 C CA . GLY A 1 2 ? 1.450 3.800 0.000 1.00 20.00 ? 2 GLY A CA 1
ATOM 7 C C . GLY A 1 2 ? 2.400 4.800 0.000 1.00 20.00 ? 2 GLY A C 1
ATOM 8 O O . GLY A 1 2 ? 2.400 6.000 0.000 1.00 20.00 ? 2 GLY A O 1
ATOM 9 N N . GLY A 1 3 ? 0.000 7.600 0.000 1.00 20.00 ? 3 GLY A N 1
ATOM 10 C CA . GLY A 1 3 ? 1.450 7.600 0.000 1.00 20.00 ? 3 GLY A CA 1
ATOM 11 C C . GLY A 1 3 ? 2.400 8.600 0.000 1.00 20.00 ? 3 GLY A C 1
ATOM 12 O O . GLY A 1 3 ? 2.400 9.800 0.000 1.00 20.00 ? 3 GLY A O 1
HETATM 13 C C1 . BNZ A 1 101 ? 5.400 3.800 0.000 1.00 20.00 ? 101 BNZ A C1 1
HETATM 14 C C2 . BNZ A 1 101 ? 4.700 5.012 0.000 1.00 20.00 ? 101 BNZ A C2 1
HETATM 15 C C3 . BNZ A 1 101 ? 3.300 5.012 0.000 1.00 20.00 ? 101 BNZ A C3 1
HETATM 16 C C4 . BNZ A 1 101 ? 2.600 3.800 0.000 1.00 20.00 ? 101 BNZ A C4 1
HETATM 17 C C5 . BNZ A 1 101 ? 3.300 2.588 0.000 1.00 20.00 ? 101 BNZ A C5 1
HETATM 18 C C6 . BNZ A 1 101 ? 4.700 2.588 0.000 1.00 20.00 ? 101 BNZ A C6 1
HETATM 19 O O . HOH A 1 201 ? 20.000 20.000 20.000 1.00 20.00 ? 201 HOH A O 1
#
"""


@pytest.fixture
def mini_complex_cif(tmp_path) -> str:
    """A tiny mmCIF holding protein, a benzene ligand with its bond table, and a water."""
    path = tmp_path / "mini_complex.cif"
    path.write_text(MINI_COMPLEX_CIF)
    return str(path)
