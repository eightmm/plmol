"""
Tests for MMCIFParser and mmCIF integration with Protein/NucleicAcid/MolecularComplex.
"""

import pytest
import numpy as np
import os

gemmi = pytest.importorskip("gemmi")

from plmol.parsers.mmcif_parser import MMCIFParser
from plmol import Ligand, NucleicAcid, Protein, MolecularComplex, MMCIFParser as TopLevelMMCIF


# ---------------------------------------------------------------------------
# Fixture: generate a minimal mmCIF file programmatically
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_cif(tmp_path):
    """Minimal mmCIF: protein chain A (ALA, GLY) + nucleotide chain B (DA)."""
    st = gemmi.Structure()
    st.name = "TEST"
    model = gemmi.Model("1")

    # Chain A: 2 protein residues
    chain_a = gemmi.Chain("A")
    for res_num, (rname, atoms) in enumerate([
        ("ALA", [("N", (0.0, 0.0, 0.0)), ("CA", (1.5, 0.0, 0.0)), ("C", (2.5, 1.0, 0.0)), ("O", (2.5, 2.0, 0.0))]),
        ("GLY", [("N", (3.5, 0.5, 0.0)), ("CA", (4.5, 0.0, 0.0)), ("C", (5.5, 1.0, 0.0)), ("O", (5.5, 2.0, 0.0))]),
    ], start=1):
        res = gemmi.Residue()
        res.name = rname
        res.seqid = gemmi.SeqId(res_num, " ")
        res.het_flag = "A"
        for aname, xyz in atoms:
            atom = gemmi.Atom()
            atom.name = aname
            atom.element = gemmi.Element(aname[0])
            atom.pos = gemmi.Position(*xyz)
            atom.b_iso = 0.0
            atom.occ = 1.0
            res.add_atom(atom)
        chain_a.add_residue(res)
    model.add_chain(chain_a)

    # Chain B: 1 DNA residue (DA)
    chain_b = gemmi.Chain("B")
    res = gemmi.Residue()
    res.name = "DA"
    res.seqid = gemmi.SeqId(1, " ")
    res.het_flag = "A"
    for aname, xyz in [("P", (10.0, 0.0, 0.0)), ("C1'", (11.5, 0.0, 0.0)), ("C4'", (12.0, 1.0, 0.0))]:
        atom = gemmi.Atom()
        atom.name = aname
        atom.element = gemmi.Element(aname[0])
        atom.pos = gemmi.Position(*xyz)
        atom.b_iso = 0.0
        atom.occ = 1.0
        res.add_atom(atom)
    chain_b.add_residue(res)
    model.add_chain(chain_b)

    st.add_model(model)
    doc = st.make_mmcif_document()
    cif_path = str(tmp_path / "test.cif")
    doc.write_file(cif_path)
    return cif_path


@pytest.fixture
def protein_only_cif(tmp_path):
    """mmCIF with only protein residues (ALA, GLY, VAL)."""
    st = gemmi.Structure()
    st.name = "PROT"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")

    for res_num, rname in enumerate(["ALA", "GLY", "VAL"], start=1):
        res = gemmi.Residue()
        res.name = rname
        res.seqid = gemmi.SeqId(res_num, " ")
        res.het_flag = "A"
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(float(res_num) * 3.8, 0.0, 0.0)
        atom.b_iso = 0.0
        atom.occ = 1.0
        res.add_atom(atom)
        chain.add_residue(res)

    model.add_chain(chain)
    st.add_model(model)
    doc = st.make_mmcif_document()
    path = str(tmp_path / "prot_only.cif")
    doc.write_file(path)
    return path


@pytest.fixture
def ligand_cif(tmp_path):
    """Minimal mmCIF with protein chain A and one HETATM ligand residue."""
    st = gemmi.Structure()
    st.name = "LIGCIF"
    model = gemmi.Model("1")

    chain_a = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "ALA"
    res.seqid = gemmi.SeqId(1, " ")
    res.het_flag = "A"
    for aname, elem, xyz in [
        ("N", "N", (0.0, 0.0, 0.0)),
        ("CA", "C", (1.5, 0.0, 0.0)),
        ("C", "C", (2.5, 1.0, 0.0)),
        ("O", "O", (2.5, 2.0, 0.0)),
    ]:
        atom = gemmi.Atom()
        atom.name = aname
        atom.element = gemmi.Element(elem)
        atom.pos = gemmi.Position(*xyz)
        atom.b_iso = 0.0
        atom.occ = 1.0
        res.add_atom(atom)
    chain_a.add_residue(res)
    model.add_chain(chain_a)

    chain_l = gemmi.Chain("L")
    lig = gemmi.Residue()
    lig.name = "LIG"
    lig.seqid = gemmi.SeqId(1, " ")
    lig.het_flag = "H"
    for aname, elem, xyz in [
        ("C1", "C", (5.0, 0.0, 0.0)),
        ("O1", "O", (6.2, 0.0, 0.0)),
        ("N1", "N", (4.2, 1.0, 0.0)),
    ]:
        atom = gemmi.Atom()
        atom.name = aname
        atom.element = gemmi.Element(elem)
        atom.pos = gemmi.Position(*xyz)
        atom.b_iso = 0.0
        atom.occ = 1.0
        lig.add_atom(atom)
    chain_l.add_residue(lig)
    model.add_chain(chain_l)

    st.add_model(model)
    doc = st.make_mmcif_document()
    path = str(tmp_path / "with_ligand.cif")
    doc.write_file(path)
    return path


# ---------------------------------------------------------------------------
# Tests: MMCIFParser basic API
# ---------------------------------------------------------------------------

class TestMMCIFParserBasic:
    def test_load(self, mini_cif):
        p = MMCIFParser(mini_cif)
        assert p.mmcif_path.endswith(".cif")

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MMCIFParser(str(tmp_path / "nonexistent.cif"))

    def test_get_chains(self, mini_cif):
        p = MMCIFParser(mini_cif)
        chains = p.get_chains()
        assert "A" in chains
        assert "B" in chains

    def test_get_entities(self, mini_cif):
        p = MMCIFParser(mini_cif)
        entities = p.get_entities()
        assert isinstance(entities, dict)

    def test_get_sequence_protein_chain(self, mini_cif):
        p = MMCIFParser(mini_cif)
        seq = p.get_sequence("A")
        assert isinstance(seq, str)
        assert len(seq) == 2  # ALA + GLY

    def test_get_sequence_empty_chain(self, mini_cif):
        p = MMCIFParser(mini_cif)
        seq = p.get_sequence("Z")
        assert seq == ""


# ---------------------------------------------------------------------------
# Tests: Atom data
# ---------------------------------------------------------------------------

class TestAtomData:
    def test_get_atom_data_returns_list(self, mini_cif):
        p = MMCIFParser(mini_cif)
        atoms = p.get_atom_data()
        assert isinstance(atoms, list)
        assert len(atoms) > 0

    def test_atom_dict_keys(self, mini_cif):
        p = MMCIFParser(mini_cif)
        atom = p.get_atom_data()[0]
        for key in ["atom_name", "res_name", "res_num", "chain_id", "coords", "element", "record_type"]:
            assert key in atom

    def test_atom_coords_tuple(self, mini_cif):
        p = MMCIFParser(mini_cif)
        atom = p.get_atom_data()[0]
        assert len(atom["coords"]) == 3
        assert all(isinstance(v, float) for v in atom["coords"])

    def test_filter_by_chain(self, mini_cif):
        p = MMCIFParser(mini_cif)
        atoms_a = p.get_atom_data(chain_id="A")
        atoms_b = p.get_atom_data(chain_id="B")
        assert all(a["chain_id"] == "A" for a in atoms_a)
        assert all(a["chain_id"] == "B" for a in atoms_b)
        assert len(atoms_a) + len(atoms_b) == len(p.get_atom_data())

    def test_get_atom_coords_shape(self, mini_cif):
        p = MMCIFParser(mini_cif)
        coords = p.get_atom_coords()
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        assert coords.dtype == np.float32

    def test_get_atom_coords_chain_filter(self, mini_cif):
        p = MMCIFParser(mini_cif)
        all_coords = p.get_atom_coords()
        chain_a_coords = p.get_atom_coords(chain_id="A")
        assert chain_a_coords.shape[0] < all_coords.shape[0]

    def test_include_nucleic_acids_false(self, mini_cif):
        p = MMCIFParser(mini_cif, include_nucleic_acids=False)
        atoms = p.get_atom_data()
        res_names = {a["res_name"] for a in atoms}
        assert "DA" not in res_names

    def test_include_nucleic_acids_true(self, mini_cif):
        p = MMCIFParser(mini_cif, include_nucleic_acids=True)
        atoms = p.get_atom_data()
        res_names = {a["res_name"] for a in atoms}
        assert "DA" in res_names


# ---------------------------------------------------------------------------
# Tests: PDB conversion
# ---------------------------------------------------------------------------

class TestPDBConversion:
    def test_to_pdb_string(self, mini_cif):
        p = MMCIFParser(mini_cif)
        pdb_str = p.to_pdb_string()
        assert isinstance(pdb_str, str)
        assert len(pdb_str) > 0

    def test_to_pdb_file(self, mini_cif, tmp_path):
        p = MMCIFParser(mini_cif)
        out = str(tmp_path / "out.pdb")
        result = p.to_pdb_file(out)
        assert result == out
        import os
        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert "ATOM" in content or "HETATM" in content

    def test_to_pdb_parser(self, mini_cif):
        from plmol.protein.utils import PDBParser
        p = MMCIFParser(mini_cif)
        pdb_parser = p.to_pdb_parser()
        assert isinstance(pdb_parser, PDBParser)
        assert len(pdb_parser.all_atoms) > 0


# ---------------------------------------------------------------------------
# Tests: Entity classification helpers
# ---------------------------------------------------------------------------

class TestEntityClassification:
    def test_get_protein_chains(self, mini_cif):
        p = MMCIFParser(mini_cif)
        chains = p.get_protein_chains()
        assert "A" in chains

    def test_get_nucleic_acid_chains(self, mini_cif):
        p = MMCIFParser(mini_cif, include_nucleic_acids=True)
        chains = p.get_nucleic_acid_chains()
        assert "B" in chains

    def test_get_ligand_residues(self, mini_cif):
        p = MMCIFParser(mini_cif)
        ligands = p.get_ligand_residues()
        assert isinstance(ligands, list)
        # No ligands in our test fixture
        assert all("chain_id" in l and "res_name" in l for l in ligands)

    def test_get_ligand_residues_with_ligand(self, ligand_cif):
        p = MMCIFParser(ligand_cif)
        ligands = p.get_ligand_residues()
        assert ligands == [{"chain_id": "L", "res_name": "LIG", "res_num": 1}]


# ---------------------------------------------------------------------------
# Tests: Protein.from_mmcif integration
# ---------------------------------------------------------------------------

class TestProteinFromMMCIF:
    def test_from_mmcif_creates_protein(self, protein_only_cif):
        prot = Protein.from_mmcif(protein_only_cif)
        assert prot is not None
        assert prot.metadata["source"] == protein_only_cif

    def test_from_mmcif_sequence(self, protein_only_cif):
        prot = Protein.from_mmcif(protein_only_cif)
        seq = prot.sequence
        assert isinstance(seq, str) or isinstance(seq, dict)

    def test_cleanup_removes_temp_pdb(self, protein_only_cif):
        prot = Protein.from_mmcif(protein_only_cif)
        tmp_path = prot._pdb_path
        assert tmp_path is not None
        assert os.path.exists(tmp_path)
        prot.cleanup()
        assert not os.path.exists(tmp_path)


# ---------------------------------------------------------------------------
# Tests: NucleicAcid.from_mmcif integration
# ---------------------------------------------------------------------------

class TestNucleicAcidFromMMCIF:
    def test_from_mmcif_creates_na(self, mini_cif):
        na = NucleicAcid.from_mmcif(mini_cif)
        assert na is not None
        assert na.metadata["source"] == mini_cif

    def test_cleanup_removes_temp_pdb(self, mini_cif):
        na = NucleicAcid.from_mmcif(mini_cif)
        tmp_path = na._pdb_path
        assert tmp_path is not None
        assert os.path.exists(tmp_path)
        na.cleanup()
        assert not os.path.exists(tmp_path)


# ---------------------------------------------------------------------------
# Tests: MolecularComplex.from_mmcif integration
# ---------------------------------------------------------------------------

class TestMolecularComplexFromMMCIF:
    def test_from_mmcif_creates_complex(self, mini_cif):
        mc = MolecularComplex.from_mmcif(mini_cif)
        assert mc is not None
        assert isinstance(mc, MolecularComplex)

    def test_from_mmcif_has_protein(self, mini_cif):
        mc = MolecularComplex.from_mmcif(mini_cif)
        assert "protein" in mc.molecules

    def test_from_mmcif_has_nucleic_acid(self, mini_cif):
        mc = MolecularComplex.from_mmcif(mini_cif)
        assert "nucleic_acid" in mc.molecules

    def test_protein_only_cif_no_na(self, protein_only_cif):
        mc = MolecularComplex.from_mmcif(protein_only_cif)
        assert "protein" in mc.molecules
        assert "nucleic_acid" not in mc.molecules

    def test_from_mmcif_extracts_ligand(self, ligand_cif):
        mc = MolecularComplex.from_mmcif(ligand_cif)
        assert "protein" in mc.molecules
        assert "ligand" in mc.molecules
        assert isinstance(mc.molecules["ligand"], Ligand)
        assert mc.molecules["ligand"]._rdmol.GetNumAtoms() == 3

    def test_from_mmcif_can_skip_ligands(self, ligand_cif):
        mc = MolecularComplex.from_mmcif(ligand_cif, extract_ligands=False)
        assert "protein" in mc.molecules
        assert "ligand" not in mc.molecules

    def test_from_mmcif_ligand_filter(self, ligand_cif):
        mc = MolecularComplex.from_mmcif(ligand_cif, ligand_resname="LIG", ligand_chain="L")
        assert "ligand" in mc.molecules

    def test_cleanup_removes_shared_temp_pdb(self, mini_cif):
        mc = MolecularComplex.from_mmcif(mini_cif)
        tmp_path = mc.molecules["protein"]._pdb_path
        assert tmp_path is not None
        assert os.path.exists(tmp_path)
        mc.cleanup()
        assert not os.path.exists(tmp_path)


# ---------------------------------------------------------------------------
# Tests: top-level import
# ---------------------------------------------------------------------------

def test_top_level_import():
    assert TopLevelMMCIF is MMCIFParser


# ---------------------------------------------------------------------------
# Tests: DependencyError when gemmi not available (mock)
# ---------------------------------------------------------------------------

def test_dependency_error_without_gemmi(monkeypatch):
    import plmol.parsers.mmcif_parser as mod
    monkeypatch.setattr(mod, "_GEMMI_AVAILABLE", False)
    with pytest.raises(Exception):  # DependencyError subclasses PlmolError
        MMCIFParser.__new__(MMCIFParser)
        mod._require_gemmi()
