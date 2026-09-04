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


gemmi = pytest.importorskip("gemmi", reason="mmCIF support needs the gemmi extra")


class TestMmcifAndPdbAgree:
    """The same structure, read either way, must featurize to the same numbers."""

    @staticmethod
    def _as_mmcif(pdb_path: str, target) -> str:
        structure = gemmi.read_structure(pdb_path)
        structure.setup_entities()
        structure.make_mmcif_document().write_file(str(target))
        return str(target)

    @pytest.mark.parametrize("standardize", [True, False])
    @pytest.mark.parametrize("mode", ["sequence", "graph", "atom_graph", "backbone"])
    def test_every_mode_matches(self, example_pdb, tmp_path, mode, standardize):
        from plmol import Protein
        from plmol.parsers.pdb_parser import PDBParser

        cif = self._as_mmcif(example_pdb, tmp_path / "converted.cif")
        PDBParser.clear_cache()
        from_pdb = Protein.from_pdb(example_pdb, standardize=standardize).featurize(mode=mode)[mode]
        from_cif = Protein.from_structure(cif, standardize=standardize).featurize(mode=mode)[mode]

        if isinstance(from_pdb, (str, dict)) and not isinstance(from_pdb, dict):
            assert from_pdb == from_cif
            return
        if all(isinstance(v, str) for v in from_pdb.values()):
            assert from_pdb == from_cif
            return
        for key, left in from_pdb.items():
            right = from_cif[key]
            pairs = list(zip(left, right)) if isinstance(left, tuple) else [(left, right)]
            for index, (a, b) in enumerate(pairs):
                if isinstance(a, (list, dict, str)):
                    # Ragged groupings such as residue_atom_indices, and the
                    # metadata dict: compare them as they are.
                    assert a == b, f"{key}[{index}]"
                    continue
                a, b = np.asarray(a), np.asarray(b)
                if a.dtype.kind in "OU" or a.ndim == 0:
                    continue
                assert a.shape == b.shape, f"{key}[{index}]"
                assert np.allclose(a, b, atol=1e-5, equal_nan=True), f"{key}[{index}]"


class TestFromPdbRejectsMmcif:
    def test_it_names_the_reader_to_use(self, example_pdb, tmp_path):
        from plmol import Protein
        from plmol.errors import InputError

        cif = TestMmcifAndPdbAgree._as_mmcif(example_pdb, tmp_path / "wrong_reader.cif")
        with pytest.raises(InputError) as caught:
            Protein.from_pdb(cif)
        assert "from_mmcif" in str(caught.value)


class TestOneCifGivesBothMolecules:
    """A PDBx entry holds the protein and its ligand; both come out featurizable."""

    def test_the_ligand_keeps_its_bond_orders(self, mini_complex_cif):
        from rdkit import Chem
        from plmol.complex import MolecularComplex

        complex_ = MolecularComplex.from_mmcif(mini_complex_cif, standardize=False)
        assert set(complex_.molecules) == {"protein", "ligand", "ligand_2"}

        ligand = complex_.molecules["ligand"]
        # Coordinates say which atoms are bonded, not which bonds are aromatic.
        # Without the file's own _chem_comp_bond table this benzene came back as
        # cyclohexane.
        assert Chem.MolToSmiles(ligand._rdmol) == "c1ccccc1"
        assert ligand.metadata["bond_orders_from_file"] is True

    def test_two_copies_of_one_ligand_come_out_the_same(self, mini_complex_cif):
        # The fixture holds the same benzene twice, the second with its atoms
        # nudged. Bonds inferred from distance differ between copies -- the four
        # hemes of 4HHB come out with 48, 49, 49 and 51 -- so the table has to be
        # what builds them, not what corrects them afterwards.
        from rdkit import Chem
        from plmol.complex import MolecularComplex

        molecules = MolecularComplex.from_mmcif(mini_complex_cif, standardize=False).molecules
        first = Chem.MolToSmiles(molecules["ligand"]._rdmol)
        second = Chem.MolToSmiles(molecules["ligand_2"]._rdmol)
        assert first == second == "c1ccccc1"

    def test_a_metal_is_not_a_ligand(self, mini_complex_cif):
        # A lone zinc used to come back as Ligand("[Zn]"), and in 3PTB the
        # calcium sits before the benzamidine in the file and took the "ligand"
        # key outright.
        from plmol.complex import MolecularComplex
        from plmol.parsers.mmcif_parser import MMCIFParser

        assert "ZN" not in {
            r["res_name"] for r in MMCIFParser(mini_complex_cif).get_ligand_residues()
        }
        molecules = MolecularComplex.from_mmcif(mini_complex_cif, standardize=False).molecules
        assert all(
            m.metadata.get("mmcif_ligand", {}).get("res_name") != "ZN"
            for k, m in molecules.items() if k.startswith("ligand")
        )

    def test_the_metal_is_on_the_protein_side(self, mini_complex_cif):
        # It belongs there: metal coordination is one of the interactions.
        from plmol.parsers.mmcif_parser import MMCIFParser

        parser = MMCIFParser(mini_complex_cif)
        assert [a.res_name for a in parser.metal_atoms] == ["ZN"]
        names = {a.res_name for a in parser.protein_atoms_with_metals}
        assert names == {"GLY", "ZN"}

    def test_a_file_without_the_table_still_loads(self, mini_complex_cif, tmp_path):
        from rdkit import Chem
        from plmol.complex import MolecularComplex

        text = open(mini_complex_cif).read()
        stripped = text[:text.index("loop_")] + text[text.index("loop_\n_atom_site."):]
        path = tmp_path / "no_table.cif"
        path.write_text(stripped)

        ligand = MolecularComplex.from_mmcif(str(path), standardize=False).molecules["ligand"]
        assert ligand.metadata["bond_orders_from_file"] is False
        # Proximity bonding is the fallback, and it gives the flat ring.
        assert Chem.MolToSmiles(ligand._rdmol) == "C1CCCCC1"

    def test_both_sides_featurize(self, mini_complex_cif):
        from plmol.complex import MolecularComplex

        result = MolecularComplex.from_mmcif(
            mini_complex_cif, standardize=False
        ).featurize(requests=["protein", "ligand"])
        assert np.asarray(result["protein"]["graph"]["coords"]).shape[0] == 3
        assert np.asarray(result["ligand"]["graph"]["node_features"]).shape[0] == 6

    def test_the_component_bond_table_is_read(self, mini_complex_cif):
        from plmol.parsers.mmcif_parser import MMCIFParser

        bonds = MMCIFParser(mini_complex_cif).get_component_bonds()
        assert set(bonds) == {"BNZ"}
        assert len(bonds["BNZ"]) == 6
        assert set(bonds["BNZ"].values()) == {"AROMATIC"}

    def test_a_bond_keeps_the_two_atoms_in_the_order_the_table_wrote_them(
        self, mini_complex_cif
    ):
        """The key was a frozenset, and a frozenset of two strings iterates in
        an order that depends on the interpreter's hash seed. The molecule
        built from it got its bonds begin-to-end one way in one process and the
        other way in the next."""
        from plmol.parsers.mmcif_parser import MMCIFParser

        bonds = MMCIFParser(mini_complex_cif).get_component_bonds()
        assert list(bonds["BNZ"]) == [
            ("C1", "C2"), ("C2", "C3"), ("C3", "C4"),
            ("C4", "C5"), ("C5", "C6"), ("C6", "C1"),
        ]

    def test_the_rebuilt_molecule_runs_each_bond_the_table_s_way(
        self, mini_complex_cif
    ):
        """The order above is only worth keeping if it reaches the molecule."""
        from rdkit import Chem

        from plmol.parsers.mmcif_parser import MMCIFParser
        from plmol.rdkit_utils import mol_from_component_bonds

        table = MMCIFParser(mini_complex_cif).get_component_bonds()["BNZ"]
        bare = Chem.MolFromPDBBlock(
            "\n".join(
                f"HETATM{index:5d}  C{index}  BNZ A 101      "
                f"{index:6.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
                for index in range(1, 7)
            ),
            sanitize=False,
            removeHs=False,
            proximityBonding=False,
        )
        mol, report = mol_from_component_bonds(bare, table)
        assert mol is not None and report["bonds_applied"] == 6

        name_of = [
            atom.GetPDBResidueInfo().GetName().strip() for atom in mol.GetAtoms()
        ]
        written = [
            (name_of[bond.GetBeginAtomIdx()], name_of[bond.GetEndAtomIdx()])
            for bond in mol.GetBonds()
        ]
        assert written == list(table)


class TestTheProteinSideExcludesTheLigand:
    """A whole entry read as one file put the ligand inside the protein."""

    def test_the_ligand_is_not_its_own_neighbour(self, mini_complex_cif):
        from rdkit import Chem
        from plmol.interaction.pli_featurizer import _without_solvent

        mol = Chem.MolFromPDBBlock(
            "\n".join([
                "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  GLY A   1       1.450   0.000   0.000  1.00 20.00           C",
                "HETATM    3  C1  BNZ A 101       5.400   3.800   0.000  1.00 20.00           C",
                "HETATM    4 ZN    ZN A 301       8.000   8.000   8.000  1.00 20.00          ZN",
                "HETATM    5  O   HOH A 201      20.000  20.000  20.000  1.00 20.00           O",
                "END",
            ]),
            removeHs=False, sanitize=False, proximityBonding=False,
        )
        kept = {
            atom.GetPDBResidueInfo().GetResidueName().strip()
            for atom in _without_solvent(mol).GetAtoms()
        }
        # The metal stays: coordination is one of the interactions detected.
        assert kept == {"GLY", "ZN"}

    @staticmethod
    def crowded_block() -> str:
        """A glycine with waters either side of it, so removal has to skip
        about rather than trim one end."""
        lines = []
        serial = 0
        for residue in range(1, 5):
            for water in range(2):
                serial += 1
                lines.append(
                    f"HETATM{serial:5d}  O   HOH A{200 + serial:4d}    "
                    f"{water * 3.0:8.3f}{residue * 9.0:8.3f}{0.0:8.3f}"
                    "  1.00 20.00           O"
                )
            for name, element, offset in (("N", "N", 0.0), ("CA", "C", 1.45)):
                serial += 1
                lines.append(
                    f"ATOM  {serial:5d}  {name:<3s} GLY A{residue:4d}    "
                    f"{offset:8.3f}{residue * 9.0:8.3f}{4.0:8.3f}"
                    "  1.00 20.00           {0:>2s}".format(element)
                )
        return "\n".join(lines) + "\nEND"

    @staticmethod
    def surviving_atoms(mol) -> list:
        return [
            (
                atom.GetPDBResidueInfo().GetResidueName().strip(),
                atom.GetPDBResidueInfo().GetResidueNumber(),
                atom.GetPDBResidueInfo().GetName().strip(),
            )
            for atom in mol.GetAtoms()
        ]

    def test_the_atoms_that_stay_keep_their_order(self):
        """The removal is batched where RDKit supports it. What the batch must
        preserve is which atoms survive and the order they are in, because
        every downstream index is a position in this list."""
        from rdkit import Chem
        from plmol.interaction.pli_featurizer import _without_solvent

        mol = Chem.MolFromPDBBlock(
            self.crowded_block(), removeHs=False, sanitize=False,
            proximityBonding=False,
        )
        before = self.surviving_atoms(mol)
        kept = self.surviving_atoms(_without_solvent(mol))
        assert kept == [atom for atom in before if atom[0] != "HOH"]
        assert len(kept) == 8

    def test_the_removal_agrees_with_rdkits_older_one_at_a_time_route(
        self, monkeypatch
    ):
        """BeginBatchEdit arrived in RDKit 2021.03 and the package accepts
        2020.09, so the loop it replaces is still live code for those."""
        import types

        from rdkit import Chem
        from plmol.interaction import pli_featurizer
        from plmol.interaction.pli_featurizer import _without_solvent

        class WithoutBatchEdit:
            """An RWMol that does not admit to having BeginBatchEdit."""

            def __init__(self, mol):
                self._editable = Chem.RWMol(mol)

            def __getattr__(self, name):
                if name == "BeginBatchEdit":
                    raise AttributeError(name)
                return getattr(self._editable, name)

        mol = Chem.MolFromPDBBlock(
            self.crowded_block(), removeHs=False, sanitize=False,
            proximityBonding=False,
        )
        batched = self.surviving_atoms(_without_solvent(mol))

        # _without_solvent reads Chem out of its own module globals, so this
        # reaches it.
        monkeypatch.setattr(
            pli_featurizer, "Chem",
            types.SimpleNamespace(RWMol=WithoutBatchEdit, Mol=Chem.Mol),
        )
        one_at_a_time = self.surviving_atoms(_without_solvent(mol))
        assert one_at_a_time == batched
        assert len(one_at_a_time) == 8


class TestALenientReadForAnAwkwardStructure:
    """MolFromPDBFile refuses a whole file over one bond it should not have made."""

    def test_a_five_valent_carbon_does_not_lose_the_protein(self, tmp_path):
        # Two carbons 1.4 A apart in a ring RDKit also bonds across: the
        # histidine CG of 4HHB ends up with five bonds and strict sanitisation
        # returns None for the entire structure.
        from plmol.rdkit_utils import mol_from_pdb_file

        path = tmp_path / "crowded.pdb"
        path.write_text(
            "ATOM      1  CB  HIS A   1       0.000   0.000   0.000  1.00 20.00           C\n"
            "ATOM      2  CG  HIS A   1       1.500   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  ND1 HIS A   1       2.300   1.100   0.000  1.00 20.00           N\n"
            "ATOM      4  CD2 HIS A   1       2.300  -1.100   0.000  1.00 20.00           C\n"
            "ATOM      5  CE1 HIS A   1       3.100   0.500   0.000  1.00 20.00           C\n"
            "ATOM      6  NE2 HIS A   1       3.400  -0.800   0.000  1.00 20.00           N\n"
            "END\n"
        )
        from rdkit import Chem
        assert Chem.MolFromPDBFile(str(path), removeHs=False) is None
        mol = mol_from_pdb_file(str(path))
        assert mol is not None and mol.GetNumAtoms() == 6

    def test_a_normal_structure_reads_the_same_either_way(self, example_pdb):
        from rdkit import Chem
        from plmol.constants import PHARMACOPHORE_SMARTS
        from plmol.rdkit_utils import mol_from_pdb_file, substructure_matches

        strict = Chem.MolFromPDBFile(example_pdb, removeHs=False)
        through = mol_from_pdb_file(example_pdb)
        assert strict.GetNumBonds() == through.GetNumBonds()
        for name in ("hydrophobic", "aromatic", "h_donor", "h_acceptor"):
            pattern = Chem.MolFromSmarts(PHARMACOPHORE_SMARTS[name])
            assert len(substructure_matches(strict, pattern)) == len(
                substructure_matches(through, pattern)
            )


class TestAQuaternaryNitrogenIsCharged:
    """The component tables carry no charge column."""

    CIF = """data_TEST
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
QNT N1 C1 sing N N 1
QNT N1 C2 sing N N 2
QNT N1 C3 sing N N 3
QNT N1 C4 sing N N 4
QNT C4 C5 sing N N 5
QNT C5 C6 sing N N 6
QNT C6 C7 sing N N 7
QNT C7 C8 sing N N 8
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
HETATM 5 N N1 . QNT A 1 101 ? 10.000 0.000 0.000 1.00 20.00 ? 101 QNT A N1 1
HETATM 6 C C1 . QNT A 1 101 ? 11.500 0.000 0.000 1.00 20.00 ? 101 QNT A C1 1
HETATM 7 C C2 . QNT A 1 101 ? 9.500 1.400 0.000 1.00 20.00 ? 101 QNT A C2 1
HETATM 8 C C3 . QNT A 1 101 ? 9.500 -0.700 1.200 1.00 20.00 ? 101 QNT A C3 1
HETATM 9 C C4 . QNT A 1 101 ? 9.500 -0.700 -1.200 1.00 20.00 ? 101 QNT A C4 1
HETATM 10 C C5 . QNT A 1 101 ? 10.000 -2.100 -1.400 1.00 20.00 ? 101 QNT A C5 1
HETATM 11 C C6 . QNT A 1 101 ? 9.500 -2.800 -2.600 1.00 20.00 ? 101 QNT A C6 1
HETATM 12 C C7 . QNT A 1 101 ? 10.000 -4.200 -2.800 1.00 20.00 ? 101 QNT A C7 1
HETATM 13 C C8 . QNT A 1 101 ? 9.500 -4.900 -4.000 1.00 20.00 ? 101 QNT A C8 1
#
"""

    def test_the_nitrogen_gets_its_charge(self, tmp_path):
        # Four bonds on a neutral nitrogen fails to sanitize, and the file has no
        # charge column to say otherwise. Charging it is the one repair tried.
        from rdkit import Chem
        from plmol.complex import MolecularComplex

        path = tmp_path / "quaternary.cif"
        path.write_text(self.CIF)
        ligand = MolecularComplex.from_mmcif(str(path), standardize=False).molecules["ligand"]

        assert ligand.metadata["bond_orders_from_file"] is True
        assert ligand.metadata["component_bond_report"].get("nitrogens_charged") is True
        assert sum(a.GetFormalCharge() for a in ligand._rdmol.GetAtoms()) == 1
        assert Chem.MolToSmiles(ligand._rdmol) == "CCCCC[N+](C)(C)C"
