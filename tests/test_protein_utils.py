"""Tests for plmol/protein/utils.py — PDB parsing utilities."""

import numpy as np
import pytest

from plmol.protein.utils import (
    ParsedAtom,
    PDBParser,
    is_atom_record,
    is_hetatm_record,
    is_hydrogen,
    normalize_residue_name,
    parse_pdb_line,
    parse_pdb_atom_line,
    calculate_sidechain_centroid,
)


# -- Low-level functions --

class TestIsAtomRecord:
    def test_atom(self):
        assert is_atom_record("ATOM  12345  CA  ALA A   1") is True

    def test_hetatm(self):
        assert is_atom_record("HETATM12345  CA  ALA A   1") is False

    def test_short_line(self):
        assert is_atom_record("ATOM") is False

    def test_remark(self):
        assert is_atom_record("REMARK this is a remark") is False


class TestIsHetatmRecord:
    def test_hetatm(self):
        assert is_hetatm_record("HETATM12345  CA  ALA A   1") is True

    def test_atom(self):
        assert is_hetatm_record("ATOM  12345  CA  ALA A   1") is False

    def test_short(self):
        assert is_hetatm_record("HET") is False


class TestIsHydrogen:
    def test_element_column(self):
        #                    |   atom  |res|c|resn|      x       y       z       occ  bf        el
        line = "ATOM      1  H   ALA A   1       1.000   2.000   3.000  1.00  0.00           H  "
        assert is_hydrogen(line) is True

    def test_atom_name_h(self):
        line = "ATOM      1  HB2 ALA A   1       1.000   2.000   3.000"
        assert is_hydrogen(line) is True

    def test_not_hydrogen(self):
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  "
        assert is_hydrogen(line) is False

    def test_short_line(self):
        assert is_hydrogen("ATOM") is False


def _pdb_line(name4, res="ALA", element=None, record="ATOM  "):
    """One ATOM/HETATM line with every field in its own column.

    *element* of None means the line stops at column 66, the way files written
    before the element column was mandatory do.
    """
    line = list(" " * 80)
    line[0:6] = record.ljust(6)
    line[6:11] = "    5"
    line[12:16] = name4.ljust(4)[:4]
    line[17:20] = res.rjust(3)[:3]
    line[21] = "A"
    line[22:26] = "   1"
    line[30:38] = f"{1.0:8.3f}"
    line[38:46] = f"{2.0:8.3f}"
    line[46:54] = f"{3.0:8.3f}"
    line[54:60] = "  1.00"
    line[60:66] = "  0.00"
    if element is None:
        return "".join(line)[:66] + "\n"
    line[76:78] = element.rjust(2)[:2]
    return "".join(line) + "\n"


#: (label, line, element) for lines that carry no element column -- the only
#: ones where plmol has to work the element out for itself.
NO_ELEMENT_COLUMN = [
    ("alpha carbon, right justified", _pdb_line(" CA"), "C"),
    ("alpha carbon, left justified", _pdb_line("CA"), "C"),
    ("sidechain sulfur", _pdb_line(" SG", "CYS"), "S"),
    ("nucleotide phosphorus", _pdb_line(" P", "DA"), "P"),
    ("hydrogen, modern name", _pdb_line(" HB2"), "H"),
    ("hydrogen, pre-2007 name", _pdb_line("1HB"), "H"),
    ("selenium in selenomethionine", _pdb_line("SE", "MSE"), "SE"),
    ("mercury", _pdb_line("HG", "HG", record="HETATM"), "HG"),
    ("zinc", _pdb_line("ZN", "ZN", record="HETATM"), "ZN"),
    ("calcium ion", _pdb_line("CA", "CA", record="HETATM"), "CA"),
    ("ligand gamma hydrogen", _pdb_line("HG21", "LIG", record="HETATM"), "H"),
    ("chlorine in a ligand", _pdb_line("CL1", "LIG", record="HETATM"), "CL"),
]


class TestElementWithoutTheColumn:
    """Where the element comes from when columns 77-78 are missing.

    Reading the stripped atom name alone got three of these wrong: CA came out
    as calcium in every residue, SE in selenomethionine came out as sulfur,
    and a hydrogen named the pre-2007 way, 1HB, was not recognised at all.
    """

    @pytest.mark.parametrize("label,line,element",
                             NO_ELEMENT_COLUMN, ids=[c[0] for c in NO_ELEMENT_COLUMN])
    def test_the_element_is_what_the_atom_is(self, label, line, element):
        assert parse_pdb_line(line).element == element

    @pytest.mark.parametrize("label,line,element",
                             NO_ELEMENT_COLUMN, ids=[c[0] for c in NO_ELEMENT_COLUMN])
    def test_hydrogen_means_element_h(self, label, line, element):
        """is_hydrogen is not a second rule: it is this one, read for H."""
        assert is_hydrogen(line) == (element == "H")

    def test_the_element_column_wins_when_it_is_there(self):
        """A carbon the file names HG is a carbon. The name check alone said
        hydrogen and dropped it."""
        line = _pdb_line(" HG", "CYS", element="C")
        assert parse_pdb_line(line).element == "C"
        assert is_hydrogen(line) is False

    @pytest.mark.parametrize("label,line,element",
                             NO_ELEMENT_COLUMN, ids=[c[0] for c in NO_ELEMENT_COLUMN])
    def test_standardizing_writes_the_element_the_parser_reads(self, label, line, element):
        """The standardizer fills in columns 77-78 for files that lack them,
        and the parser reads that file back. The two must agree."""
        from plmol.protein.pdb_standardizer import PDBStandardizer

        written = PDBStandardizer()._format_atom_line(line, 1, 1, "A", line[17:20].strip())
        assert parse_pdb_line(written).element == element


class TestModifiedResiduesStayInTheChain:
    """A residue plmol cannot fully describe is still a residue.

    LLP and PTR were deleted from protein_atoms by name, in five places, while
    RESIDUE_NAME_MAPPING said PTR is a tyrosine and PTM_RESIDUES listed it as a
    modification to handle. Deleting them left a hole in the backbone: a
    four-residue chain came back as a two-residue sequence.
    """

    @staticmethod
    def _chain(tmp_path):
        def line(serial, name, res, resnum, xyz, element):
            row = list(" " * 80)
            row[0:6] = "ATOM  "
            row[6:11] = f"{serial:5d}"
            row[12:16] = name.ljust(4)[:4] if len(name) == 4 else (" " + name).ljust(4)[:4]
            row[17:20] = res.rjust(3)[:3]
            row[21] = "A"
            row[22:26] = f"{resnum:4d}"
            row[30:38] = f"{xyz[0]:8.3f}"
            row[38:46] = f"{xyz[1]:8.3f}"
            row[46:54] = f"{xyz[2]:8.3f}"
            row[54:60] = "  1.00"
            row[60:66] = "  0.00"
            row[76:78] = element.rjust(2)
            return "".join(row).rstrip() + "\n"

        atoms = [
            ("N", "ALA", 1, (0.0, 0.0, 0.0), "N"), ("CA", "ALA", 1, (1.458, 0, 0), "C"),
            ("C", "ALA", 1, (2.009, 1.42, 0), "C"), ("O", "ALA", 1, (1.251, 2.39, 0), "O"),
            ("N", "PTR", 2, (3.332, 1.541, 0), "N"), ("CA", "PTR", 2, (3.988, 2.841, 0), "C"),
            ("C", "PTR", 2, (5.5, 2.7, 0), "C"), ("O", "PTR", 2, (6.1, 1.63, 0), "O"),
            ("CB", "PTR", 2, (3.6, 3.7, 1.2), "C"), ("P", "PTR", 2, (3.1, 6.9, 3.0), "P"),
            ("N", "LLP", 3, (6.1, 3.8, 0), "N"), ("CA", "LLP", 3, (7.55, 3.9, 0), "C"),
            ("C", "LLP", 3, (8.1, 5.3, 0), "C"), ("O", "LLP", 3, (7.4, 6.3, 0), "O"),
            ("CB", "LLP", 3, (8.0, 3.1, 1.2), "C"), ("C4A", "LLP", 3, (10.0, 1.0, 3.0), "C"),
            ("N", "GLY", 4, (9.4, 5.4, 0), "N"), ("CA", "GLY", 4, (10.1, 6.7, 0), "C"),
            ("C", "GLY", 4, (11.6, 6.5, 0), "C"), ("O", "GLY", 4, (12.2, 5.4, 0), "O"),
        ]
        path = tmp_path / "modified.pdb"
        path.write_text("".join(line(i, *a) for i, a in enumerate(atoms, 1)) + "END\n")
        return str(path), len(atoms)

    def test_the_sequence_has_every_residue(self, tmp_path):
        from plmol.parsers import PDBParser

        path, _ = self._chain(tmp_path)
        parser = PDBParser(path, skip_cache=True)
        assert len(parser.get_sequence()) == 4, "a modified residue is still a residue"
        assert parser.get_sequence()[1] == "Y", "PTR is a tyrosine"

    def test_the_residue_graph_has_every_residue(self, tmp_path):
        from plmol import Protein

        path, _ = self._chain(tmp_path)
        graph = Protein.from_pdb(path).featurize(mode="graph")["graph"]
        assert np.asarray(graph["coords"]).shape[0] == 4

    def test_sasa_lines_up_with_the_atoms(self, tmp_path):
        """The atom featurizer filtered these residues out again after the
        parser had kept them, so the SASA array was longer than the atom array
        and got truncated -- silently misaligning every value."""
        from plmol import Protein

        path, _ = self._chain(tmp_path)
        atom_graph = Protein.from_pdb(path).featurize(mode="atom_graph")["atom_graph"]
        atoms = np.asarray(atom_graph["coords"]).shape[0]
        # 20 in the file, 19 here: standardising maps PTR onto TYR and TYR has
        # no phosphorus, which is the PTM-to-parent behaviour the mapping asks
        # for. What matters is that nothing downstream drops more.
        assert atoms == 19
        for key in ("sasa", "relative_sasa", "burial_index", "is_polar_sasa"):
            assert np.asarray(atom_graph[key]).shape[0] == atoms, key


class TestProteinAtomFilter:
    """One filter, shared by the PDB and mmCIF parsers.

    StructureParser promises protein_atoms holds no water, hydrogen or metal.
    The mmCIF parser used to write its own shorter rule and hand back its
    metals, its ligands and its terminal oxygens.
    """

    @staticmethod
    def _atom(**kwargs):
        from plmol.parsers.pdb_parser import ParsedAtom

        fields = dict(record_type="ATOM", atom_name="CA", res_name="ALA", res_num=1,
                      chain_id="A", coords=(0.0, 0.0, 0.0), element="C")
        fields.update(kwargs)
        return ParsedAtom(**fields)

    @pytest.mark.parametrize("label,fields", [
        ("hydrogen", dict(atom_name="HB2", element="H")),
        ("deuterium", dict(atom_name="DB2", element="D")),
        ("water", dict(res_name="HOH", atom_name="O", element="O")),
        ("metal", dict(record_type="HETATM", res_name="ZN", atom_name="ZN", element="ZN")),
        ("ligand", dict(record_type="HETATM", res_name="LIG", atom_name="C1")),
        ("terminal oxygen", dict(atom_name="OXT", element="O")),
        ("nucleotide when not asked for", dict(res_name="DA", atom_name="P", element="P")),
    ], ids=lambda v: v if isinstance(v, str) else "")
    def test_what_is_left_out(self, label, fields):
        from plmol.parsers.pdb_parser import is_protein_atom

        assert is_protein_atom(self._atom(**fields)) is False

    def test_a_backbone_atom_stays(self):
        from plmol.parsers.pdb_parser import is_protein_atom

        assert is_protein_atom(self._atom()) is True

    def test_a_nucleotide_stays_when_asked_for(self):
        from plmol.parsers.pdb_parser import is_protein_atom

        atom = self._atom(res_name="DA", atom_name="P", element="P")
        assert is_protein_atom(atom, include_nucleic_acids=True) is True

    def test_the_two_parsers_agree_on_a_real_structure(self, example_pdb, tmp_path):
        """Same structure, both formats, same atoms."""
        gemmi = pytest.importorskip("gemmi")
        from plmol.parsers import MMCIFParser, PDBParser

        cif = tmp_path / "structure.cif"
        structure = gemmi.read_structure(example_pdb)
        structure.setup_entities()
        structure.make_mmcif_document().write_file(str(cif))

        key = lambda a: (a.chain_id, a.res_num, a.atom_name)  # noqa: E731
        from_pdb = PDBParser(example_pdb, skip_cache=True).protein_atoms
        from_cif = MMCIFParser(str(cif), include_nucleic_acids=False).protein_atoms
        assert {key(a) for a in from_cif} == {key(a) for a in from_pdb}


class TestParsePdbLine:
    def test_basic_parse(self):
        # Standard PDB ATOM line with proper column formatting
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  \n"
        atom = parse_pdb_line(line)
        assert isinstance(atom, ParsedAtom)
        assert atom.record_type == "ATOM"
        assert atom.atom_name == "CA"
        assert atom.res_name == "ALA"
        assert atom.chain_id == "A"
        assert atom.res_num == 1
        assert abs(atom.coords[0] - 1.0) < 1e-3
        assert abs(atom.coords[1] - 2.0) < 1e-3
        assert abs(atom.coords[2] - 3.0) < 1e-3
        assert atom.element == "C"
        assert abs(atom.b_factor - 10.0) < 1e-3

    def test_hetatm(self):
        line = "HETATM    1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  \n"
        atom = parse_pdb_line(line)
        assert atom.record_type == "HETATM"

    def test_element_inference(self):
        # Short line without element column
        line = "ATOM      1  CB  ALA A   1       1.000   2.000   3.000"
        atom = parse_pdb_line(line)
        assert atom.element == "C"


class TestNormalizeResidueName:
    def test_standard(self):
        assert normalize_residue_name("ALA") == "ALA"

    def test_histidine_variant(self):
        result = normalize_residue_name("HID")
        assert result == "HIS"

    def test_unknown(self):
        assert normalize_residue_name("XYZ") == "UNK"

    def test_metal_detection(self):
        assert normalize_residue_name("ZN", "ZN") == "METAL"


class TestParsePdbAtomLine:
    def test_returns_tuple(self):
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  \n"
        result = parse_pdb_atom_line(line)
        assert isinstance(result, tuple)
        assert len(result) == 7
        assert result[0] == "ATOM"
        assert result[1] == "CA"
        assert result[2] == "ALA"
        assert result[3] == 1
        assert result[4] == "A"


class TestCalculateSidechainCentroid:
    def test_with_sidechain(self):
        # N, CA, C, O, CB, CG (6 atoms)
        coords = np.array([
            [0, 0, 0],  # N
            [1, 0, 0],  # CA
            [2, 0, 0],  # C
            [2, 1, 0],  # O
            [1, 1, 0],  # CB
            [1, 2, 0],  # CG
        ], dtype=float)
        centroid = calculate_sidechain_centroid(coords)
        expected = np.array([1, 1.5, 0])  # mean of CB, CG
        np.testing.assert_allclose(centroid, expected)

    def test_gly_no_sidechain(self):
        # N, CA, C, O (4 atoms, index 1=CA used)
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
        ], dtype=float)
        centroid = calculate_sidechain_centroid(coords)
        np.testing.assert_allclose(centroid, [1, 0, 0])

    def test_single_atom(self):
        coords = np.array([[5.0, 6.0, 7.0]])
        centroid = calculate_sidechain_centroid(coords)
        np.testing.assert_allclose(centroid, [5, 6, 7])

    def test_1d_input(self):
        coords = np.array([5.0, 6.0, 7.0])
        centroid = calculate_sidechain_centroid(coords)
        np.testing.assert_allclose(centroid, [5, 6, 7])


# -- PDBParser class --

class TestPDBParser:
    def test_parse_mini(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        assert parser.get_num_residues() == 10
        assert parser.get_num_atoms() > 0

    def test_all_atoms(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        assert len(parser.all_atoms) > 0
        assert len(parser.protein_atoms) > 0
        assert len(parser.protein_atoms) <= len(parser.all_atoms)

    def test_residues(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        residues = parser.residues
        assert len(residues) == 10
        # Check chain A present
        chain_a_keys = [k for k in residues if k[0] == "A"]
        assert len(chain_a_keys) == 5

    def test_get_sequence(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        seq = parser.get_sequence()
        assert len(seq) == 10
        assert "A" in seq  # ALA -> A
        assert "G" in seq  # GLY -> G

    def test_get_sequence_by_chain(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        seq_dict = parser.get_sequence_by_chain()
        assert "A" in seq_dict
        assert "B" in seq_dict
        assert len(seq_dict["A"]) == 5
        assert len(seq_dict["B"]) == 5
        assert seq_dict["A"] == "AAAAA"
        assert seq_dict["B"] == "GGGGG"

    def test_get_sequence_single_chain(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        seq = parser.get_sequence(chain_id="A")
        assert seq == "AAAAA"

    def test_get_atom_coords(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        coords = parser.get_atom_coords()
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        assert coords.shape[0] == parser.get_num_atoms()

    def test_get_atom_data(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        data = parser.get_atom_data()
        keys = ["atom_names", "res_names", "res_nums", "chain_ids", "coords", "elements", "residue_keys"]
        for key in keys:
            assert key in data
            assert len(data[key]) == parser.get_num_atoms()

    def test_get_residue_list(self, mini_pdb):
        PDBParser.clear_cache()
        parser = PDBParser(mini_pdb)
        res_list = parser.get_residue_list()
        assert len(res_list) == 10
        assert all(len(r) == 3 for r in res_list)

    def test_cache(self, mini_pdb):
        PDBParser.clear_cache()
        p1 = PDBParser(mini_pdb)
        p2 = PDBParser(mini_pdb)
        assert p2._all_atoms is p1._all_atoms  # Same cached data

    def test_clear_cache(self, mini_pdb):
        PDBParser.clear_cache()
        PDBParser(mini_pdb)
        assert PDBParser.get_cached(mini_pdb) is not None
        PDBParser.clear_cache()
        assert PDBParser.get_cached(mini_pdb) is None

    def test_skip_cache(self, mini_pdb):
        PDBParser.clear_cache()
        p1 = PDBParser(mini_pdb, skip_cache=True)
        assert PDBParser.get_cached(mini_pdb) is None

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDBParser(str(tmp_path / "nonexistent.pdb"))

    def test_empty_file(self, tmp_path):
        empty_pdb = str(tmp_path / "empty.pdb")
        with open(empty_pdb, "w") as f:
            pass
        PDBParser.clear_cache()
        with pytest.raises(ValueError, match="empty"):
            PDBParser(empty_pdb)

    def test_filters_water(self, tmp_path):
        """Water molecules should be excluded from protein_atoms."""
        pdb_content = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  \n"
            "HETATM    2  O   HOH A   2       5.000   5.000   5.000  1.00  0.00           O  \n"
            "END\n"
        )
        pdb_file = str(tmp_path / "water.pdb")
        with open(pdb_file, "w") as f:
            f.write(pdb_content)
        PDBParser.clear_cache()
        parser = PDBParser(pdb_file)
        assert parser.get_num_atoms() == 1
        assert parser.protein_atoms[0].atom_name == "CA"

    def test_example_pdb(self, example_pdb):
        """Test with real PDB file."""
        PDBParser.clear_cache()
        parser = PDBParser(example_pdb)
        assert parser.get_num_residues() > 0
        assert parser.get_num_atoms() > 0
        seq = parser.get_sequence()
        assert len(seq) > 0
