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


class TestDeuteriumCountsAsHydrogen:
    """A neutron structure writes D where an X-ray one writes H.

    is_protein_atom dropped both, is_hydrogen dropped only H, and the
    standardizer asked is_hydrogen -- so remove_hydrogens=True wrote a file
    still carrying every deuterium, beside the hydrogens it had removed.
    """

    @staticmethod
    def _line(serial, name, res, element, record="ATOM  ", num=1):
        row = list(" " * 80)
        row[0:6] = record.ljust(6)
        row[6:11] = f"{serial:5d}"
        row[12:16] = (" " + name).ljust(4)[:4]
        row[17:20] = res.rjust(3)[:3]
        row[21] = "A"
        row[22:26] = f"{num:4d}"
        row[30:38] = f"{serial:8.3f}"
        row[38:46] = f"{0.0:8.3f}"
        row[46:54] = f"{0.0:8.3f}"
        row[54:60] = "  1.00"
        row[60:66] = "  0.00"
        row[76:78] = element.rjust(2)
        return "".join(row).rstrip() + "\n"

    def test_is_hydrogen_says_so(self):
        assert is_hydrogen(self._line(1, "D1", "LIG", "D"))
        assert is_hydrogen(self._line(1, "H1", "LIG", "H"))
        assert not is_hydrogen(self._line(1, "C1", "LIG", "C"))

    def test_removing_hydrogens_removes_deuterium(self, tmp_path):
        """A ligand is the case that shows it: a standard residue's atom list
        would have dropped the D whatever the rule said."""
        from plmol.protein.pdb_standardizer import PDBStandardizer

        text = "".join(self._line(i, n, "ALA", e) for i, (n, e) in
                       enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")], 1))
        text += "".join(self._line(10 + i, n, "LIG", e, "HETATM", 900) for i, (n, e) in
                        enumerate([("C1", "C"), ("O1", "O"), ("H1", "H"), ("D1", "D")], 1))
        source = tmp_path / "neutron.pdb"
        source.write_text(text + "END\n")

        stripped = tmp_path / "stripped.pdb"
        PDBStandardizer(remove_hydrogens=True).standardize(str(source), str(stripped))
        names = [l[12:16].strip() for l in open(stripped)
                 if l[:6].strip() in ("ATOM", "HETATM")]
        assert "H1" not in names and "D1" not in names

        kept = tmp_path / "kept.pdb"
        PDBStandardizer(remove_hydrogens=False).standardize(str(source), str(kept))
        names = [l[12:16].strip() for l in open(kept)
                 if l[:6].strip() in ("ATOM", "HETATM")]
        assert "H1" in names and "D1" in names, "keeping hydrogens keeps both"


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
        assert parser.get_sequence() == "AYKG", "PTR is a tyrosine, LLP a lysine"

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
        # 20 in the file, 18 here: standardising maps PTR onto TYR and LLP onto
        # LYS, and neither parent has the atom the modification adds. That is
        # the PTM-to-parent behaviour the mapping asks for; what matters is
        # that nothing downstream drops more.
        assert atoms == 18
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


class TestInsertionCodes:
    """100, 100A and 100B are three residues.

    Every antibody numbered the Kabat or Chothia way has them -- 52A in CDR-L2,
    82A/B/C in the heavy framework, 100A through 100K in CDR-H3. plmol keyed
    residues on chain and number alone in four places, so the three collapsed
    into one: the parser reported three residues where the file has five and
    dropped two letters from the sequence, and the SASA grouping piled their
    areas together and warned about it.
    """

    @staticmethod
    def _structure(tmp_path, residue_names):
        def line(serial, name, res, resnum, icode, xyz, element):
            row = list(" " * 80)
            row[0:6] = "ATOM  "
            row[6:11] = f"{serial:5d}"
            row[12:16] = (" " + name).ljust(4)[:4]
            row[17:20] = res.rjust(3)[:3]
            row[21] = "A"
            row[22:26] = f"{resnum:4d}"
            row[26] = icode
            row[30:38] = f"{xyz[0]:8.3f}"
            row[38:46] = f"{xyz[1]:8.3f}"
            row[46:54] = f"{xyz[2]:8.3f}"
            row[54:60] = "  1.00"
            row[60:66] = "  0.00"
            row[76:78] = element.rjust(2)
            return "".join(row).rstrip() + "\n"

        backbone = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")]
        numbering = [(100, " "), (100, "A"), (100, "B"), (101, " "), (102, " ")]
        text, serial = "", 1
        for k, ((num, icode), res) in enumerate(zip(numbering, residue_names)):
            for j, (atom_name, element) in enumerate(backbone):
                if res == "GLY" and atom_name == "CB":
                    continue
                text += line(serial, atom_name, res, num, icode,
                             (k * 3.8 + j * 0.4, 0.0, 0.0), element)
                serial += 1
        path = tmp_path / "insertion.pdb"
        path.write_text(text + "END\n")
        return str(path)

    #: The hard case is three residues of the *same* type: without the code
    #: they are indistinguishable, so nothing downstream can separate them.
    SAME = ["SER", "SER", "SER", "VAL", "LEU"]
    MIXED = ["ALA", "GLY", "SER", "VAL", "LEU"]

    @pytest.mark.parametrize("names,sequence",
                             [(SAME, "SSSVL"), (MIXED, "AGSVL")],
                             ids=["same residue type", "different types"])
    def test_the_parser_counts_them_separately(self, tmp_path, names, sequence):
        from plmol.parsers import PDBParser

        parser = PDBParser(self._structure(tmp_path, names), skip_cache=True)
        assert len(parser.residues) == 5
        assert parser.get_sequence() == sequence
        assert len(parser.get_residue_list()) == 5

    @pytest.mark.parametrize("standardize", [True, False])
    @pytest.mark.parametrize("names,sequence",
                             [(SAME, "SSSVL"), (MIXED, "AGSVL")],
                             ids=["same residue type", "different types"])
    def test_the_residue_graph_keeps_all_five(self, tmp_path, standardize, names, sequence):
        from plmol import Protein

        protein = Protein.from_pdb(self._structure(tmp_path, names), standardize=standardize)
        assert np.asarray(protein.featurize(mode="graph")["graph"]["coords"]).shape[0] == 5
        assert protein.featurize(mode="sequence")["sequence"] == sequence

    def test_the_sasa_grouping_separates_them(self, tmp_path):
        """A mismatch here used to fire a warning and truncate the block."""
        from plmol.sasa import native_structure_result

        _, result = native_structure_result(self._structure(tmp_path, self.SAME))
        per_residue = [v for chain in result.residueAreas().values() for v in chain.values()]
        assert len(per_residue) == 5

    def test_they_come_out_in_sequence_order(self, tmp_path):
        """100 before 100A before 100B, not sorted by residue type."""
        from plmol.protein.residue_featurizer import ResidueFeaturizer

        featurizer = ResidueFeaturizer(self._structure(tmp_path, self.MIXED))
        codes = [residue[3] for residue in featurizer.get_residues()]
        assert codes == ["", "A", "B", "", ""]


class TestAlternateConformations:
    """An alternate location is a competing position for one atom, not a
    second atom.

    A structure refined at high resolution models a side chain in two or three
    places, tagged A/B/C in column 17 and weighted by occupancy. plmol read
    none of that: the parser kept every one, so a serine had two CB atoms two
    Angstrom apart, and the standardizer kept whichever came last -- which is
    the *minor* conformer, since A is written before B.
    """

    @staticmethod
    def _structure(tmp_path, occupancies=(0.60, 0.40)):
        def line(serial, name, altloc, res, resnum, xyz, element, occ):
            row = list(" " * 80)
            row[0:6] = "ATOM  "
            row[6:11] = f"{serial:5d}"
            row[12:16] = (" " + name).ljust(4)[:4]
            row[16] = altloc
            row[17:20] = res.rjust(3)[:3]
            row[21] = "A"
            row[22:26] = f"{resnum:4d}"
            row[30:38] = f"{xyz[0]:8.3f}"
            row[38:46] = f"{xyz[1]:8.3f}"
            row[46:54] = f"{xyz[2]:8.3f}"
            row[54:60] = f"{occ:6.2f}"
            row[60:66] = "  0.00"
            row[76:78] = element.rjust(2)
            return "".join(row).rstrip() + "\n"

        text, serial = "", 1
        for i, res in enumerate(["ALA", "SER", "ALA"], 1):
            for j, (name, element) in enumerate(
                [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
            ):
                text += line(serial, name, " ", res, i,
                             (i * 3.8 + j * 0.4, 0.0, 0.0), element, 1.0)
                serial += 1
            if res == "SER":
                for altloc, side, occ in (("A", 1.0, occupancies[0]),
                                          ("B", -1.0, occupancies[1])):
                    text += line(serial, "CB", altloc, res, i,
                                 (i * 3.8, side, 0.0), "C", occ); serial += 1
                    text += line(serial, "OG", altloc, res, i,
                                 (i * 3.8, side * 2, 0.0), "O", occ); serial += 1
            else:
                text += line(serial, "CB", " ", res, i,
                             (i * 3.8, 1.0, 0.0), "C", 1.0); serial += 1
        path = tmp_path / "altloc.pdb"
        path.write_text(text + "END\n")
        return str(path)

    def test_the_parser_keeps_one_position_per_atom(self, tmp_path):
        from plmol.parsers import PDBParser

        parser = PDBParser(self._structure(tmp_path), skip_cache=True)
        assert len(parser.protein_atoms) == 16, "18 lines, two of them alternates"
        names = [a.atom_name for a in parser.protein_atoms if a.res_num == 2]
        assert names == ["N", "CA", "C", "O", "CB", "OG"]

    @pytest.mark.parametrize("occupancies,expected_sign",
                             [((0.60, 0.40), +1.0), ((0.40, 0.60), -1.0)],
                             ids=["A is major", "B is major"])
    def test_the_higher_occupancy_wins(self, tmp_path, occupancies, expected_sign):
        from plmol.parsers import PDBParser

        parser = PDBParser(self._structure(tmp_path, occupancies), skip_cache=True)
        cb = [a for a in parser.protein_atoms if a.res_num == 2 and a.atom_name == "CB"]
        assert len(cb) == 1
        assert np.sign(cb[0].coords[1]) == expected_sign

    @pytest.mark.parametrize("occupancies,expected_sign",
                             [((0.60, 0.40), +1.0), ((0.40, 0.60), -1.0)],
                             ids=["A is major", "B is major"])
    def test_standardizing_keeps_the_same_one(self, tmp_path, occupancies, expected_sign):
        from plmol.protein.pdb_standardizer import PDBStandardizer

        source = self._structure(tmp_path, occupancies)
        out = str(tmp_path / "standardized.pdb")
        PDBStandardizer().standardize(source, out)
        ys = [float(line[38:46]) for line in open(out)
              if line.startswith("ATOM") and line[17:20].strip() == "SER"
              and line[12:16].strip() == "CB"]
        assert len(ys) == 1
        assert np.sign(ys[0]) == expected_sign

    def test_the_atom_graph_has_no_duplicate_atom(self, tmp_path):
        from plmol import Protein

        atom_graph = Protein.from_pdb(self._structure(tmp_path)).featurize(
            mode="atom_graph")["atom_graph"]
        coords = np.asarray(atom_graph["coords"])
        assert coords.shape[0] == 16
        distances = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() > 0.1, "two conformers of one atom sit almost on top"

    def test_both_parsers_choose_the_same_conformer(self, tmp_path, example_pdb):
        """The rule is one function, so a structure read as mmCIF keeps what it
        keeps as PDB. The mmCIF parser used to keep every conformer."""
        gemmi = pytest.importorskip("gemmi")
        from plmol.parsers import MMCIFParser, PDBParser

        lines = [l for l in open(example_pdb) if l[:6].strip() in ("ATOM", "HETATM")]
        altered = tmp_path / "alternates.pdb"
        with open(altered, "w") as handle:
            for line in lines:
                if line[17:20].strip() == "SER" and line[12:16].strip() == "OG":
                    major = list(line); major[16] = "A"; major[54:60] = list(f"{0.70:6.2f}")
                    minor = list(line); minor[16] = "B"; minor[54:60] = list(f"{0.30:6.2f}")
                    minor[38:46] = list(f"{float(line[38:46]) + 1.5:8.3f}")
                    handle.write("".join(major)); handle.write("".join(minor))
                else:
                    handle.write(line)
            handle.write("END\n")

        structure = gemmi.read_structure(str(altered))
        structure.setup_entities()
        cif = tmp_path / "alternates.cif"
        structure.make_mmcif_document().write_file(str(cif))

        from_pdb = PDBParser(str(altered), skip_cache=True).protein_atoms
        from_cif = MMCIFParser(str(cif), include_nucleic_acids=False).protein_atoms
        assert len(from_pdb) == len(from_cif) == 3260
        key = lambda a: (a.chain_id, a.res_num, a.atom_name, round(a.coords[1], 3))  # noqa: E731
        assert {key(a) for a in from_pdb} == {key(a) for a in from_cif}

    @staticmethod
    def _alt(name, alt_loc, occupancy, res_num=1, y=0.0):
        from plmol.parsers.pdb_parser import ParsedAtom

        return ParsedAtom(record_type="ATOM", atom_name=name, res_name="SER",
                          res_num=res_num, chain_id="A", coords=(0.0, y, 0.0),
                          element="C", alt_loc=alt_loc, occupancy=occupancy)

    @pytest.mark.parametrize("occupancies,winner", [
        ((0.6, 0.4), "A"), ((0.4, 0.6), "B"),
        ((0.5, 0.5), "A"),          # a tie goes to the one written first
        ((1.0, 1.0), "A"),          # which is what an NMR ensemble looks like
    ], ids=["A major", "B major", "tie", "both full"])
    def test_the_tie_goes_to_the_first(self, occupancies, winner):
        from plmol.parsers.pdb_parser import best_conformers

        kept = best_conformers([self._alt("OG", "A", occupancies[0], y=1.0),
                                self._alt("OG", "B", occupancies[1], y=-1.0)])
        assert len(kept) == 1
        assert kept[0].alt_loc == winner

    def test_three_alternates(self):
        from plmol.parsers.pdb_parser import best_conformers

        kept = best_conformers([self._alt("OG", "A", 0.3), self._alt("OG", "B", 0.5),
                                self._alt("OG", "C", 0.2)])
        assert [a.alt_loc for a in kept] == ["B"]

    def test_file_order_survives_a_replacement(self):
        """When the minor conformer is written first the winner takes its slot,
        so atoms do not get reordered across residues -- the hierarchical
        atom-to-residue mapping reads them positionally."""
        from plmol.parsers.pdb_parser import best_conformers

        atoms = []
        for res_num in range(1, 6):
            atoms.append(self._alt("N", "", 1.0, res_num=res_num))
            atoms.append(self._alt("CB", "A", 0.4, res_num=res_num, y=1.0))
            atoms.append(self._alt("CB", "B", 0.6, res_num=res_num, y=-1.0))
        kept = best_conformers(atoms)
        assert len(kept) == 10
        numbers = [a.res_num for a in kept]
        assert numbers == sorted(numbers)
        assert [a.atom_name for a in kept[:2]] == ["N", "CB"]

    def test_an_atom_without_an_alternate_is_untouched(self, tmp_path):
        """Only atoms carrying a code go through the rule."""
        from plmol.parsers import PDBParser

        parser = PDBParser(self._structure(tmp_path), skip_cache=True)
        plain = [a for a in parser.protein_atoms if not a.alt_loc]
        assert len(plain) == 14
        assert all(a.occupancy == 1.0 for a in plain)


class TestMultiModelEnsembles:
    """An NMR file stacks its models between MODEL and ENDMDL.

    They are alternatives, not more structure. plmol read every one: a
    three-model ensemble gave every atom three times, which piled 21 atoms
    into a residue array 15 slots wide and raised a bare ValueError with
    standardize=False, and let each atom occlude its own copies when the SASA
    was sampled.
    """

    @staticmethod
    def _ensemble(tmp_path, example_pdb, offsets=(0.0, 10.0, 20.0)):
        lines = [l for l in open(example_pdb) if l[:6].strip() in ("ATOM", "HETATM")]
        path = tmp_path / "ensemble.pdb"
        with open(path, "w") as handle:
            for model, dz in enumerate(offsets, 1):
                handle.write(f"MODEL     {model:4d}\n")
                handle.writelines(
                    line[:46] + f"{float(line[46:54]) + dz:8.3f}" + line[54:]
                    for line in lines
                )
                handle.write("ENDMDL\n")
            handle.write("END\n")
        return str(path)

    def test_only_the_first_model_is_read(self, tmp_path, example_pdb):
        from plmol.parsers import PDBParser

        one = PDBParser(example_pdb, skip_cache=True)
        many = PDBParser(self._ensemble(tmp_path, example_pdb), skip_cache=True)
        assert len(many.protein_atoms) == len(one.protein_atoms)
        assert many.get_sequence() == one.get_sequence()

    @pytest.mark.parametrize("standardize", [True, False])
    def test_featurizing_an_ensemble_gives_the_first_model(
        self, tmp_path, example_pdb, standardize
    ):
        """The models here are shifted in z, so the mean says which one was
        taken. It has to be the first: that is the one the PDB nominates."""
        from plmol import Protein

        ensemble = self._ensemble(tmp_path, example_pdb)
        reference = np.asarray(
            Protein.from_pdb(example_pdb).featurize(mode="atom_graph")["atom_graph"]["coords"]
        )
        got = np.asarray(
            Protein.from_pdb(ensemble, standardize=standardize)
            .featurize(mode="atom_graph")["atom_graph"]["coords"]
        )
        assert got.shape == reference.shape
        assert abs(float(got[:, 2].mean()) - float(reference[:, 2].mean())) < 1e-3

    def test_a_file_without_model_records_is_untouched(self, example_pdb):
        from plmol.parsers import PDBParser

        assert len(PDBParser(example_pdb, skip_cache=True).protein_atoms) == 3260
