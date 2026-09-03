"""The constants have to agree with each other.

Every table here is a mapping some featurizer reads with a ``.get(key,
default)``. A key missing from one table and present in another does not
raise -- it silently returns the default, and a manganese comes out with
carbon's van der Waals radius. These pin the chains that must not have holes.
"""

import pytest
from rdkit import Chem

import plmol.constants as K

#: Every element plmol claims to handle: the ligand featurizer's own list,
#: plus mercury, which the PDB parser recognises as a monatomic ion.
DECLARED_ELEMENTS = [s for s in K.ATOM_TYPES if s != "UNK"] + ["Hg"]

#: Tables keyed by atomic number that a featurizer reads per atom.
BY_ATOMIC_NUMBER = {
    "VDW_RADIUS": K.VDW_RADIUS,
    "COVALENT_RADIUS": K.COVALENT_RADIUS,
    "IONIZATION_ENERGY": K.IONIZATION_ENERGY,
    "POLARIZABILITY": K.POLARIZABILITY,
    "VALENCE_ELECTRONS": K.VALENCE_ELECTRONS,
    "ATOMIC_MASS": K.ATOMIC_MASS,
}


def _atomic_number(symbol):
    return Chem.GetPeriodicTable().GetAtomicNumber(symbol)


class TestEveryDeclaredElementIsFullyMapped:
    @pytest.mark.parametrize("table", sorted(BY_ATOMIC_NUMBER))
    def test_no_declared_element_falls_back_to_a_default(self, table):
        missing = [s for s in DECLARED_ELEMENTS
                   if _atomic_number(s) not in BY_ATOMIC_NUMBER[table]]
        assert missing == [], f"{table} has no entry for {missing}"

    def test_every_declared_element_has_an_atomic_number(self):
        missing = [s for s in DECLARED_ELEMENTS
                   if s.upper() not in K.ELEMENT_SYMBOL_TO_ATOMIC_NUMBER]
        assert missing == []


class TestTheMetalChain:
    """METAL_RESIDUES -> METAL_ELEMENTS -> atomic number -> radius.

    A metal named in one link and absent from the next is a metal that
    featurizes as carbon.
    """

    def test_the_residue_and_element_sets_match(self):
        assert set(K.METAL_RESIDUES) == set(K.METAL_ELEMENTS)

    def test_every_metal_has_an_atomic_number(self):
        assert set(K.METAL_ELEMENTS) <= set(K.ELEMENT_SYMBOL_TO_ATOMIC_NUMBER)

    @pytest.mark.parametrize("table", ["VDW_RADIUS", "COVALENT_RADIUS"])
    def test_every_metal_has_a_radius(self, table):
        numbers = {K.ELEMENT_SYMBOL_TO_ATOMIC_NUMBER[m] for m in K.METAL_ELEMENTS}
        assert numbers <= set(BY_ATOMIC_NUMBER[table])

    def test_every_metal_can_be_asked_for_a_donor_preference(self):
        assert set(K.METAL_PREFERRED_DONORS) == set(K.METAL_RESIDUES)


class TestAtomicNumbersResolve:
    def test_every_symbol_maps_to_the_right_number(self):
        table = Chem.GetPeriodicTable()
        wrong = {s: z for s, z in K.ELEMENT_SYMBOL_TO_ATOMIC_NUMBER.items()
                 if table.GetAtomicNumber(s.capitalize()) != z}
        assert wrong == {}

    def test_the_keys_are_upper_case(self):
        """The PDB parser uppercases what it reads from columns 77-78, so this
        table is looked up with upper-case keys."""
        assert all(k == k.upper() for k in K.ELEMENT_SYMBOL_TO_ATOMIC_NUMBER)

    def test_every_entry_has_a_van_der_waals_radius(self):
        assert set(K.ELEMENT_SYMBOL_TO_ATOMIC_NUMBER.values()) <= set(K.VDW_RADIUS)


class TestCovalentRadiiAreRdkits:
    """They agreed on all 24 entries before mercury and the three metals were
    added, so RDKit is the source and new entries come from it."""

    def test_they_match_rdkit(self):
        table = Chem.GetPeriodicTable()
        off = {z: (r, table.GetRcovalent(z)) for z, r in K.COVALENT_RADIUS.items()
               if abs(r - table.GetRcovalent(z)) > 0.005}
        assert off == {}


class TestAtomicMassesAreRdkits:
    def test_they_match_rdkit(self):
        table = Chem.GetPeriodicTable()
        off = {z: (m, table.GetAtomicWeight(z)) for z, m in K.ATOMIC_MASS.items()
               if abs(m - table.GetAtomicWeight(z)) > 0.01}
        assert off == {}


class TestResidueTablesCoverTheTwenty:
    TWENTY = set(K.AMINO_ACID_LETTERS)

    @pytest.mark.parametrize("table", [
        "AMINO_ACID_3TO1", "RESIDUE_TOKEN", "STANDARD_ATOMS",
        "RESIDUE_MAX_SASA", "RESIDUE_MAX_CLASS_SASA", "RESIDUE_PROPERTIES",
    ])
    def test_no_standard_residue_is_missing(self, table):
        assert self.TWENTY <= set(getattr(K, table))

    def test_every_standard_atom_has_a_token(self):
        missing = [(res, atom) for res, atoms in K.STANDARD_ATOMS.items()
                   for atom in atoms if (res, atom) not in K.RESIDUE_ATOM_TOKEN]
        assert missing == []

    def test_every_standard_atom_name_maps_to_an_element(self):
        names = {a for atoms in K.STANDARD_ATOMS.values() for a in atoms}
        assert names <= set(K.ATOM_NAME_TO_ELEMENT)

    def test_the_name_mapping_lands_on_something_real(self):
        """Capping groups map to themselves; everything else must reach a
        standard residue, UNK or METAL."""
        caps = {"ACE", "NH2", "NME"}
        assert set(K.RESIDUE_NAME_MAPPING.values()) <= self.TWENTY | caps | {"UNK", "METAL"}


class TestTheModifiedResidueTablesStayInStep:
    """RESIDUE_NAME_MAPPING says what a modified residue becomes;
    PTM_RESIDUES says which entries are modifications rather than protonation
    states, and that is what ptm_handling switches on. A name in one and not
    the other is a residue one of the four modes cannot see."""

    def test_every_ptm_has_a_parent(self):
        assert set(K.PTM_RESIDUES) <= set(K.RESIDUE_NAME_MAPPING)

    def test_every_ptm_maps_to_a_standard_residue(self):
        parents = {K.RESIDUE_NAME_MAPPING[p] for p in K.PTM_RESIDUES}
        assert parents <= set(K.AMINO_ACID_LETTERS)

    def test_the_preserved_atom_lists_name_a_known_ptm(self):
        assert set(K.STANDARD_ATOMS_PTM) <= set(K.PTM_RESIDUES)


class TestInteractionTablesAgree:
    def test_the_type_index_is_dense_and_matches_its_count(self):
        assert sorted(K.INTERACTION_TYPE_IDX.values()) == list(range(K.NUM_INTERACTION_TYPES))
        assert set(K.INTERACTION_TYPE_IDX) == set(K.INTERACTION_TYPES)

    def test_the_pharmacophore_index_is_dense_and_matches_its_count(self):
        assert sorted(K.PHARMACOPHORE_IDX.values()) == list(range(K.NUM_PHARMACOPHORE_TYPES))

    def test_every_interaction_type_has_an_ideal_distance(self):
        assert set(K.INTERACTION_TYPE_IDX) <= set(K.IDEAL_DISTANCES)

    def test_compatibility_is_pharmacophore_pairs_to_interaction_types(self):
        pharmacophores, types = set(K.PHARMACOPHORE_IDX), set(K.INTERACTION_TYPE_IDX)
        for pair, interaction in K.INTERACTION_COMPATIBILITY.items():
            assert set(pair) <= pharmacophores, pair
            assert interaction in types, interaction

    def test_compatibility_is_symmetric(self):
        """The table is consulted with whichever endpoint comes first, so a
        pair listed one way round has to be listed the other."""
        missing = [(a, b) for (a, b) in K.INTERACTION_COMPATIBILITY
                   if (b, a) not in K.INTERACTION_COMPATIBILITY]
        assert missing == []


class TestNucleotideTablesCoverDnaAndRna:
    RESIDUES = set(K.DNA_RESIDUES) | set(K.RNA_RESIDUES)

    @pytest.mark.parametrize("table", [
        "NUCLEOTIDE_TOKEN", "NUCLEOTIDE_MAX_SASA", "BASE_ATOMS",
        "STANDARD_NUCLEOTIDE_ATOMS", "NUCLEOTIDE_PROPERTIES", "NUCLEIC_ACID_RESIDUES",
    ])
    def test_no_nucleotide_is_missing(self, table):
        assert self.RESIDUES <= set(getattr(K, table))

    def test_every_nucleotide_is_a_purine_or_a_pyrimidine(self):
        assert self.RESIDUES <= set(K.PURINES) | set(K.PYRIMIDINES)


class TestTheNucleotideNameMapping:
    """Its counterpart on the protein side is RESIDUE_NAME_MAPPING; both must
    land on a residue the rest of the library has a row for."""

    def test_every_modified_base_maps_to_a_canonical_one(self):
        canonical = set(K.DNA_RESIDUES) | set(K.RNA_RESIDUES)
        assert set(K.NUCLEOTIDE_NAME_MAPPING.values()) <= canonical

    def test_every_legacy_spelling_offers_an_rna_and_a_dna_form(self):
        canonical = set(K.DNA_RESIDUES) | set(K.RNA_RESIDUES)
        for name, forms in K.LEGACY_BASE_NAMES.items():
            assert len(forms) == 2, name
            assert set(forms) <= canonical, name

    def test_the_mapped_names_are_names_the_parser_keeps(self):
        known = set(K.NUCLEIC_ACID_RESIDUES)
        assert set(K.NUCLEOTIDE_NAME_MAPPING) <= known
        assert set(K.LEGACY_BASE_NAMES) <= known
        assert set(K.UNMAPPED_NUCLEOTIDES) <= known

    def test_nothing_is_both_mapped_and_deliberately_unmapped(self):
        mapped = set(K.NUCLEOTIDE_NAME_MAPPING) | set(K.LEGACY_BASE_NAMES)
        assert mapped & set(K.UNMAPPED_NUCLEOTIDES) == set()

    def test_every_nucleic_name_is_accounted_for(self):
        """Canonical, mapped, legacy, or explicitly left alone -- nothing may
        sit in NUCLEIC_ACID_RESIDUES with no decision recorded about it."""
        canonical = set(K.DNA_RESIDUES) | set(K.RNA_RESIDUES)
        accounted = (canonical | set(K.NUCLEOTIDE_NAME_MAPPING)
                     | set(K.LEGACY_BASE_NAMES) | set(K.UNMAPPED_NUCLEOTIDES))
        assert set(K.NUCLEIC_ACID_RESIDUES) - accounted == set()


class TestHydrogenBondTablesNameAtomsThatExist:
    @pytest.mark.parametrize("table", [
        "HBOND_DONOR_ATOMS_BY_RESIDUE", "HBOND_ACCEPTOR_ATOMS_BY_RESIDUE",
        "AROMATIC_RING_ATOMS", "POS_IONIZABLE_ATOMS", "NEG_IONIZABLE_ATOMS",
    ])
    def test_the_atoms_belong_to_the_residue(self, table):
        wrong = [(res, atom) for res, atoms in getattr(K, table).items()
                 if res in K.STANDARD_ATOMS
                 for atom in atoms if atom not in K.STANDARD_ATOMS[res]]
        assert wrong == []
