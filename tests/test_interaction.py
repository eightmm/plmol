"""Tests for plmol/interaction/ — PLInteractionFeaturizer + PocketExtractor."""

import numpy as np
import pytest

from rdkit import Chem

from plmol.interaction.pli_featurizer import PLInteractionFeaturizer
from plmol.interaction.pocket_extractor import PocketExtractor, extract_pocket


@pytest.fixture
def protein_ligand_mols(example_pdb, example_sdf):
    """Load protein and ligand as RDKit Mol objects."""
    protein_mol = Chem.MolFromPDBFile(example_pdb, removeHs=True)
    suppl = Chem.SDMolSupplier(example_sdf)
    ligand_mol = next(suppl)
    return protein_mol, ligand_mol


class TestPLInteractionFeaturizer:
    def test_init(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        assert plif.num_protein_atoms > 0
        assert plif.num_ligand_atoms > 0

    def test_detect_all_interactions(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        interactions = plif.detect_all_interactions()
        assert isinstance(interactions, list)
        # 10gs has known interactions
        assert len(interactions) > 0

    def test_get_interaction_edges(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        edge_index, edge_features = plif.get_interaction_edges()
        assert edge_index.shape[0] == 2
        assert edge_features.shape[0] == edge_index.shape[1]
        assert edge_features.shape[1] == 79

    def test_get_interaction_graph(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        graph = plif.get_interaction_graph()
        assert isinstance(graph, dict)
        assert "edges" in graph
        assert "edge_features" in graph

    def test_get_heavy_atom_coords(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        p_coords, l_coords = plif.get_heavy_atom_coords()
        assert isinstance(p_coords, np.ndarray)
        assert isinstance(l_coords, np.ndarray)
        assert p_coords.shape[1] == 3
        assert l_coords.shape[1] == 3

    def test_get_atom_pharmacophore_features(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        p_feats, l_feats = plif.get_atom_pharmacophore_features()
        assert p_feats.shape[0] == plif.num_protein_atoms
        assert l_feats.shape[0] == plif.num_ligand_atoms

    def test_get_atom_chemical_features(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        p_feats, l_feats = plif.get_atom_chemical_features()
        assert p_feats.shape[0] == plif.num_protein_atoms
        assert l_feats.shape[0] == plif.num_ligand_atoms

    def test_get_distance_based_edges(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        edge_index, edge_features = plif.get_distance_based_edges()
        assert edge_index.shape[0] == 2

    def test_get_interaction_summary(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        summary = plif.get_interaction_summary()
        assert isinstance(summary, str)

    def test_get_feature_description(self, protein_ligand_mols):
        protein_mol, ligand_mol = protein_ligand_mols
        plif = PLInteractionFeaturizer(protein_mol, ligand_mol)
        desc = plif.get_feature_description()
        assert isinstance(desc, dict)


class TestPocketExtractor:
    def test_from_files(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        pocket = pe.extract()
        assert pocket is not None
        assert pocket.pocket_mol is not None

    def test_from_protein(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_protein(example_pdb)
        ligand_mol = next(Chem.SDMolSupplier(example_sdf))
        pocket = pe.extract_for_ligand(ligand_mol)
        assert pocket is not None

    def test_get_pocket_pdb_block(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        pdb_block = pe.get_pocket_pdb_block()
        assert isinstance(pdb_block, str)
        assert "ATOM" in pdb_block

    def test_get_pocket_residue_mask(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        mask = pe.get_pocket_residue_mask()
        assert mask.dtype == bool
        assert mask.any()

    def test_get_residue_distances(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        dists = pe.get_residue_distances()
        assert dists.ndim == 1
        assert (dists >= 0).all()

    def test_save_pocket_pdb(self, example_pdb, example_sdf, tmp_path):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        output = str(tmp_path / "pocket.pdb")
        pe.save_pocket_pdb(output)
        import os
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_properties(self, example_pdb, example_sdf):
        pe = PocketExtractor.from_files(example_pdb, example_sdf)
        assert pe.num_residues > 0
        assert len(pe.residue_keys) > 0


class TestExtractPocket:
    def test_convenience_function(self, example_pdb, example_sdf):
        ligand_mol = next(Chem.SDMolSupplier(example_sdf))
        pockets = extract_pocket(example_pdb, ligand_mol)
        assert isinstance(pockets, list)
        assert len(pockets) > 0


class TestPocketExtractorUsesTheSharedRules:
    """It kept a private metal set and a hydrogen rule of its own.

    The metal set was missing sodium and potassium, so a thrombin sodium site
    or a potassium channel lost its ion. The hydrogen rule read the atom name,
    so a hydrogen PDB numbered the pre-2007 way, 1HB, stayed in the pocket as
    a heavy atom.
    """

    @staticmethod
    def _structure(tmp_path):
        def line(serial, name, res, resnum, xyz, element, record="ATOM  "):
            row = list(" " * 80)
            row[0:6] = record.ljust(6)
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
            if element is None:                     # a file written without the column
                return "".join(row)[:66].rstrip() + "\n"
            row[76:78] = element.rjust(2)
            return "".join(row).rstrip() + "\n"

        atoms = [
            ("N", "ALA", 1, (0., 0., 0.), "N"), ("CA", "ALA", 1, (1.5, 0, 0), "C"),
            ("C", "ALA", 1, (2., 1.4, 0), "C"), ("O", "ALA", 1, (1.2, 2.4, 0), "O"),
            ("CB", "ALA", 1, (2., -0.8, -1.2), "C"),
            ("1HB", "ALA", 1, (3.1, -0.8, -1.2), None),
            ("N", "GLY", 2, (3.3, 1.5, 0), "N"), ("CA", "GLY", 2, (4., 2.8, 0), "C"),
            ("C", "GLY", 2, (5.5, 2.7, 0), "C"), ("O", "GLY", 2, (6.1, 1.6, 0), "O"),
        ]
        metals = [("ZN", "ZN", 301, (3., 3., 3.), "ZN"), ("NA", "NA", 302, (3.5, 3.5, 3.5), "NA"),
                  ("K", "K", 303, (4., 4., 4.), "K"), ("MG", "MG", 304, (2.5, 2.5, 2.5), "MG")]
        text = "".join(line(i, *a) for i, a in enumerate(atoms, 1))
        text += "".join(line(100 + i, *m, record="HETATM") for i, m in enumerate(metals, 1))
        path = tmp_path / "pocket.pdb"
        path.write_text(text + "END\n")
        return str(path)

    def _parsed(self, tmp_path):
        from plmol.interaction.pocket_extractor import PocketExtractor

        extractor = PocketExtractor(self._structure(tmp_path))
        extractor._parse_protein_pdb()
        return extractor

    def test_every_metal_the_library_knows_is_preserved(self, tmp_path):
        extractor = self._parsed(tmp_path)
        kept = {line[17:20].strip() for line in extractor._metal_lines}
        assert kept == {"ZN", "NA", "K", "MG"}

    def test_the_private_metal_set_is_the_shared_one(self):
        from plmol.constants import METAL_ELEMENTS
        from plmol.interaction import pocket_extractor

        assert pocket_extractor._METAL_ELEMENTS is METAL_ELEMENTS

    def test_no_hydrogen_survives_as_a_pocket_atom(self, tmp_path):
        extractor = self._parsed(tmp_path)
        names = [line[12:16].strip() for lines in extractor._residue_lines for line in lines]
        assert names == ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]


class TestSolventIsNotProtein:
    """Chem.MolFromPDBFile keeps crystallographic waters, and this class had no
    notion of them, so they arrived as protein atoms."""

    def test_waters_are_dropped_from_the_protein(self, example_pdb):
        from rdkit import Chem

        from plmol.interaction.pli_featurizer import _without_solvent

        mol = Chem.MolFromPDBFile(example_pdb, removeHs=False)
        before = mol.GetNumAtoms()
        after = _without_solvent(mol).GetNumAtoms()
        assert after < before
        remaining = {
            atom.GetPDBResidueInfo().GetResidueName().strip()
            for atom in _without_solvent(mol).GetAtoms()
            if atom.GetPDBResidueInfo() is not None
        }
        assert "HOH" not in remaining

    def test_a_mol_without_waters_is_handed_back_unchanged(self):
        from rdkit import Chem

        from plmol.interaction.pli_featurizer import _without_solvent

        mol = Chem.MolFromSmiles("CCO")
        assert _without_solvent(mol) is mol

    def test_none_survives(self):
        from plmol.interaction.pli_featurizer import _without_solvent

        assert _without_solvent(None) is None

    def test_the_ligand_sees_only_protein_neighbours(self, example_pdb, example_sdf):
        """The cross-contact density counts what is near a ligand atom. With
        the solvent shell attached it counted waters, and on this complex 16 of
        33 ligand atoms came out too high."""
        from plmol import Ligand, Protein
        from plmol.complex import MolecularComplex

        result = MolecularComplex(
            molecules={"protein": Protein.from_pdb(example_pdb),
                       "ligand": Ligand.from_sdf(example_sdf)}
        ).featurize(requests=["interaction"])["interaction"]
        assert result["num_protein_atoms"] == 3262, "3431 with the waters counted"
        assert np.asarray(result["protein_coords"]).shape[0] == result["num_protein_atoms"]


class TestWhatDistanceCutoffActuallyBounds:
    """It reads like the interaction range and it is not.

    Every detector uses its own entry from INTERACTION_TYPES -- 3.5 A for a
    hydrogen bond, 6.0 for a cation-pi -- and distance_cutoff bounds the
    optional contact edges. Three docstrings said otherwise until 0.4.x, so a
    caller raising it saw nothing change and had no way to know why.
    """

    @staticmethod
    def _run(example_pdb, example_sdf, cutoff):
        from plmol import Ligand, Protein
        from plmol.complex import MolecularComplex

        return MolecularComplex(
            molecules={"protein": Protein.from_pdb(example_pdb),
                       "ligand": Ligand.from_sdf(example_sdf)}
        ).featurize(
            requests=["interaction"],
            interaction_kwargs={"distance_cutoff": cutoff, "include_contacts": True},
        )["interaction"]

    def test_the_pharmacophore_count_does_not_move(self, example_pdb, example_sdf):
        counts = {c: self._run(example_pdb, example_sdf, c)["num_interactions"]
                  for c in (3.5, 4.5, 8.0)}
        assert len(set(counts.values())) == 1, counts

    def test_the_contact_edges_do_move(self, example_pdb, example_sdf):
        narrow = self._run(example_pdb, example_sdf, 3.5)
        wide = self._run(example_pdb, example_sdf, 8.0)
        n = np.asarray(narrow["contact_edges"]).shape[1]
        w = np.asarray(wide["contact_edges"]).shape[1]
        assert w > n * 10, f"{n} -> {w}"

    def test_the_per_type_ranges_are_what_the_detectors_read(self):
        from plmol.constants import INTERACTION_TYPES

        ranges = {t: spec["distance_cutoff"] for t, spec in INTERACTION_TYPES.items()}
        assert ranges["hydrogen_bond"] == 3.5
        assert ranges["cation_pi"] == 6.0
        assert len(set(ranges.values())) > 1, "one range for all types would make the parameter honest"


class TestSubstructureMatchesAreNotCapped:
    """RDKit stops at 1000 matches by default and says nothing."""

    def test_the_helper_finds_what_the_default_truncates(self, example_pdb):
        from plmol.constants import PHARMACOPHORE_SMARTS
        from plmol.rdkit_utils import substructure_matches

        mol = Chem.MolFromPDBFile(example_pdb, removeHs=False)
        pattern = Chem.MolFromSmarts(PHARMACOPHORE_SMARTS["hydrophobic"])

        capped = mol.GetSubstructMatches(pattern)
        uncapped = substructure_matches(mol, pattern)
        assert len(capped) == 1000, "the default cap is what this guards against"
        assert len(uncapped) > 1000
        assert set(capped).issubset(set(uncapped))

    def test_the_recursive_cap_too(self, example_pdb):
        # An exclusion written !$(...) stops excluding once it runs out of
        # recursion budget, so the default both admits and drops atoms.
        from plmol.constants import PHARMACOPHORE_SMARTS
        from plmol.rdkit_utils import substructure_matches

        mol = Chem.MolFromPDBFile(example_pdb, removeHs=False)
        pattern = Chem.MolFromSmarts(PHARMACOPHORE_SMARTS["positive"])
        assert len(mol.GetSubstructMatches(pattern)) < len(substructure_matches(mol, pattern))


class TestInteractionsDoNotDependOnFileOrder:
    """The same complex, written with its chains the other way round."""

    @staticmethod
    def _chains_reversed(source: str, target) -> str:
        head, body, tail, seen = [], [], [], False
        for line in open(source):
            line = line.rstrip("\n")
            if line.startswith(("ATOM  ", "HETATM")):
                body.append(line)
                seen = True
            elif seen:
                tail.append(line)
            else:
                head.append(line)
        by_chain: dict = {}
        for line in body:
            by_chain.setdefault(line[21], []).append(line)
        out = [l for chain in reversed(list(by_chain)) for l in by_chain[chain]]
        renumbered = [l[:6] + f"{i + 1:5d}" + l[11:] for i, l in enumerate(out)]
        target.write_text("\n".join(head + renumbered + tail) + "\nEND\n")
        return str(target)

    def test_the_same_interactions_are_found(self, example_pdb, example_sdf, tmp_path):
        from plmol import Ligand, Protein
        from plmol.complex import MolecularComplex
        from plmol.parsers.pdb_parser import PDBParser

        reordered = self._chains_reversed(example_pdb, tmp_path / "b_first.pdb")

        def counts(path):
            PDBParser.clear_cache()
            complex_ = MolecularComplex(molecules={
                "protein": Protein.from_pdb(path),
                "ligand": Ligand.from_sdf(example_sdf),
            })
            return complex_.featurize(requests=["interaction"])["interaction"]["interaction_counts"]

        assert counts(example_pdb) == counts(reordered)


class TestPocketKeepsInsertionCodedResidues:
    """100 and 100A are two residues; the pocket used to keep only the first."""

    @staticmethod
    def _tryptophans(path, numbering):
        names = ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1",
                 "CE2", "CE3", "CZ2", "CZ3", "CH2"]
        lines = []
        for step, (number, icode) in enumerate(numbering):
            for i, name in enumerate(names):
                x, y, z = i * 0.4, step * 3.0, 0.0
                lines.append(
                    f"ATOM      1 {name:^4s} TRP A{number:4d}{icode:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {name[0]}"
                )
        renumbered = [l[:6] + f"{i + 1:5d}" + l[11:] for i, l in enumerate(lines)]
        path.write_text("\n".join(renumbered) + "\nEND\n")
        return str(path)

    def test_no_residue_swallows_another(self, tmp_path):
        from plmol.interaction.pocket_extractor import PocketExtractor

        path = self._tryptophans(tmp_path / "icodes.pdb",
                                 [(100, " "), (100, "A"), (101, " ")])
        extractor = PocketExtractor.from_protein(path)

        assert len(extractor._residue_keys) == 3
        assert [key[3] for key in extractor._residue_keys] == ["", "A", ""]
        kept = (~np.isnan(extractor._residue_coords[:, :, 0])).sum()
        assert kept == 42, "three tryptophans, fourteen heavy atoms each"

    def test_an_alternate_conformation_is_not_an_extra_atom(self, tmp_path):
        # A side chain refined in two conformations is written twice. Both
        # copies used to count toward the fourteen-atom cap, so a tryptophan
        # reached it on NE1 and lost the rest of its indole ring.
        from plmol.interaction.pocket_extractor import PocketExtractor

        names = ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1",
                 "CE2", "CE3", "CZ2", "CZ3", "CH2"]
        lines = []
        for i, name in enumerate(names):
            if i < 4:
                lines.append(
                    f"ATOM      1 {name:^4s} TRP A 100    "
                    f"{i * 0.4:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           {name[0]}"
                )
                continue
            for alt, occupancy, z in (("A", 0.60, 0.0), ("B", 0.40, 2.0)):
                lines.append(
                    f"ATOM      1 {name:^4s}{alt}TRP A 100    "
                    f"{i * 0.4:8.3f}{0.0:8.3f}{z:8.3f}  {occupancy:4.2f} 20.00           {name[0]}"
                )
        path = tmp_path / "altloc.pdb"
        path.write_text("\n".join(l[:6] + f"{i + 1:5d}" + l[11:]
                                  for i, l in enumerate(lines)) + "\nEND\n")

        extractor = PocketExtractor.from_protein(str(path))
        kept = [l[12:16].strip() for l in extractor._residue_lines[0]]
        assert kept == names
        assert (~np.isnan(extractor._residue_coords[0, :, 0])).sum() == 14
        # The A conformer wins on occupancy, so z stays 0 for the side chain.
        assert np.allclose(extractor._residue_coords[0, 4:, 2], 0.0)

    def test_the_public_triple_is_unchanged_and_the_code_rides_beside_it(self, tmp_path):
        from plmol.interaction.pocket_extractor import PocketExtractor

        path = self._tryptophans(tmp_path / "icodes_public.pdb",
                                 [(100, " "), (100, "A")])
        pocket = PocketExtractor.from_protein(path).extract_for_residues(
            [("A", 100)]
        )
        assert pocket.pocket_residues == [("A", 100, "TRP"), ("A", 100, "TRP")]
        assert pocket.insertion_codes == ["", "A"]


class TestPocketUsesTheOneParser:
    """The pocket comes from PDBParser, not from a text walk of its own."""

    STRUCTURE = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1       2.400   1.000   0.000  1.00 20.00           C\n"
        "ATOM      4  O   ALA A   1       2.400   2.200   0.000  1.00 20.00           O\n"
        "ATOM      5  OXT ALA A   1       3.000   3.000   0.000  1.00 20.00           O\n"
        "HETATM    6  N   MSE A   2       3.700   0.400   0.000  1.00 20.00           N\n"
        "HETATM    7  CA  MSE A   2       4.900   1.200   0.000  1.00 20.00           C\n"
        "HETATM    8  C   MSE A   2       6.100   0.400   0.000  1.00 20.00           C\n"
        "HETATM    9  O   MSE A   2       6.100  -0.800   0.000  1.00 20.00           O\n"
        "HETATM   10  SE  MSE A   2       5.100   2.900   0.000  1.00 20.00          SE\n"
        "END\n"
    )

    def test_a_hetatm_amino_acid_is_a_residue(self, tmp_path):
        # Selenomethionine is deposited as HETATM. The old walk tested the
        # record type for the string ATOM and dropped the whole residue; every
        # other path in plmol asks is_protein_atom, which keeps it.
        from plmol.interaction.pocket_extractor import PocketExtractor

        path = tmp_path / "mse.pdb"
        path.write_text(self.STRUCTURE)
        extractor = PocketExtractor.from_protein(str(path))

        assert [key[:3] for key in extractor._residue_keys] == [
            ("A", 1, "ALA"), ("A", 2, "MSE")
        ]
        assert "SE" in [l[12:16].strip() for l in extractor._residue_lines[1]]

    def test_a_terminal_oxygen_is_not(self, tmp_path):
        # is_protein_atom excludes OXT, so the pocket agrees with the graph.
        from plmol.interaction.pocket_extractor import PocketExtractor

        path = tmp_path / "oxt.pdb"
        path.write_text(self.STRUCTURE)
        extractor = PocketExtractor.from_protein(str(path))
        assert "OXT" not in [l[12:16].strip() for l in extractor._residue_lines[0]]

    def test_the_example_pocket_is_unchanged(self, example_pdb, example_sdf):
        # Lines are regenerated rather than copied, so what RDKit is handed has
        # to perceive the same way.
        from plmol.interaction.pocket_extractor import extract_pocket
        from plmol.parsers.pdb_parser import PDBParser

        PDBParser.clear_cache()
        ligand = Chem.MolFromMolFile(example_sdf, removeHs=False)
        pocket = extract_pocket(example_pdb, ligand, distance_cutoff=6.0)[0]
        assert pocket.num_residues == 24
        assert pocket.num_atoms == 204
        assert pocket.pocket_residues[0] == ("A", 7, "TYR")
