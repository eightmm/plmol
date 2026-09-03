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
