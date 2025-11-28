"""
Atom-level protein featurizer for extracting atomic features and SASA.
"""

import logging
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import freesasa

logger = logging.getLogger(__name__)

from .pdb_utils import (
    is_atom_record, is_hetatm_record, is_hydrogen, parse_pdb_atom_line,
    normalize_residue_name,
)
from ..constants import (
    # Amino acid mappings
    AMINO_ACID_LETTERS,
    # Residue tokens
    RESIDUE_TOKEN,
    RESIDUE_ATOM_TOKEN,
    UNK_TOKEN,
    # Element mappings
    PROTEIN_ELEMENT_TYPES,
    ATOM_NAME_TO_ELEMENT,
)


class AtomFeaturizer:
    """
    Atom-level featurizer for protein structures.
    Extracts atomic features including tokens, coordinates, and SASA.
    """

    def __init__(self):
        """Initialize the atom featurizer."""
        self.res_atm_token = RESIDUE_ATOM_TOKEN
        self.res_token = RESIDUE_TOKEN
        self.aa_letter = AMINO_ACID_LETTERS

    def get_protein_atom_features_from_parser(
        self,
        pdb_parser: 'PDBParser',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract atom-level features from pre-parsed PDB data.

        Args:
            pdb_parser: Pre-initialized PDBParser instance

        Returns:
            Tuple of (token, coord):
                - token: torch.Tensor of shape [n_atoms] with atom type tokens
                - coord: torch.Tensor of shape [n_atoms, 3] with 3D coordinates
        """
        token, coord = [], []

        for atom in pdb_parser.protein_atoms:
            # Skip terminal oxygen and modified residues
            if atom.atom_name == 'OXT' or atom.res_name in ['LLP', 'PTR']:
                continue

            # Normalize residue name
            res_type_norm = normalize_residue_name(atom.res_name, atom.atom_name)

            # Handle unknown residues
            if res_type_norm == 'UNK':
                res_type = 'XXX'
                atom_type = atom.atom_name
                if atom_type not in ['N', 'CA', 'C', 'O', 'CB', 'P', 'S', 'SE']:
                    atom_type = atom_type[0] if atom_type else 'C'
            else:
                res_type = res_type_norm
                atom_type = atom.atom_name

            # Get token ID
            tok = self.res_atm_token.get((res_type, atom_type), UNK_TOKEN)

            token.append(tok)
            coord.append(atom.coords)

        token = torch.tensor(token, dtype=torch.long)
        coord = torch.tensor(coord, dtype=torch.float32)

        return token, coord

    def get_protein_atom_features(self, pdb_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract atom-level features from PDB file.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Tuple of (token, coord):
                - token: torch.Tensor of shape [n_atoms] with atom type tokens
                - coord: torch.Tensor of shape [n_atoms, 3] with 3D coordinates
        """
        token, coord = [], []

        with open(pdb_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Use unified parsing functions
            if not (is_atom_record(line) or is_hetatm_record(line)):
                continue

            # Skip hydrogens
            if is_hydrogen(line):
                continue

            # Parse line components (now includes element)
            record_type, atom_type, res_type, res_num, chain_id, xyz, element = parse_pdb_atom_line(line)

            # Skip water molecules
            if res_type == 'HOH':
                continue

            # Skip terminal oxygen and modified residues
            if atom_type == 'OXT' or res_type in ['LLP', 'PTR']:
                continue

            # Normalize residue name (handles metal, HIS/CYS variants, unknown)
            res_type_norm = normalize_residue_name(res_type, atom_type)

            # Handle unknown residues - need special atom_type handling
            if res_type_norm == 'UNK':
                res_type = 'XXX'
                # For non-standard residues, try to preserve key atoms
                if atom_type not in ['N', 'CA', 'C', 'O', 'CB', 'P', 'S', 'SE']:
                    # Use first character as generic atom type
                    atom_type = atom_type[0] if atom_type else 'C'
            else:
                res_type = res_type_norm

            # Get token ID
            tok = self.res_atm_token.get((res_type, atom_type), UNK_TOKEN)

            token.append(tok)
            coord.append(xyz)

        token = torch.tensor(token, dtype=torch.long)
        coord = torch.tensor(coord, dtype=torch.float32)

        return token, coord

    def get_atom_sasa(self, pdb_file: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate atom-level SASA using FreeSASA.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Tuple of (atom_sasa, atom_info):
                - atom_sasa: torch.Tensor of shape [n_atoms] with SASA values
                - atom_info: Dictionary containing:
                    - 'residue_name': Residue names for each atom
                    - 'residue_number': Residue numbers
                    - 'atom_name': Atom names
                    - 'chain_label': Chain labels
                    - 'radius': Atomic radii
        """
        # Calculate SASA using FreeSASA
        structure = freesasa.Structure(pdb_file)
        result = freesasa.calc(structure)

        n_atoms = result.nAtoms()

        atom_sasa = []
        residue_names = []
        residue_numbers = []
        atom_names = []
        chain_labels = []
        radii = []

        for i in range(n_atoms):
            # Get SASA value
            sasa = result.atomArea(i)
            atom_sasa.append(sasa)

            # Get atom information
            residue_names.append(structure.residueName(i))
            residue_numbers.append(int(structure.residueNumber(i)))
            atom_names.append(structure.atomName(i).strip())
            chain_labels.append(structure.chainLabel(i))
            radii.append(structure.radius(i))

        # Convert to tensors
        atom_sasa = torch.tensor(atom_sasa, dtype=torch.float32)

        atom_info = {
            'residue_name': residue_names,
            'residue_number': torch.tensor(residue_numbers, dtype=torch.long),
            'atom_name': atom_names,
            'chain_label': chain_labels,
            'radius': torch.tensor(radii, dtype=torch.float32)
        }

        return atom_sasa, atom_info

    def get_all_atom_features(self, pdb_file: str) -> Dict[str, torch.Tensor]:
        """
        Get all atom-level features including tokens, coordinates, and SASA.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary containing:
                - 'token': Atom type tokens [n_atoms]
                - 'coord': 3D coordinates [n_atoms, 3]
                - 'sasa': SASA values [n_atoms]
                - 'residue_token': Residue type for each atom [n_atoms]
                - 'atom_element': Element type for each atom [n_atoms]
                - 'radius': Atomic radii [n_atoms]
        """
        # Get basic atom features
        token, coord = self.get_protein_atom_features(pdb_file)

        # Get SASA features
        atom_sasa, atom_info = self.get_atom_sasa(pdb_file)

        # Parse PDB again to get element information directly
        residue_tokens = []
        atom_elements = []

        with open(pdb_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Use same filtering as get_protein_atom_features
            if not (is_atom_record(line) or is_hetatm_record(line)):
                continue
            if is_hydrogen(line):
                continue

            # Parse with element
            record_type, atom_name, res_name, res_num, chain_id, xyz, element = parse_pdb_atom_line(line)

            # Skip water
            if res_name == 'HOH':
                continue

            # Skip same atoms as get_protein_atom_features
            if atom_name == 'OXT' or res_name in ['LLP', 'PTR']:
                continue

            # Handle residue name normalization (centralized function)
            res_name_clean = normalize_residue_name(res_name, atom_name)
            if res_name_clean == 'UNK':
                res_name_clean = 'XXX'

            res_tok = self.res_token.get(res_name_clean, RESIDUE_TOKEN['UNK'])
            residue_tokens.append(res_tok)

            # Map element symbol to element type integer
            # Handle special cases for metals and 2-letter elements
            if element in PROTEIN_ELEMENT_TYPES:
                element_type = PROTEIN_ELEMENT_TYPES[element]
            elif element in ['CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'NA', 'K']:
                # Metal ions
                element_type = PROTEIN_ELEMENT_TYPES.get(element, PROTEIN_ELEMENT_TYPES['METAL'])
            elif len(element) == 1 and element in ['C', 'N', 'O', 'S', 'P', 'H']:
                # Single letter elements
                element_type = PROTEIN_ELEMENT_TYPES[element]
            else:
                # Unknown element - try to infer from atom name
                atom_name_clean = atom_name.strip()
                fallback_element = ATOM_NAME_TO_ELEMENT.get(atom_name_clean, None)
                if fallback_element:
                    element_type = PROTEIN_ELEMENT_TYPES.get(fallback_element, PROTEIN_ELEMENT_TYPES['UNK'])
                else:
                    element_type = PROTEIN_ELEMENT_TYPES['UNK']

            atom_elements.append(element_type)

        # Ensure all tensors have the same length
        orig_token_len = len(token)
        orig_sasa_len = len(atom_sasa)
        orig_element_len = len(atom_elements)
        min_len = min(orig_token_len, orig_sasa_len, orig_element_len)

        if not (orig_token_len == orig_sasa_len == orig_element_len):
            logger.warning(
                f"Atom count mismatch in {pdb_file}: "
                f"tokens={orig_token_len}, SASA={orig_sasa_len}, elements={orig_element_len}. "
                f"Truncating to {min_len} atoms."
            )

        features = {
            'token': token[:min_len],
            'coord': coord[:min_len],
            'sasa': atom_sasa[:min_len],
            'residue_token': torch.tensor(residue_tokens[:min_len], dtype=torch.long),
            'atom_element': torch.tensor(atom_elements[:min_len], dtype=torch.long),
            'radius': atom_info['radius'][:min_len] if len(atom_info['radius']) > min_len else atom_info['radius'],
            'metadata': {
                'n_atoms': min_len,
                'residue_names': atom_info['residue_name'][:min_len],
                'residue_numbers': atom_info['residue_number'][:min_len],
                'atom_names': atom_info['atom_name'][:min_len],
                'chain_labels': atom_info['chain_label'][:min_len]
            }
        }

        return features

    def get_residue_aggregated_features(self, pdb_file: str) -> Dict[str, torch.Tensor]:
        """
        Get residue-level features by aggregating atom features.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary with residue-aggregated features
        """
        # Get all atom features
        atom_features = self.get_all_atom_features(pdb_file)

        # Group by residue
        residue_numbers = atom_features['metadata']['residue_numbers']
        unique_residues = torch.unique(residue_numbers)

        residue_features = {
            'residue_token': [],
            'center_of_mass': [],
            'total_sasa': [],
            'mean_sasa': [],
            'n_atoms': []
        }

        for res_num in unique_residues:
            mask = residue_numbers == res_num

            # Get residue token (should be same for all atoms in residue)
            res_tokens = atom_features['residue_token'][mask]
            residue_features['residue_token'].append(res_tokens[0])

            # Calculate center of mass
            coords = atom_features['coord'][mask]
            center_of_mass = coords.mean(dim=0)
            residue_features['center_of_mass'].append(center_of_mass)

            # Aggregate SASA
            sasa = atom_features['sasa'][mask]
            residue_features['total_sasa'].append(sasa.sum())
            residue_features['mean_sasa'].append(sasa.mean())

            # Count atoms
            residue_features['n_atoms'].append(mask.sum())

        # Convert to tensors
        for key in residue_features:
            residue_features[key] = torch.stack(residue_features[key]) if key == 'center_of_mass' else torch.tensor(residue_features[key])

        return residue_features


# Convenience function for direct use
def get_protein_atom_features(pdb_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract atom-level features from PDB file.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Tuple of (token, coord)
    """
    featurizer = AtomFeaturizer()
    return featurizer.get_protein_atom_features(pdb_file)


def get_atom_features_with_sasa(pdb_file: str) -> Dict[str, torch.Tensor]:
    """
    Get all atom-level features including SASA.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Dictionary with all atom features
    """
    featurizer = AtomFeaturizer()
    return featurizer.get_all_atom_features(pdb_file)


