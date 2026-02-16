"""
Atom-level protein featurizer for extracting atomic features and SASA.
"""

import logging
import math
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
try:
    import freesasa
except ImportError:
    freesasa = None

logger = logging.getLogger(__name__)

from .utils import (
    PDBParser,
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
    # Atom-level feature constants
    RESIDUE_MAX_SASA,
    FORMAL_CHARGE_MAP,
    HBOND_DONOR_ATOMS,
    HBOND_ACCEPTOR_ATOMS,
    BACKBONE_ATOM_SET,
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
        if freesasa is None:
            raise ImportError(
                "freesasa is required for atom-level SASA calculation. "
                "Install it with: pip install freesasa"
            )
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

    def _collect_per_atom_data(self, parser: 'PDBParser') -> Dict[str, list]:
        """Collect per-atom properties from parsed protein atoms.

        Returns dict of lists: residue_tokens, atom_elements, b_factors,
        is_backbone, formal_charges, is_hbond_donor, is_hbond_acceptor,
        atom_names, res_names, res_nums, chain_ids.
        """
        residue_tokens = []
        atom_elements = []
        b_factors = []
        is_backbone = []
        formal_charges = []
        is_hbond_donor = []
        is_hbond_acceptor = []
        atom_names = []
        res_names = []
        res_nums = []
        chain_ids = []

        for atom in parser.protein_atoms:
            if atom.atom_name == 'OXT' or atom.res_name in ['LLP', 'PTR']:
                continue

            res_name_clean = normalize_residue_name(atom.res_name, atom.atom_name)
            if res_name_clean == 'UNK':
                res_name_clean = 'XXX'

            residue_tokens.append(self.res_token.get(res_name_clean, RESIDUE_TOKEN['UNK']))

            # Element type
            element = atom.element
            if element in PROTEIN_ELEMENT_TYPES:
                element_type = PROTEIN_ELEMENT_TYPES[element]
            elif element in ['CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'NA', 'K']:
                element_type = PROTEIN_ELEMENT_TYPES.get(element, PROTEIN_ELEMENT_TYPES['METAL'])
            elif len(element) == 1 and element in ['C', 'N', 'O', 'S', 'P', 'H']:
                element_type = PROTEIN_ELEMENT_TYPES[element]
            else:
                fallback_element = ATOM_NAME_TO_ELEMENT.get(atom.atom_name.strip(), None)
                if fallback_element:
                    element_type = PROTEIN_ELEMENT_TYPES.get(fallback_element, PROTEIN_ELEMENT_TYPES['UNK'])
                else:
                    element_type = PROTEIN_ELEMENT_TYPES['UNK']
            atom_elements.append(element_type)

            b_factors.append(min(atom.b_factor / 100.0, 1.0))
            is_backbone.append(1.0 if atom.atom_name in BACKBONE_ATOM_SET else 0.0)
            formal_charges.append(FORMAL_CHARGE_MAP.get((res_name_clean, atom.atom_name), 0.0))

            # H-bond donor: backbone N (except PRO) + sidechain donors
            donor = (atom.atom_name == 'N' and res_name_clean != 'PRO') or \
                    (res_name_clean, atom.atom_name) in HBOND_DONOR_ATOMS
            is_hbond_donor.append(1.0 if donor else 0.0)

            # H-bond acceptor: backbone O + sidechain acceptors
            acceptor = atom.atom_name == 'O' or \
                       (res_name_clean, atom.atom_name) in HBOND_ACCEPTOR_ATOMS
            is_hbond_acceptor.append(1.0 if acceptor else 0.0)

            atom_names.append(atom.atom_name)
            res_names.append(res_name_clean)
            res_nums.append(atom.res_num)
            chain_ids.append(atom.chain_id)

        return {
            'residue_tokens': residue_tokens, 'atom_elements': atom_elements,
            'b_factors': b_factors, 'is_backbone': is_backbone,
            'formal_charges': formal_charges, 'is_hbond_donor': is_hbond_donor,
            'is_hbond_acceptor': is_hbond_acceptor, 'atom_names': atom_names,
            'res_names': res_names, 'res_nums': res_nums, 'chain_ids': chain_ids,
        }

    def _compute_derived_scalars(
        self, parser: 'PDBParser', per_atom: Dict[str, list],
        atom_sasa: torch.Tensor, min_len: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute relative SASA, secondary structure, and B-factor z-score."""
        # Relative SASA
        sasa_truncated = atom_sasa[:min_len]
        relative_sasa = torch.zeros(min_len, dtype=torch.float32)
        for i in range(min_len):
            max_sasa = RESIDUE_MAX_SASA.get(per_atom['res_names'][i], 200.0)
            relative_sasa[i] = min(sasa_truncated[i].item() / max_sasa, 1.0) if max_sasa > 0 else 0.0

        # Secondary structure from phi/psi
        ss = self._compute_secondary_structure(
            parser, per_atom['atom_names'][:min_len], min_len
        )

        # Per-chain B-factor z-score
        b_factor_tensor = torch.tensor(per_atom['b_factors'][:min_len], dtype=torch.float32)
        b_factor_zscore = torch.zeros(min_len, dtype=torch.float32)
        chain_groups: Dict[str, List[int]] = {}
        for i, cid in enumerate(per_atom['chain_ids'][:min_len]):
            if cid not in chain_groups:
                chain_groups[cid] = []
            chain_groups[cid].append(i)
        for indices in chain_groups.values():
            idx = torch.tensor(indices, dtype=torch.long)
            chain_b = b_factor_tensor[idx]
            mean, std = chain_b.mean(), chain_b.std()
            if std > 1e-6:
                b_factor_zscore[idx] = (chain_b - mean) / std

        return {
            'sasa': sasa_truncated,
            'relative_sasa': relative_sasa,
            'secondary_structure': ss,
            'b_factor': b_factor_tensor,
            'b_factor_zscore': b_factor_zscore,
        }

    def get_all_atom_features(self, pdb_file: str) -> Dict[str, torch.Tensor]:
        """Get all atom-level features including tokens, coordinates, SASA,
        and enriched per-atom properties."""
        parser = PDBParser(pdb_file)
        token, coord = self.get_protein_atom_features_from_parser(parser)
        atom_sasa, atom_info = self.get_atom_sasa(pdb_file)
        per_atom = self._collect_per_atom_data(parser)

        # Reconcile parser vs freesasa atom counts
        n_parser, n_sasa = len(token), len(atom_sasa)
        min_len = min(n_parser, n_sasa)
        if n_parser != n_sasa:
            logger.warning(
                f"SASA atom count mismatch in {pdb_file}: "
                f"parser={n_parser}, freesasa={n_sasa}. "
                f"Truncating to {min_len} atoms."
            )

        derived = self._compute_derived_scalars(parser, per_atom, atom_sasa, min_len)

        return {
            'token': token[:min_len],
            'coords': coord[:min_len],
            'sasa': derived['sasa'],
            'relative_sasa': derived['relative_sasa'],
            'residue_token': torch.tensor(per_atom['residue_tokens'][:min_len], dtype=torch.long),
            'atom_element': torch.tensor(per_atom['atom_elements'][:min_len], dtype=torch.long),
            'radius': atom_info['radius'][:min_len] if len(atom_info['radius']) >= min_len else atom_info['radius'],
            'b_factor': derived['b_factor'],
            'b_factor_zscore': derived['b_factor_zscore'],
            'is_backbone': torch.tensor(per_atom['is_backbone'][:min_len], dtype=torch.float32),
            'formal_charge': torch.tensor(per_atom['formal_charges'][:min_len], dtype=torch.float32),
            'is_hbond_donor': torch.tensor(per_atom['is_hbond_donor'][:min_len], dtype=torch.float32),
            'is_hbond_acceptor': torch.tensor(per_atom['is_hbond_acceptor'][:min_len], dtype=torch.float32),
            'secondary_structure': derived['secondary_structure'],
            'metadata': {
                'n_atoms': min_len,
                'residue_names': per_atom['res_names'][:min_len],
                'residue_numbers': per_atom['res_nums'][:min_len],
                'atom_names': per_atom['atom_names'][:min_len],
                'chain_labels': per_atom['chain_ids'][:min_len],
            }
        }

    def _compute_secondary_structure(
        self,
        parser: 'PDBParser',
        atom_names: List[str],
        n_atoms: int,
    ) -> torch.Tensor:
        """
        Assign secondary structure from backbone phi/psi angles.

        Uses Ramachandran region heuristic:
        - Helix: phi in [-160, -20], psi in [-80, 20]
        - Sheet: phi in [-180, -60], psi in [60, 180]
        - Coil: everything else

        Returns:
            (n_atoms, 3) float32 tensor: [is_helix, is_sheet, is_coil] per atom
        """
        # Group residues and find backbone N/CA/C coords
        residue_order = []  # list of (chain, resnum)
        residue_backbone = {}  # (chain, resnum) -> {'N': xyz, 'CA': xyz, 'C': xyz}
        seen = set()

        for atom in parser.protein_atoms:
            if atom.atom_name == 'OXT' or atom.res_name in ['LLP', 'PTR']:
                continue
            key = (atom.chain_id, atom.res_num)
            if key not in seen:
                seen.add(key)
                residue_order.append(key)
                residue_backbone[key] = {}
            if atom.atom_name in ('N', 'CA', 'C'):
                residue_backbone[key][atom.atom_name] = atom.coords

        # Compute phi/psi per residue
        residue_ss = {}  # (chain, resnum) -> (helix, sheet, coil)

        for i, key in enumerate(residue_order):
            chain, resnum = key
            phi, psi = None, None

            # phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
            if i > 0:
                prev_key = residue_order[i - 1]
                if prev_key[0] == chain:  # same chain
                    prev_bb = residue_backbone.get(prev_key, {})
                    curr_bb = residue_backbone.get(key, {})
                    if 'C' in prev_bb and 'N' in curr_bb and 'CA' in curr_bb and 'C' in curr_bb:
                        phi = self._dihedral_angle(
                            prev_bb['C'], curr_bb['N'], curr_bb['CA'], curr_bb['C']
                        )

            # psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})
            if i < len(residue_order) - 1:
                next_key = residue_order[i + 1]
                if next_key[0] == chain:  # same chain
                    curr_bb = residue_backbone.get(key, {})
                    next_bb = residue_backbone.get(next_key, {})
                    if 'N' in curr_bb and 'CA' in curr_bb and 'C' in curr_bb and 'N' in next_bb:
                        psi = self._dihedral_angle(
                            curr_bb['N'], curr_bb['CA'], curr_bb['C'], next_bb['N']
                        )

            # Assign SS from Ramachandran regions
            if phi is not None and psi is not None:
                phi_deg = math.degrees(phi)
                psi_deg = math.degrees(psi)

                if -160 <= phi_deg <= -20 and -80 <= psi_deg <= 20:
                    residue_ss[key] = (1.0, 0.0, 0.0)  # helix
                elif -180 <= phi_deg <= -60 and (60 <= psi_deg <= 180 or -180 <= psi_deg <= -120):
                    residue_ss[key] = (0.0, 1.0, 0.0)  # sheet
                else:
                    residue_ss[key] = (0.0, 0.0, 1.0)  # coil
            else:
                residue_ss[key] = (0.0, 0.0, 1.0)  # coil (boundary residues)

        # Map residue SS back to atoms
        ss = torch.zeros(n_atoms, 3, dtype=torch.float32)
        atom_idx = 0
        for atom in parser.protein_atoms:
            if atom.atom_name == 'OXT' or atom.res_name in ['LLP', 'PTR']:
                continue
            if atom_idx >= n_atoms:
                break
            key = (atom.chain_id, atom.res_num)
            h, s, c = residue_ss.get(key, (0.0, 0.0, 1.0))
            ss[atom_idx, 0] = h
            ss[atom_idx, 1] = s
            ss[atom_idx, 2] = c
            atom_idx += 1

        return ss

    @staticmethod
    def _dihedral_angle(
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float],
    ) -> float:
        """Compute dihedral angle in radians from 4 points."""
        b0 = np.array(p0) - np.array(p1)
        b1 = np.array(p2) - np.array(p1)
        b2 = np.array(p3) - np.array(p2)

        # Normalize b1
        b1_norm = np.linalg.norm(b1)
        if b1_norm < 1e-8:
            return 0.0
        b1 = b1 / b1_norm

        # Compute planes
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)

        return math.atan2(y, x)

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
            coords = atom_features['coords'][mask]
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


