"""
Atom-level protein featurizer for extracting atomic features and SASA.
"""

import logging
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
try:
    import freesasa
    freesasa.setVerbosity(freesasa.nowarnings)
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
from ..utils import dihedral_angles, sasa_structure_result
from ..sasa import is_polar_element


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
            logger.warning(
                "freesasa is not available for atom-level SASA calculation. "
                "Returning zeros. Install it with: pip install freesasa"
            )
            empty_sasa = torch.zeros(0, dtype=torch.float32)
            empty_info = {
                'residue_name': [],
                'residue_number': torch.zeros(0, dtype=torch.long),
                'atom_name': [],
                'chain_label': [],
                'radius': torch.zeros(0, dtype=torch.float32),
            }
            return empty_sasa, empty_info
        structure, result = sasa_structure_result(pdb_file)

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

        Returns dict of lists: residue_tokens, atom_elements,
        is_backbone, formal_charges, is_hbond_donor, is_hbond_acceptor,
        atom_names, res_names, res_nums, chain_ids.
        """
        residue_tokens = []
        atom_elements = []
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
            'is_backbone': is_backbone,
            'formal_charges': formal_charges, 'is_hbond_donor': is_hbond_donor,
            'is_hbond_acceptor': is_hbond_acceptor, 'atom_names': atom_names,
            'res_names': res_names, 'res_nums': res_nums, 'chain_ids': chain_ids,
        }

    def _compute_derived_scalars(
        self, parser: 'PDBParser', per_atom: Dict[str, list],
        atom_sasa: torch.Tensor, min_len: int,
        pdb_file: str,
    ) -> Dict[str, torch.Tensor]:
        """Compute relative SASA, burial index, polar classification, and secondary structure."""
        # Relative SASA
        sasa_truncated = atom_sasa[:min_len]
        relative_sasa = torch.zeros(min_len, dtype=torch.float32)
        for i in range(min_len):
            max_sasa = RESIDUE_MAX_SASA.get(per_atom['res_names'][i], 200.0)
            relative_sasa[i] = min(sasa_truncated[i].item() / max_sasa, 1.0) if max_sasa > 0 else 0.0

        # Burial index: 1.0 = fully buried, 0.0 = fully exposed
        burial_index = 1.0 - relative_sasa

        # Per-atom polar/apolar SASA classification. freesasa's classifier is
        # used when available; the element rule below reproduces it exactly on
        # protein atoms (verified 100% on a 3260-atom structure) and is what
        # runs when freesasa is absent.
        is_polar_sasa = torch.zeros(min_len, dtype=torch.float32)
        classifier = None
        if freesasa is not None:
            try:
                classifier = freesasa.Classifier()
            except Exception:
                logger.warning("freesasa.Classifier unavailable; using the element rule.")
        for i in range(min_len):
            res_name = per_atom['res_names'][i]
            atom_name = per_atom['atom_names'][i]
            if classifier is not None:
                is_polar_sasa[i] = 1.0 if classifier.classify(res_name, atom_name) == freesasa.polar else 0.0
            else:
                is_polar_sasa[i] = 1.0 if is_polar_element(atom_name) else 0.0

        # Secondary structure from phi/psi
        ss = self._compute_secondary_structure(
            parser, per_atom['atom_names'][:min_len], min_len
        )

        return {
            'sasa': sasa_truncated,
            'relative_sasa': relative_sasa,
            'burial_index': burial_index,
            'is_polar_sasa': is_polar_sasa,
            'secondary_structure': ss,
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

        derived = self._compute_derived_scalars(parser, per_atom, atom_sasa, min_len, pdb_file)

        return {
            'token': token[:min_len],
            'coords': coord[:min_len],
            'sasa': derived['sasa'],
            'relative_sasa': derived['relative_sasa'],
            'burial_index': derived['burial_index'],
            'is_polar_sasa': derived['is_polar_sasa'],
            'residue_token': torch.tensor(per_atom['residue_tokens'][:min_len], dtype=torch.long),
            'atom_element': torch.tensor(per_atom['atom_elements'][:min_len], dtype=torch.long),
            'radius': atom_info['radius'][:min_len] if len(atom_info['radius']) >= min_len else atom_info['radius'],
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

        # Collect the residues that have a complete phi and psi definition,
        # then evaluate both dihedrals in one batched pass.
        n_res = len(residue_order)
        phi_quads, psi_quads, angle_rows = [], [], []
        for idx in range(n_res):
            key = residue_order[idx]
            chain = key[0]
            curr_bb = residue_backbone.get(key, {})
            if not ('N' in curr_bb and 'CA' in curr_bb and 'C' in curr_bb):
                continue
            if idx == 0 or idx == n_res - 1:
                continue
            prev_key = residue_order[idx - 1]
            next_key = residue_order[idx + 1]
            if prev_key[0] != chain or next_key[0] != chain:
                continue
            prev_bb = residue_backbone.get(prev_key, {})
            next_bb = residue_backbone.get(next_key, {})
            if 'C' not in prev_bb or 'N' not in next_bb:
                continue
            phi_quads.append((prev_bb['C'], curr_bb['N'], curr_bb['CA'], curr_bb['C']))
            psi_quads.append((curr_bb['N'], curr_bb['CA'], curr_bb['C'], next_bb['N']))
            angle_rows.append(key)

        # Default every residue to coil; boundary and incomplete residues stay there.
        residue_ss = {key: (0.0, 0.0, 1.0) for key in residue_order}

        if angle_rows:
            phi_pts = np.asarray(phi_quads, dtype=np.float64)
            psi_pts = np.asarray(psi_quads, dtype=np.float64)
            phi_deg = np.degrees(self._dihedral_angles(
                phi_pts[:, 0], phi_pts[:, 1], phi_pts[:, 2], phi_pts[:, 3]
            ))
            psi_deg = np.degrees(self._dihedral_angles(
                psi_pts[:, 0], psi_pts[:, 1], psi_pts[:, 2], psi_pts[:, 3]
            ))

            is_helix = (phi_deg >= -160) & (phi_deg <= -20) & (psi_deg >= -80) & (psi_deg <= 20)
            is_sheet = (~is_helix) & (phi_deg >= -180) & (phi_deg <= -60) & (
                ((psi_deg >= 60) & (psi_deg <= 180)) | ((psi_deg >= -180) & (psi_deg <= -120))
            )
            for row, helix, sheet in zip(angle_rows, is_helix, is_sheet):
                if helix:
                    residue_ss[row] = (1.0, 0.0, 0.0)
                elif sheet:
                    residue_ss[row] = (0.0, 1.0, 0.0)

        # Map residue SS back to atoms
        ss_rows = []
        for atom in parser.protein_atoms:
            if atom.atom_name == 'OXT' or atom.res_name in ['LLP', 'PTR']:
                continue
            if len(ss_rows) >= n_atoms:
                break
            ss_rows.append(residue_ss.get((atom.chain_id, atom.res_num), (0.0, 0.0, 1.0)))

        ss = torch.zeros(n_atoms, 3, dtype=torch.float32)
        if ss_rows:
            ss[:len(ss_rows)] = torch.tensor(ss_rows, dtype=torch.float32)
        return ss

    @staticmethod
    def _dihedral_angles(
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
    ) -> np.ndarray:
        """Dihedral angles in radians for batches of 4 points, each (M, 3)."""
        return dihedral_angles(p0, p1, p2, p3)


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
