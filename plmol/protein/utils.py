"""
Unified PDB Parsing Utilities.

Centralizes all PDB parsing logic to avoid code duplication across:
- atom_featurizer.py
- residue_featurizer.py
- hierarchical_featurizer.py
- esm_featurizer.py

Usage:
    from .utils import PDBParser

    # Parse once, use everywhere
    parser = PDBParser(pdb_file)

    # Access parsed data
    atoms = parser.get_protein_atoms()
    sequence = parser.get_sequence()
    sequence_by_chain = parser.get_sequence_by_chain()
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

from ..constants import (
    AMINO_ACID_3TO1,
    AMINO_ACID_3_TO_INT,
    HISTIDINE_VARIANTS,
    CYSTEINE_VARIANTS,
    AMINO_ACID_LETTERS,
    RESIDUE_NAME_MAPPING,
    METAL_RESIDUES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Parsed Atoms
# ============================================================================

@dataclass
class ParsedAtom:
    """Single parsed atom from PDB file."""
    record_type: str        # 'ATOM' or 'HETATM'
    atom_name: str          # e.g., 'CA', 'CB', 'N'
    res_name: str           # e.g., 'ALA', 'GLY'
    res_num: int            # Residue number
    chain_id: str           # Chain identifier
    coords: Tuple[float, float, float]
    element: str            # Element symbol (C, N, O, S, etc.)
    insertion_code: str = '' # Insertion code if any
    b_factor: float = 0.0   # B-factor (temperature factor, PDB columns 61-66)


@dataclass
class ParsedResidue:
    """Parsed residue information."""
    chain_id: str
    res_num: int
    res_name: str
    res_type_int: int       # Integer code for residue type
    atoms: List[ParsedAtom] = field(default_factory=list)


# ============================================================================
# Low-level Parsing Functions
# ============================================================================

def is_atom_record(line: str) -> bool:
    """Check if a PDB line is an ATOM record."""
    if len(line) < 6:
        return False
    return line[:6].strip() == 'ATOM'


def is_hetatm_record(line: str) -> bool:
    """Check if a PDB line is a HETATM record."""
    if len(line) < 6:
        return False
    return line[:6].strip() == 'HETATM'


def is_hydrogen(line: str) -> bool:
    """Check if atom is hydrogen based on PDB line."""
    if len(line) < 14:
        return False
    # Check element column (77-78) first
    if len(line) > 77:
        element = line[76:78].strip()
        if element and element.upper() == 'H':
            return True
    # Fallback: check atom name (column 13-16)
    if len(line) > 13:
        atom_name = line[12:16].strip()
        if atom_name and atom_name[0] == 'H':
            return True
    return False


def parse_pdb_line(line: str) -> ParsedAtom:
    """
    Parse a PDB ATOM/HETATM line into ParsedAtom.

    PDB format columns:
        1-6:   Record type (ATOM/HETATM)
        13-16: Atom name
        17:    Alternate location indicator
        18-20: Residue name
        22:    Chain identifier
        23-26: Residue sequence number
        27:    Code for insertion of residues
        31-38: X coordinate
        39-46: Y coordinate
        47-54: Z coordinate
        77-78: Element symbol

    Returns:
        ParsedAtom with all extracted information
    """
    record_type = line[:6].strip()
    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21] if len(line) > 21 else ' '
    insertion_code = line[26] if len(line) > 26 and line[26] != ' ' else ''

    try:
        res_num = int(line[22:26]) if len(line) > 26 else 0
    except ValueError:
        res_num = 0

    # Parse element symbol (columns 77-78)
    element = ''
    if len(line) > 77:
        element = line[76:78].strip().upper()

    # Fallback: infer from atom name if element not present
    if not element and atom_name:
        element = _infer_element(atom_name)

    # Parse coordinates
    try:
        x = float(line[30:38]) if len(line) > 38 else 0.0
        y = float(line[38:46]) if len(line) > 46 else 0.0
        z = float(line[46:54]) if len(line) > 54 else 0.0
        coords = (x, y, z)
    except (ValueError, IndexError):
        coords = (0.0, 0.0, 0.0)

    # Parse B-factor (columns 61-66)
    b_factor = 0.0
    if len(line) > 65:
        try:
            b_factor = float(line[60:66])
        except (ValueError, IndexError):
            b_factor = 0.0

    return ParsedAtom(
        record_type=record_type,
        atom_name=atom_name,
        res_name=res_name,
        res_num=res_num,
        chain_id=chain_id,
        coords=coords,
        element=element,
        insertion_code=insertion_code,
        b_factor=b_factor,
    )


def _infer_element(atom_name: str) -> str:
    """Infer element symbol from atom name."""
    # Remove digits and special characters
    element = ''.join(c for c in atom_name if c.isalpha())
    if not element:
        return 'C'  # Default to carbon

    # Check for 2-letter elements first
    two_letter_elements = ['CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'NA', 'CL', 'BR', 'SE']
    if len(element) >= 2 and element[:2].upper() in two_letter_elements:
        return element[:2].upper()

    return element[0].upper()


def normalize_residue_name(res_name: str, atom_name: str = '') -> str:
    """
    Normalize residue name handling variants and special cases.

    Uses comprehensive RESIDUE_NAME_MAPPING for protonation states, PTMs,
    and modified residues.

    Args:
        res_name: Original residue name from PDB
        atom_name: Atom name (for metal ion detection)

    Returns:
        Normalized residue name (standard 3-letter code, 'METAL', or 'UNK')
    """
    res_name = res_name.strip()

    # Metal ion detection: atom name matches first 2 chars of residue
    if len(atom_name) >= 2 and len(res_name) >= 2 and atom_name[:2] == res_name[:2]:
        return 'METAL'

    # Check comprehensive mapping (covers HIS/CYS variants, PTMs, protonation states)
    if res_name in RESIDUE_NAME_MAPPING:
        return RESIDUE_NAME_MAPPING[res_name]

    # Standard amino acids (already in canonical form)
    if res_name in AMINO_ACID_LETTERS:
        return res_name

    # Unknown residue
    return 'UNK'


# ============================================================================
# PDBParser Class - Main API
# ============================================================================

class PDBParser:
    """
    Unified PDB parser with caching.

    Parses a PDB file once and caches results for efficient reuse.

    Attributes:
        pdb_path: Path to the PDB file
        atoms: List of all parsed atoms
        protein_atoms: List of protein atoms only (no HETATM, water, hydrogen)
        residues: Dictionary mapping (chain, resnum) to ParsedResidue
    """

    # Class-level cache for parsed PDB files
    _cache: Dict[str, 'PDBParser'] = {}

    def __init__(self, pdb_path: str, skip_cache: bool = False):
        """
        Initialize parser and parse PDB file.

        Args:
            pdb_path: Path to PDB file
            skip_cache: If True, don't use or store in cache

        Raises:
            FileNotFoundError: If PDB file doesn't exist
            ValueError: If PDB file is empty or invalid
        """
        self.pdb_path = os.path.abspath(pdb_path)

        # Check file existence
        if not os.path.exists(self.pdb_path):
            raise FileNotFoundError(f"PDB file not found: {self.pdb_path}")

        # Check if cached
        if not skip_cache and self.pdb_path in PDBParser._cache:
            cached = PDBParser._cache[self.pdb_path]
            self._all_atoms = cached._all_atoms
            self._protein_atoms = cached._protein_atoms
            self._residues = cached._residues
            self._lines = cached._lines
            return

        # Parse PDB file
        self._lines: List[str] = []
        self._all_atoms: List[ParsedAtom] = []
        self._protein_atoms: List[ParsedAtom] = []
        self._residues: Dict[Tuple[str, int], ParsedResidue] = {}

        self._parse()

        # Cache if requested
        if not skip_cache:
            PDBParser._cache[self.pdb_path] = self

    def _parse(self):
        """Parse the PDB file."""
        try:
            with open(self.pdb_path, 'r') as f:
                self._lines = f.readlines()
        except IOError as e:
            raise ValueError(f"Failed to read PDB file {self.pdb_path}: {e}")

        if not self._lines:
            raise ValueError(f"PDB file is empty: {self.pdb_path}")

        for line in self._lines:
            # Only process ATOM and HETATM records
            if not (is_atom_record(line) or is_hetatm_record(line)):
                continue

            atom = parse_pdb_line(line)
            self._all_atoms.append(atom)

            # Filter for protein atoms
            if self._is_protein_atom(atom, line):
                self._protein_atoms.append(atom)

                # Build residue dictionary
                key = (atom.chain_id, atom.res_num)
                if key not in self._residues:
                    res_type_int = AMINO_ACID_3_TO_INT.get(
                        normalize_residue_name(atom.res_name, atom.atom_name),
                        20  # UNK
                    )
                    self._residues[key] = ParsedResidue(
                        chain_id=atom.chain_id,
                        res_num=atom.res_num,
                        res_name=atom.res_name,
                        res_type_int=res_type_int,
                    )
                self._residues[key].atoms.append(atom)

        if not self._protein_atoms:
            logger.warning(f"No protein atoms found in {self.pdb_path}")

    def _is_protein_atom(self, atom: ParsedAtom, line: str) -> bool:
        """
        Check if atom should be included as protein atom.

        Preprocessing is done here - downstream featurizers receive clean data.
        Excludes: hydrogens, water, metal ions, ligands, terminal oxygens.
        """
        # Skip hydrogens
        if is_hydrogen(line):
            return False

        # Skip water molecules
        if atom.res_name == 'HOH':
            return False

        # Skip metal ions (centralized constant)
        if atom.res_name in METAL_RESIDUES:
            return False

        # Skip terminal oxygen and specific modified residues
        if atom.atom_name == 'OXT':
            return False

        if atom.res_name in ['LLP', 'PTR']:
            return False

        # For HETATM, only include known amino acids (no metals, no ligands)
        if atom.record_type == 'HETATM':
            norm_res = normalize_residue_name(atom.res_name, atom.atom_name)
            # Must be a standard amino acid (including variants like HID, CYX, etc.)
            if norm_res not in AMINO_ACID_LETTERS:
                return False

        return True

    # ============================================================================
    # Public API - Data Access
    # ============================================================================

    @property
    def all_atoms(self) -> List[ParsedAtom]:
        """Get all parsed atoms."""
        return self._all_atoms

    @property
    def protein_atoms(self) -> List[ParsedAtom]:
        """Get protein atoms only (no water, hydrogen, metal ions, ligands)."""
        return self._protein_atoms

    @property
    def residues(self) -> Dict[Tuple[str, int], ParsedResidue]:
        """Get residues dictionary keyed by (chain, resnum)."""
        return self._residues

    def get_residue_list(self) -> List[Tuple[str, int, int]]:
        """
        Get list of residues as (chain, resnum, restype_int) tuples.

        Sorted by chain and residue number.
        Metal ions are already excluded during parsing (see _is_protein_atom).
        """
        residue_list = []
        for (chain, resnum), residue in sorted(self._residues.items()):
            res_type = AMINO_ACID_3_TO_INT.get(
                normalize_residue_name(residue.res_name),
                20  # UNK
            )
            residue_list.append((chain, resnum, res_type))
        return residue_list

    def get_sequence(self, chain_id: Optional[str] = None) -> str:
        """
        Get amino acid sequence in one-letter code.

        Args:
            chain_id: Specific chain to extract (None = all chains concatenated)

        Returns:
            One-letter amino acid sequence

        Note:
            Metal ions are already excluded during parsing (see _is_protein_atom).
        """
        sequence = []
        seen_residues = set()

        for atom in self._protein_atoms:
            if chain_id and atom.chain_id != chain_id:
                continue

            key = (atom.chain_id, atom.res_num)
            if key in seen_residues:
                continue
            seen_residues.add(key)

            norm_res = normalize_residue_name(atom.res_name, atom.atom_name)
            aa = AMINO_ACID_3TO1.get(norm_res, 'X')
            sequence.append(aa)

        return ''.join(sequence)

    def get_sequence_by_chain(self) -> Dict[str, str]:
        """
        Get sequences separated by chain.

        Returns:
            Dictionary mapping chain_id -> one-letter sequence

        Note:
            Metal ions are already excluded during parsing (see _is_protein_atom).
        """
        chains: Dict[str, List[str]] = {}
        seen: Dict[str, set] = {}

        for atom in self._protein_atoms:
            chain = atom.chain_id
            if chain not in chains:
                chains[chain] = []
                seen[chain] = set()

            key = (chain, atom.res_num)
            if key in seen[chain]:
                continue
            seen[chain].add(key)

            norm_res = normalize_residue_name(atom.res_name, atom.atom_name)
            aa = AMINO_ACID_3TO1.get(norm_res, 'X')
            chains[chain].append(aa)

        return {chain: ''.join(seq) for chain, seq in chains.items()}

    def get_atom_coords(self) -> np.ndarray:
        """Get coordinates of all protein atoms as numpy array [N, 3]."""
        if not self._protein_atoms:
            return np.zeros((0, 3))
        return np.array([atom.coords for atom in self._protein_atoms])

    def get_atom_data(self) -> Dict[str, List]:
        """
        Get all atom data as dictionary of lists.

        Returns:
            Dictionary with keys:
                - atom_names: List[str]
                - res_names: List[str]
                - res_nums: List[int]
                - chain_ids: List[str]
                - coords: List[Tuple[float, float, float]]
                - elements: List[str]
                - residue_keys: List[Tuple[str, int]]
        """
        return {
            'atom_names': [a.atom_name for a in self._protein_atoms],
            'res_names': [a.res_name for a in self._protein_atoms],
            'res_nums': [a.res_num for a in self._protein_atoms],
            'chain_ids': [a.chain_id for a in self._protein_atoms],
            'coords': [a.coords for a in self._protein_atoms],
            'elements': [a.element for a in self._protein_atoms],
            'residue_keys': [(a.chain_id, a.res_num) for a in self._protein_atoms],
        }

    def get_num_atoms(self) -> int:
        """Get number of protein atoms."""
        return len(self._protein_atoms)

    def get_num_residues(self) -> int:
        """Get number of residues."""
        return len(self._residues)

    @classmethod
    def clear_cache(cls):
        """Clear the parser cache."""
        cls._cache.clear()

    @classmethod
    def get_cached(cls, pdb_path: str) -> Optional['PDBParser']:
        """Get cached parser if available."""
        abs_path = os.path.abspath(pdb_path)
        return cls._cache.get(abs_path)


# ============================================================================
# Convenience Functions (backward compatibility)
# ============================================================================

def parse_pdb_atom_line(line: str) -> Tuple[str, str, str, int, str, Tuple[float, float, float], str]:
    """
    Parse a PDB ATOM/HETATM line into components.

    For backward compatibility with atom_featurizer.py.

    Returns:
        Tuple of (record_type, atom_name, res_name, res_num, chain_id, coordinates, element)
    """
    atom = parse_pdb_line(line)
    return (
        atom.record_type,
        atom.atom_name,
        atom.res_name,
        atom.res_num,
        atom.chain_id,
        atom.coords,
        atom.element,
    )


# ============================================================================
# Coordinate Utilities
# ============================================================================

def calculate_sidechain_centroid(residue_coords: 'np.ndarray') -> 'np.ndarray':
    """
    Calculate sidechain centroid from residue atom coordinates.

    Standard atom order: [N, CA, C, O, CB, ...sidechain atoms...]
    Sidechain atoms are at indices 4+ (after N, CA, C, O).

    Args:
        residue_coords: Residue atom coordinates [num_atoms, 3] or [3,] for single atom

    Returns:
        Sidechain centroid coordinates [3,]

    Notes:
        - For residues with sidechain atoms (index 4+): mean of sidechain atoms
        - For GLY or residues with no sidechain: use CA position (index 1)
        - If no CA available: use mean of all available atoms
    """
    # Handle 1D case (single atom)
    if residue_coords.ndim == 1:
        return residue_coords

    num_atoms = residue_coords.shape[0]

    # Standard case: sidechain atoms available (index 4+)
    if num_atoms > 4:
        return residue_coords[4:, :].mean(axis=0)

    # Fallback: use CA position (index 1) for residues without sidechain (e.g., GLY)
    if num_atoms > 1:
        return residue_coords[1, :]

    # Last resort: return the single atom's coordinates
    return residue_coords[0, :]
