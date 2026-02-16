"""
PDB Standardizer Module

This module provides functionality to standardize and clean PDB files for protein analysis.
It handles residue reordering, atom standardization, and removal of unwanted molecules.
"""

import os
from typing import Dict, List, Tuple, Optional

from ..constants import (
    STANDARD_ATOMS,
    STANDARD_ATOMS_PTM,
    RESIDUE_NAME_MAPPING,
    PTM_RESIDUES,
    NUCLEIC_ACID_RESIDUES,
    BACKBONE_ATOMS_WITH_CB,
    PTM_HANDLING_MODES,
)


class PDBStandardizer:
    """
    A class for standardizing PDB files.

    This class provides methods to clean and standardize PDB files by:
    - Removing hydrogen atoms (optional)
    - Removing water molecules
    - Removing DNA/RNA residues
    - Reordering atoms according to standard definitions
    - Renumbering residues sequentially
    - Handling post-translational modifications (PTMs) based on use case
    """

    # Backbone atoms to keep for UNK residues
    BACKBONE_ATOMS = BACKBONE_ATOMS_WITH_CB

    def __init__(self, remove_hydrogens: bool = True, ptm_handling: str = 'base_aa'):
        """
        Initialize the PDB standardizer.

        Args:
            remove_hydrogens: Whether to remove hydrogen atoms from the PDB
            ptm_handling: How to handle post-translational modifications (PTMs)
                - 'base_aa': Convert PTMs to their base amino acids (default)
                             SEP→SER, PTR→TYR, MSE→MET, etc.
                             Note: May lose atoms if atom names don't match parent
                             Use for: Legacy compatibility
                - 'unk': Convert PTMs to UNK and keep only backbone atoms (recommended)
                         MSE→UNK (N,CA,C,O,CB only), SEP→UNK, etc.
                         Protonation variants (HID,CYX) still map to parent (HIS,CYS)
                         Use for: ML models, ensures consistency across all levels
                - 'preserve': Keep PTM residues and all their atoms intact
                              SEP stays as SEP with phosphate groups
                              Use for: Protein-ligand modeling, structural analysis
                - 'remove': Remove all PTM residues from the structure
                            Use for: Cleaning structures for standard-AA-only analysis
        """
        self.remove_hydrogens = remove_hydrogens
        self.ptm_handling = ptm_handling

        # Validate ptm_handling parameter
        valid_modes = PTM_HANDLING_MODES
        if ptm_handling not in valid_modes:
            raise ValueError(
                f"Invalid ptm_handling mode: '{ptm_handling}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )

        # Use centralized constants (combine standard + PTM atoms for preserve mode)
        self.standard_atoms = {**STANDARD_ATOMS, **STANDARD_ATOMS_PTM}
        self.nucleic_acid_residues = NUCLEIC_ACID_RESIDUES
        self.residue_name_mapping = RESIDUE_NAME_MAPPING
        self.ptm_residues = PTM_RESIDUES

    def _normalize_residue_name(self, res_name: str) -> str:
        """
        Normalize residue name based on ptm_handling mode.

        Args:
            res_name: Original residue name (e.g., 'HID', 'HIE', 'MSE', 'SEP')

        Returns:
            Normalized residue name based on ptm_handling mode:
            - 'base_aa': Maps to standard amino acid (e.g., 'HIS', 'MET', 'SER')
            - 'unk': Maps PTMs to 'UNK', protonation variants to parent (HID→HIS)
            - 'preserve': Returns original name for PTMs (e.g., 'SEP', 'MSE')
            - 'remove': Returns original name (removal handled in _process_atom_line)
        """
        if self.ptm_handling == 'preserve':
            # Don't map PTM residues, keep original names
            # Still map protonation states to standard names (HID→HIS, CYX→CYS)
            if res_name in self.ptm_residues:
                return res_name  # Keep PTM name
            else:
                return self.residue_name_mapping.get(res_name, res_name)
        elif self.ptm_handling == 'unk':
            # PTMs become UNK, protonation variants map to parent
            if res_name in self.ptm_residues:
                return 'UNK'  # PTMs → UNK
            else:
                # Protonation variants (HID, CYX, etc.) map to parent
                return self.residue_name_mapping.get(res_name, res_name)
        else:
            # 'base_aa' or 'remove' mode: use normal mapping
            return self.residue_name_mapping.get(res_name, res_name)

    def standardize(self, input_pdb_path: str, output_pdb_path: str) -> str:
        """
        Standardize a PDB file.

        Args:
            input_pdb_path: Path to the input PDB file
            output_pdb_path: Path where the standardized PDB will be saved

        Returns:
            Path to the standardized PDB file
        """
        # Create output directory if needed
        output_dir = os.path.dirname(output_pdb_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Read and parse PDB file
        protein_residues, hetatm_residues = self._parse_pdb(input_pdb_path)

        # Write standardized PDB
        self._write_standardized_pdb(protein_residues, hetatm_residues, output_pdb_path)

        return output_pdb_path

    def _parse_pdb(self, pdb_path: str) -> Tuple[Dict, Dict]:
        """
        Parse a PDB file and extract residue information.

        Args:
            pdb_path: Path to the PDB file

        Returns:
            Tuple of (protein_residues, hetatm_residues) dictionaries
        """
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        protein_residues = {}
        hetatm_residues = {}

        for line in lines:
            if line.startswith('ATOM'):
                self._process_atom_line(line, protein_residues, hetatm_residues, is_hetatm=False)
            elif line.startswith('HETATM'):
                self._process_atom_line(line, protein_residues, hetatm_residues, is_hetatm=True)

        return protein_residues, hetatm_residues

    def _process_atom_line(self, line: str, protein_dict: Dict, hetatm_dict: Dict, is_hetatm: bool = False):
        """
        Process a single ATOM or HETATM line from PDB file.

        Args:
            line: PDB line to process
            protein_dict: Dictionary to store protein residue information
            hetatm_dict: Dictionary to store HETATM residue information
            is_hetatm: Whether this is a HETATM line
        """
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21]
        res_num_str = line[22:27].strip()  # Include insertion code
        element = line[76:78].strip() if len(line) > 76 else atom_name[0]

        # Skip hydrogens if requested
        if self.remove_hydrogens and (atom_name.startswith('H') or element.upper() == 'H'):
            return

        # Skip water molecules
        if res_name in ['HOH', 'WAT']:
            return

        # Skip nucleic acid residues
        if res_name in self.nucleic_acid_residues:
            return

        # Handle PTM removal if requested
        if self.ptm_handling == 'remove':
            if res_name in self.ptm_residues:
                return

        # Handle PTM → UNK with backbone only
        is_ptm_to_unk = False
        if self.ptm_handling == 'unk':
            if res_name in self.ptm_residues:
                is_ptm_to_unk = True
                # Only keep backbone atoms for PTMs
                if atom_name not in self.BACKBONE_ATOMS:
                    return

        # Normalize residue name to standard amino acid
        normalized_res_name = self._normalize_residue_name(res_name)

        # Determine target dictionary:
        # - PTMs converted to UNK should go to protein_dict (treated as ATOM)
        # - Standard amino acids go to protein_dict
        # - Other HETATM (ligands, etc.) go to hetatm_dict
        if is_hetatm and not is_ptm_to_unk:
            # Check if it's an amino acid variant (should be protein)
            if normalized_res_name in self.standard_atoms and normalized_res_name != 'UNK':
                target_dict = protein_dict
            else:
                target_dict = hetatm_dict
        else:
            # ATOM records or PTM→UNK always go to protein
            target_dict = protein_dict

        # Store residue information with normalized name
        residue_key = (chain_id, res_num_str, normalized_res_name)
        if residue_key not in target_dict:
            target_dict[residue_key] = {}
        target_dict[residue_key][atom_name] = line

    def _sort_residue_key(self, residue_key: Tuple) -> Tuple:
        """
        Create a sort key for residue ordering.

        Args:
            residue_key: Tuple of (chain_id, res_num_str, res_name)

        Returns:
            Sort key tuple
        """
        chain_id, res_num_str, res_name = residue_key
        # Extract numeric part and insertion code
        res_num = int(''.join(filter(str.isdigit, res_num_str)) or '0')
        insertion_code = ''.join(filter(str.isalpha, res_num_str))
        return (chain_id, res_num, insertion_code)

    def _write_standardized_pdb(self, protein_residues: Dict, hetatm_residues: Dict,
                                output_path: str):
        """
        Write standardized PDB file.

        Args:
            protein_residues: Dictionary of protein residues
            hetatm_residues: Dictionary of HETATM residues
            output_path: Path to write the standardized PDB
        """
        standardized_lines = []
        atom_counter = 1

        # Process protein residues
        standardized_lines, atom_counter = self._write_protein_residues(
            protein_residues, standardized_lines, atom_counter
        )

        # Process HETATM residues
        standardized_lines, atom_counter = self._write_hetatm_residues(
            hetatm_residues, standardized_lines, atom_counter
        )

        # Write to file
        with open(output_path, 'w') as f:
            f.writelines(standardized_lines)

    def _write_protein_residues(self, protein_residues: Dict, lines: List[str],
                                atom_counter: int) -> Tuple[List[str], int]:
        """
        Write protein residues in standardized format.
        """
        # Group by chain
        protein_by_chain = {}
        for residue_key in protein_residues.keys():
            chain_id = residue_key[0]
            if chain_id not in protein_by_chain:
                protein_by_chain[chain_id] = []
            protein_by_chain[chain_id].append(residue_key)

        # Process each chain
        for chain_id in sorted(protein_by_chain.keys()):
            sorted_residues = sorted(protein_by_chain[chain_id], key=self._sort_residue_key)
            res_counter = 1

            for residue_key in sorted_residues:
                chain_id, res_num_str, res_name = residue_key
                residue_atoms = protein_residues[residue_key]

                # Write atoms in standard order if possible
                if res_name in self.standard_atoms:
                    for standard_atom in self.standard_atoms[res_name]:
                        if standard_atom in residue_atoms:
                            line = self._format_atom_line(
                                residue_atoms[standard_atom],
                                atom_counter, res_counter, chain_id, res_name
                            )
                            lines.append(line)
                            atom_counter += 1
                else:
                    # Non-standard residue - write all atoms
                    for atom_name, atom_line in residue_atoms.items():
                        line = self._format_atom_line(
                            atom_line, atom_counter, res_counter, chain_id, res_name
                        )
                        lines.append(line)
                        atom_counter += 1

                res_counter += 1

        return lines, atom_counter

    def _write_hetatm_residues(self, hetatm_residues: Dict, lines: List[str],
                               atom_counter: int) -> Tuple[List[str], int]:
        """
        Write HETATM residues in standardized format.
        """
        # Group by chain
        hetatm_by_chain = {}
        for residue_key in hetatm_residues.keys():
            chain_id = residue_key[0]
            if chain_id not in hetatm_by_chain:
                hetatm_by_chain[chain_id] = []
            hetatm_by_chain[chain_id].append(residue_key)

        # Process each chain
        for chain_id in sorted(hetatm_by_chain.keys()):
            sorted_residues = sorted(hetatm_by_chain[chain_id], key=self._sort_residue_key)
            hetatm_counter = 1

            for residue_key in sorted_residues:
                chain_id, res_num_str, res_name = residue_key
                residue_atoms = hetatm_residues[residue_key]

                for atom_name, atom_line in residue_atoms.items():
                    line = self._format_hetatm_line(
                        atom_line, atom_counter, hetatm_counter, res_name, chain_id
                    )
                    lines.append(line)
                    atom_counter += 1

                hetatm_counter += 1

        return lines, atom_counter

    def _format_atom_line(self, original_line: str, atom_counter: int,
                         res_counter: int, chain_id: str, res_name: str) -> str:
        """
        Format an ATOM line in standardized PDB format.
        """
        atom_name = original_line[12:16].strip()
        x = float(original_line[30:38])
        y = float(original_line[38:46])
        z = float(original_line[46:54])
        occupancy = original_line[54:60].strip() if len(original_line) > 54 else "1.00"
        temp_factor = original_line[60:66].strip() if len(original_line) > 60 else "0.00"
        element = original_line[76:78].strip() if len(original_line) > 76 else atom_name[0]

        return f"ATOM  {atom_counter:5d}  {atom_name:<4s}{res_name:3s} {chain_id}{res_counter:>4d}    " \
               f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:>6s}{temp_factor:>6s}          {element:>2s}\n"

    def _format_hetatm_line(self, original_line: str, atom_counter: int,
                           hetatm_counter: int, res_name: str, chain_id: str) -> str:
        """
        Format a HETATM line in standardized PDB format.
        """
        atom_name = original_line[12:16].strip()
        x = float(original_line[30:38])
        y = float(original_line[38:46])
        z = float(original_line[46:54])
        occupancy = original_line[54:60].strip() if len(original_line) > 54 else "1.00"
        temp_factor = original_line[60:66].strip() if len(original_line) > 60 else "0.00"
        element = original_line[76:78].strip() if len(original_line) > 76 else atom_name[0]

        return f"HETATM{atom_counter:5d}  {atom_name:<4s}{res_name:3s} {chain_id}{hetatm_counter:>4d}    " \
               f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:>6s}{temp_factor:>6s}          {element:>2s}\n"


def standardize_pdb(input_pdb_path: str, output_pdb_path: str,
                     remove_hydrogens: bool = True, ptm_handling: str = 'base_aa') -> str:
    """
    Convenience function to standardize a PDB file.

    Args:
        input_pdb_path: Path to input PDB file
        output_pdb_path: Path for output standardized PDB
        remove_hydrogens: Whether to remove hydrogen atoms
        ptm_handling: How to handle PTMs:
            - 'base_aa': Convert PTMs to parent amino acids (legacy)
            - 'unk': Convert PTMs to UNK with backbone only (recommended for ML)
            - 'preserve': Keep PTMs intact
            - 'remove': Remove PTM residues entirely

    Returns:
        Path to the standardized PDB file
    """
    standardizer = PDBStandardizer(remove_hydrogens=remove_hydrogens, ptm_handling=ptm_handling)
    return standardizer.standardize(input_pdb_path, output_pdb_path)

