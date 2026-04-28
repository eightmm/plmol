"""Backward compatibility — imports from parsers.pdb_parser."""
from ..parsers.pdb_parser import (
    ParsedAtom, ParsedResidue,
    PDBParser,
    is_atom_record, is_hetatm_record, is_hydrogen,
    parse_pdb_line, parse_pdb_atom_line,
    normalize_residue_name,
    calculate_sidechain_centroid,
)

__all__ = [
    "ParsedAtom", "ParsedResidue", "PDBParser",
    "is_atom_record", "is_hetatm_record", "is_hydrogen",
    "parse_pdb_line", "parse_pdb_atom_line",
    "normalize_residue_name",
    "calculate_sidechain_centroid",
]
