"""Abstract base class for structure file parsers (PDB, mmCIF, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .pdb_parser import ParsedAtom


class StructureParser(ABC):
    """Unified interface for all structure file parsers.

    Downstream featurizers depend only on this interface, not on
    PDB- or mmCIF-specific details.
    """

    @property
    @abstractmethod
    def protein_atoms(self) -> List[ParsedAtom]:
        """Filtered atoms (protein/nucleic acid, no water/hydrogen/metal)."""
        ...

    @property
    @abstractmethod
    def all_atoms(self) -> List[ParsedAtom]:
        """All parsed atoms including HETATM."""
        ...

    @property
    def metal_atoms(self) -> List["ParsedAtom"]:
        """HETATM records whose element is a metal.

        ``protein_atoms`` excludes them -- they are not residues -- but they are
        part of the site: coordination is one of the interactions plmol detects,
        and a pocket keeps its cofactor.
        """
        from ..constants import METAL_ELEMENTS

        return [
            atom for atom in self.all_atoms
            if atom.record_type == "HETATM"
            and atom.element.strip().upper() in METAL_ELEMENTS
        ]

    @property
    def protein_atoms_with_metals(self) -> List["ParsedAtom"]:
        """``protein_atoms`` followed by :attr:`metal_atoms`.

        The protein side of an interaction, as every consumer of it wants it:
        the residues plus the ions they hold, and nothing else. Reading a whole
        structure file with RDKit instead gives the ligands and the solvent
        too, and for 4HHB gives no molecule at all -- proximity bonding across
        the heme boundary puts five bonds on a carbon and sanitisation refuses.
        """
        return list(self.protein_atoms) + self.metal_atoms

    @property
    @abstractmethod
    def file_path(self) -> str:
        """Path to the source structure file."""
        ...

    @abstractmethod
    def get_sequence(self, chain_id: Optional[str] = None) -> str:
        """One-letter sequence for a chain (or all chains)."""
        ...

    @abstractmethod
    def get_sequence_by_chain(self) -> Dict[str, str]:
        """Dict of chain_id -> one-letter sequence."""
        ...

    @abstractmethod
    def get_atom_coords(self) -> np.ndarray:
        """Coordinates of protein_atoms as (N, 3) float32 array."""
        ...
