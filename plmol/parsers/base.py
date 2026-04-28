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
