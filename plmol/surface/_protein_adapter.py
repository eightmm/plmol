"""Protein mock objects for surface feature computation."""

from __future__ import annotations

import logging
from functools import lru_cache

from rdkit import Chem

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _atomic_number(element: str) -> int:
    """Atomic number for an element symbol, 0 when unrecognized."""
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(element)
    except Exception:
        logger.warning("Unknown element '%s', defaulting atomic number to 0", element)
        return 0


class _SimplePDBResidueInfo:
    def __init__(self, res_name: str, atom_name: str, b_factor: float = 0.0):
        self._res_name = res_name
        self._atom_name = atom_name
        self._b_factor = b_factor

    def GetResidueName(self) -> str:
        return self._res_name

    def GetName(self) -> str:
        return self._atom_name

    def GetTempFactor(self) -> float:
        return self._b_factor


class _SimpleAtom:
    def __init__(self, res_name: str, atom_name: str, element: str,
                 b_factor: float = 0.0, idx: int = 0):
        self._idx = idx
        self._res_info = _SimplePDBResidueInfo(res_name, atom_name, b_factor=b_factor)
        self._atomic_num = _atomic_number(element)

    def GetIdx(self) -> int:
        return self._idx

    def GetPDBResidueInfo(self) -> _SimplePDBResidueInfo:
        return self._res_info

    def GetAtomicNum(self) -> int:
        return self._atomic_num


class _SimpleMol:
    def __init__(self, atoms: list[_SimpleAtom]):
        self._atoms = atoms

    def GetAtoms(self):
        return self._atoms

    def GetNumAtoms(self) -> int:
        return len(self._atoms)


def _build_simple_protein_mol(atom_metadata: list[dict]) -> _SimpleMol:
    atoms = [
        _SimpleAtom(
            res_name=meta.get("res_name", "UNK"),
            atom_name=meta.get("atom_name", ""),
            element=meta.get("element", ""),
            b_factor=meta.get("b_factor", 0.0),
            idx=i,
        )
        for i, meta in enumerate(atom_metadata)
    ]
    return _SimpleMol(atoms)
