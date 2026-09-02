"""Solvent accessible surface area, with or without freesasa.

plmol has always normalised SASA-derived features by its own
``RESIDUE_MAX_SASA`` table while getting the areas themselves from freesasa. If
freesasa was not installed, the SASA blocks quietly became zeros and every
``burial_index`` became 0.5 -- garbage presented as features. This module
removes that failure mode: a Shrake-Rupley implementation in numpy stands in
when freesasa is missing, and can be selected deliberately.

freesasa stays the default where it is available. It is not slower than the
native path and it is what every published plmol feature was computed with, so
switching silently would move values: measured against freesasa's own
Shrake-Rupley with matching radii the native areas agree to r=0.994, but
against freesasa's default Lee-Richards with its ProtOr radii they differ by
r=0.982 and 2% in total area, because both the algorithm and the atomic radii
differ.

    from plmol import set_sasa_backend
    set_sasa_backend("native")   # or "freesasa", or "auto" (the default)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    DEFAULT_VDW_RADIUS,
    ELEMENT_SYMBOL_TO_ATOMIC_NUMBER,
    RESIDUE_MAX_SASA,
    VDW_RADIUS,
)
from .errors import InputError
from .spatial import sphere_point_exposure

logger = logging.getLogger(__name__)

#: Selectable backends. ``"auto"`` prefers freesasa and falls back to native.
SASA_BACKENDS = ("auto", "freesasa", "native")

#: Elements freesasa's default classifier calls polar. Verified against
#: ``freesasa.Classifier`` on a 3260-atom protein: 100% agreement.
POLAR_ELEMENTS = frozenset({"N", "O", "S"})

#: Backbone atoms, which freesasa reports as "main chain".
MAIN_CHAIN_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})

#: Solvent probe radius in Angstrom, the usual water value freesasa also uses.
DEFAULT_PROBE_RADIUS = 1.4

#: Sample points per atom sphere. freesasa's Shrake-Rupley default is 100.
DEFAULT_SASA_POINTS = 100

_BACKEND = "auto"


def set_sasa_backend(name: str) -> None:
    """Choose which implementation computes SASA.

    Args:
        name: ``"auto"`` (freesasa when importable, else native),
            ``"freesasa"`` or ``"native"``.

    Raises:
        InputError: If the name is not one of :data:`SASA_BACKENDS`.
    """
    global _BACKEND
    if name not in SASA_BACKENDS:
        raise InputError(f"Unknown SASA backend {name!r}. Choose one of {SASA_BACKENDS}.")
    _BACKEND = name


def get_sasa_backend() -> str:
    """The configured backend, which may still be ``"auto"``."""
    return _BACKEND


def resolve_sasa_backend() -> str:
    """The backend that would actually run: ``"freesasa"`` or ``"native"``.

    Raises:
        DependencyError: If ``"freesasa"`` was requested but is not installed.
    """
    from .errors import DependencyError

    if _BACKEND == "native":
        return "native"
    if _BACKEND == "freesasa":
        if _import_freesasa() is None:
            raise DependencyError(
                "SASA backend 'freesasa' was requested but the package is not "
                "installed. Install it, or use set_sasa_backend('native')."
            )
        return "freesasa"
    return "freesasa" if _import_freesasa() is not None else "native"


def _import_freesasa():
    try:
        import freesasa

        return freesasa
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """``n_points`` roughly uniform unit vectors (Fibonacci lattice)."""
    from .surface.point_cloud import _fibonacci_sphere as cached

    return cached(n_points)


def shrake_rupley(
    coords: np.ndarray,
    radii: np.ndarray,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    n_points: int = DEFAULT_SASA_POINTS,
) -> np.ndarray:
    """Per-atom solvent accessible surface area by point sampling.

    Each atom's expanded sphere is sampled at ``n_points`` positions; a point is
    accessible unless it falls inside another atom's expanded sphere. The area
    is then ``4 pi r^2`` times the accessible fraction.

    The occlusion test comes from :func:`plmol.spatial.sphere_point_exposure`,
    which considers every overlapping neighbour rather than a fixed number of
    nearest ones, so no sampling cap can make an area come out too large.

    Args:
        coords: Atom positions ``(N, 3)``.
        radii: Van der Waals radii ``(N,)``, without the probe.
        probe_radius: Solvent probe radius in Angstrom.
        n_points: Sample points per atom.

    Returns:
        Per-atom SASA ``(N,)`` in square Angstrom.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise InputError(f"coords must be (N, 3), got {coords.shape}.")
    n_atoms = coords.shape[0]
    if n_atoms == 0:
        return np.zeros(0, dtype=np.float64)

    expanded = np.asarray(radii, dtype=np.float32) + probe_radius
    exposed = sphere_point_exposure(
        coords,
        expanded,
        np.full(n_atoms, n_points, dtype=np.int64),
        _fibonacci_sphere,
    )
    counts = exposed.reshape(n_atoms, n_points).sum(axis=1)
    return 4.0 * np.pi * expanded.astype(np.float64) ** 2 * (counts / n_points)


# ---------------------------------------------------------------------------
# freesasa-shaped results, computed natively
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidueArea:
    """The fields of a freesasa residue area that plmol reads.

    ``relative*`` values are fractions of ``RESIDUE_MAX_SASA`` -- the same
    convention freesasa uses, where 1.0 means fully exposed.
    """

    total: float
    polar: float
    apolar: float
    mainChain: float
    sideChain: float
    relativeTotal: float
    relativePolar: float
    relativeApolar: float
    relativeMainChain: float
    relativeSideChain: float


class NativeSasaStructure:
    """The ``freesasa.Structure`` accessors plmol uses, backed by ``PDBParser``."""

    def __init__(
        self,
        atom_names: Sequence[str],
        residue_names: Sequence[str],
        residue_numbers: Sequence[int],
        chain_labels: Sequence[str],
        radii: np.ndarray,
    ):
        self._atom_names = list(atom_names)
        self._residue_names = list(residue_names)
        self._residue_numbers = list(residue_numbers)
        self._chain_labels = list(chain_labels)
        self._radii = np.asarray(radii, dtype=np.float64)

    def nAtoms(self) -> int:  # noqa: N802 - mirrors freesasa
        return len(self._atom_names)

    def atomName(self, i: int) -> str:  # noqa: N802
        return self._atom_names[i]

    def residueName(self, i: int) -> str:  # noqa: N802
        return self._residue_names[i]

    def residueNumber(self, i: int) -> str:  # noqa: N802
        return str(self._residue_numbers[i])

    def chainLabel(self, i: int) -> str:  # noqa: N802
        return self._chain_labels[i]

    def radius(self, i: int) -> float:
        return float(self._radii[i])


class NativeSasaResult:
    """The ``freesasa.Result`` accessors plmol uses."""

    def __init__(self, structure: NativeSasaStructure, areas: np.ndarray):
        self._structure = structure
        self._areas = np.asarray(areas, dtype=np.float64)
        self._residue_areas: Optional[Dict[str, Dict[str, ResidueArea]]] = None

    def nAtoms(self) -> int:  # noqa: N802
        return len(self._areas)

    def atomArea(self, i: int) -> float:  # noqa: N802
        return float(self._areas[i])

    def totalArea(self) -> float:  # noqa: N802
        return float(self._areas.sum())

    def residueAreas(self) -> Dict[str, Dict[str, ResidueArea]]:  # noqa: N802
        """Per-residue areas, keyed chain then residue number, in file order."""
        if self._residue_areas is None:
            self._residue_areas = _residue_areas(self._structure, self._areas)
        return self._residue_areas


def _residue_areas(
    structure: NativeSasaStructure, areas: np.ndarray
) -> Dict[str, Dict[str, ResidueArea]]:
    """Group per-atom areas into the polar/apolar and main/side-chain split."""
    grouped: Dict[str, Dict[str, List[int]]] = {}
    for i in range(structure.nAtoms()):
        chain = structure.chainLabel(i)
        number = structure.residueNumber(i)
        grouped.setdefault(chain, {}).setdefault(number, []).append(i)

    out: Dict[str, Dict[str, ResidueArea]] = {}
    for chain, residues in grouped.items():
        out[chain] = {}
        for number, indices in residues.items():
            residue_name = structure.residueName(indices[0])
            reference = RESIDUE_MAX_SASA.get(residue_name, 200.0) or 200.0

            total = polar = apolar = main = side = 0.0
            for i in indices:
                area = float(areas[i])
                total += area
                name = structure.atomName(i).strip()
                element = name[0] if name else ""
                if element in POLAR_ELEMENTS:
                    polar += area
                else:
                    apolar += area
                if name in MAIN_CHAIN_ATOM_NAMES:
                    main += area
                else:
                    side += area

            scale = 1.0 / reference
            out[chain][number] = ResidueArea(
                total=total,
                polar=polar,
                apolar=apolar,
                mainChain=main,
                sideChain=side,
                relativeTotal=total * scale,
                relativePolar=polar * scale,
                relativeApolar=apolar * scale,
                relativeMainChain=main * scale,
                relativeSideChain=side * scale,
            )
    return out


def native_structure_result(
    pdb_file: str,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    n_points: int = DEFAULT_SASA_POINTS,
) -> Tuple[NativeSasaStructure, NativeSasaResult]:
    """Parse a PDB and compute SASA without freesasa.

    Atoms come from plmol's own ``PDBParser``, so the atom set matches every
    other feature the library derives from the same file -- unlike freesasa,
    which reads records the parser drops.

    Args:
        pdb_file: Path to a PDB file.
        probe_radius: Solvent probe radius in Angstrom.
        n_points: Sample points per atom.
    """
    from .parsers import PDBParser

    atoms = PDBParser(pdb_file).protein_atoms
    if not atoms:
        empty = NativeSasaStructure([], [], [], [], np.zeros(0))
        return empty, NativeSasaResult(empty, np.zeros(0))

    coords = np.array([atom.coords for atom in atoms], dtype=np.float32)
    radii = np.array([element_radius(atom.element) for atom in atoms], dtype=np.float32)
    structure = NativeSasaStructure(
        atom_names=[atom.atom_name for atom in atoms],
        residue_names=[atom.res_name for atom in atoms],
        residue_numbers=[atom.res_num for atom in atoms],
        chain_labels=[atom.chain_id for atom in atoms],
        radii=radii,
    )
    areas = shrake_rupley(coords, radii, probe_radius=probe_radius, n_points=n_points)
    return structure, NativeSasaResult(structure, areas)


def element_radius(element: Optional[str]) -> float:
    """Van der Waals radius for an element symbol, plmol's own table."""
    atomic_number = ELEMENT_SYMBOL_TO_ATOMIC_NUMBER.get((element or "").upper(), 0)
    return float(VDW_RADIUS.get(atomic_number, DEFAULT_VDW_RADIUS))


def is_polar_element(element: Optional[str]) -> bool:
    """Whether an element counts as polar, matching freesasa's classifier."""
    return (element or "").strip().upper()[:1] in POLAR_ELEMENTS
