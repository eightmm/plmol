"""Solvent accessible surface area.

Shrake-Rupley: sample each atom's expanded sphere and keep the points no
neighbouring atom covers. The occlusion test is
:func:`plmol.spatial.sphere_point_exposure`, which considers every overlapping
neighbour rather than a fixed number of nearest ones, so nothing here has a
sampling cap that could make an area come out too large.

Up to 0.3.x this deferred to freesasa when it was installed. It no longer does:
this path is 1.3 to 2.0 times faster on every mode that uses SASA, and one
implementation that plmol owns beats two that disagree. Values from 0.3.x and
earlier were freesasa's, which uses Lee-Richards with ProtOr radii where this
uses Shrake-Rupley with plmol's own element table -- per atom the two correlate
at 0.982 and the totals differ by about 2%.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    DEFAULT_VDW_RADIUS,
    ELEMENT_SYMBOL_TO_ATOMIC_NUMBER,
    RESIDUE_MAX_CLASS_SASA,
    RESIDUE_MAX_SASA,
    VDW_RADIUS,
)
from .errors import InputError
from .spatial import sphere_point_exposure

logger = logging.getLogger(__name__)

#: Elements that count as polar. Verified against freesasa's classifier while
#: that was still a dependency: 100% agreement on a 3260-atom protein.
POLAR_ELEMENTS = frozenset({"N", "O", "S"})

#: Backbone atoms, reported separately as "main chain".
MAIN_CHAIN_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})

#: Solvent probe radius in Angstrom, the usual value for water.
DEFAULT_PROBE_RADIUS = 1.4

#: Sample points per atom sphere, the usual Shrake-Rupley default.
DEFAULT_SASA_POINTS = 100


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """``n_points`` roughly uniform unit vectors (Fibonacci lattice)."""
    from .surface.point_cloud import _fibonacci_sphere as cached

    return cached(n_points)


#: The last few Shrake-Rupley results, keyed on what they were computed from.
#: A single Protein asked for graph, atom_graph, voxel and surface runs this
#: four times on the same atoms.
_AREA_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_AREA_CACHE_MAX = 4
_AREA_CACHE_LOCK = threading.Lock()


def shrake_rupley(
    coords: np.ndarray,
    radii: np.ndarray,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    n_points: int = DEFAULT_SASA_POINTS,
) -> np.ndarray:
    """Per-atom solvent accessible surface area by point sampling.

    .. warning::

       The sample directions are fixed in space, not carried with the
       molecule, so this answer depends on how the structure is oriented.
       Rotating the example protein and measuring per-atom areas over four
       orientations: the mean spread is 43% of the atom's own area at the
       default 100 points, the worst atom varies by four times its mean, and
       362 of 3260 atoms come out as exactly zero in one orientation and
       non-zero in another. Raising ``n_points`` to 1000 -- ten times the work
       -- brings the mean spread to 18% and still leaves 133 such atoms.

       Translating a structure changes nothing at all; this is rotation alone.

       Everything plmol derives from SASA inherits it: the residue SASA block,
       ``burial_index`` and ``relative_sasa`` on the atom graph, the surface
       point cloud's burial channel, and the voxel's. It is a property of
       point-sampled SASA rather than of this implementation, and the cure is
       a lattice oriented by the molecule rather than by the axes.

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
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise InputError(f"coords must be (N, 3), got {coords.shape}.")
    n_atoms = coords.shape[0]
    if n_atoms == 0:
        return np.zeros(0, dtype=np.float64)

    expanded = np.asarray(radii, dtype=np.float32) + probe_radius
    key = (n_atoms, n_points, _digest(coords), _digest(expanded))
    with _AREA_CACHE_LOCK:
        cached = _AREA_CACHE.get(key)
        if cached is not None:
            _AREA_CACHE.move_to_end(key)
            return cached
    exposed = sphere_point_exposure(
        coords,
        expanded,
        np.full(n_atoms, n_points, dtype=np.int64),
        _fibonacci_sphere,
    )
    counts = exposed.reshape(n_atoms, n_points).sum(axis=1)
    areas = 4.0 * np.pi * expanded.astype(np.float64) ** 2 * (counts / n_points)
    areas.setflags(write=False)
    with _AREA_CACHE_LOCK:
        _AREA_CACHE[key] = areas
        while len(_AREA_CACHE) > _AREA_CACHE_MAX:
            _AREA_CACHE.popitem(last=False)
    return areas


def _digest(array: np.ndarray) -> bytes:
    """Content key for an array, so a recycled buffer cannot be served stale."""
    return blake2b(np.ascontiguousarray(array), digest_size=16).digest()


# ---------------------------------------------------------------------------
# freesasa-shaped results, computed natively
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidueArea:
    """Per-residue areas, absolute and relative.

    ``relativeTotal`` is a fraction of ``RESIDUE_MAX_SASA``; the other four are
    fractions of :data:`RESIDUE_MAX_CLASS_SASA`, the surface each class can
    expose. 1.0 means fully exposed either way. The names are the ones freesasa
    used, and so is the per-class normalisation.
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
    """The atom table a SASA result is indexed by, from plmol's own parser."""

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
    """Per-atom and per-residue areas for one structure."""

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


def _relative(area: float, reference: Optional[float]) -> float:
    """*area* over its class reference; 0.0 when the class cannot exist.

    Glycine has no side chain, so its side-chain reference is zero and the
    relative value is zero rather than a division by it.
    """
    if not reference:
        return 0.0
    return area / reference


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

            # Each class is measured against how much of that class the
            # residue can expose, not against its total surface. Dividing
            # everything by the total would make these the same numbers as the
            # absolute areas over RESIDUE_MAX_SASA.
            classes = RESIDUE_MAX_CLASS_SASA.get(residue_name, {})
            out[chain][number] = ResidueArea(
                total=total,
                polar=polar,
                apolar=apolar,
                mainChain=main,
                sideChain=side,
                relativeTotal=total / reference,
                relativePolar=_relative(polar, classes.get("polar")),
                relativeApolar=_relative(apolar, classes.get("apolar")),
                relativeMainChain=_relative(main, classes.get("mainChain")),
                relativeSideChain=_relative(side, classes.get("sideChain")),
            )
    return out


def native_structure_result(
    pdb_file: str,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    n_points: int = DEFAULT_SASA_POINTS,
) -> Tuple[NativeSasaStructure, NativeSasaResult]:
    """Parse a PDB and compute its SASA.

    Atoms come from plmol's own ``PDBParser``, so the atom set matches every
    other feature the library derives from the same file.

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
    """Whether an element counts as polar."""
    return (element or "").strip().upper()[:1] in POLAR_ELEMENTS
