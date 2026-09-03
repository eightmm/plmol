"""Which nucleotides are paired with which.

A nucleic acid graph built from distance alone cannot tell a base pair from
any other close contact, and the pairing is most of what the molecule is: it
says which strands go together, where a helix runs, and which bases are free.
The constants for it have been in ``constants/nucleic_acids.py`` all along --
this is the code that uses them.

A pair is recognised by its hydrogen bonds, not by sequence: the purine's N1
to the pyrimidine's N3 is the anchor, at least one of the pair's other
canonical bonds has to be there too, and the two bases have to be roughly
coplanar. Stacked bases are coplanar as well, which is why the bond distance
is checked first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..arrays import FLOAT
from ..constants import (
    BASE_ATOMS,
    PURINES,
    PYRIMIDINES,
    WC_BASE_PAIRS,
    WC_HBOND_ATOMS,
    WC_HBOND_MAX_DISTANCE,
    WC_MAX_PLANE_ANGLE,
)


@dataclass(frozen=True)
class BasePair:
    """One Watson-Crick pair.

    Attributes:
        purine_index: Index of the purine in the residue list given.
        pyrimidine_index: Index of its partner.
        kind: ``"AT"``, ``"AU"`` or ``"GC"``.
        hbond_distances: The canonical bond lengths found, in Angstrom, in the
            order :data:`WC_HBOND_ATOMS` lists them. A bond that is missing or
            beyond the cutoff is ``inf``.
        plane_angle: Angle between the two base planes, in degrees.
        c1_distance: Distance between the two C1' atoms, in Angstrom. A
            canonical pair is near 10.5; ``inf`` when either is absent.
    """

    purine_index: int
    pyrimidine_index: int
    kind: str
    hbond_distances: Tuple[float, ...]
    plane_angle: float
    c1_distance: float


def find_base_pairs(
    residues: Sequence[Dict],
    *,
    max_hbond_distance: float = WC_HBOND_MAX_DISTANCE,
    max_plane_angle: float = WC_MAX_PLANE_ANGLE,
) -> List[BasePair]:
    """Every Watson-Crick pair among *residues*.

    Args:
        residues: Nucleotide dicts as the featurizer builds them, each with
            ``res_name`` and an ``atoms`` mapping of name to coordinate.
        max_hbond_distance: Longest a canonical hydrogen bond may be.
        max_plane_angle: How far from coplanar the two bases may be, in
            degrees.

    Returns:
        Pairs in residue order, each listed once with the purine first. A base
        appears in at most one pair -- the best one it has.
    """
    bases = [_base_geometry(residue) for residue in residues]

    candidates: List[BasePair] = []
    for purine in range(len(residues)):
        if bases[purine] is None or bases[purine][1] not in PURINES:
            continue
        for pyrimidine in range(len(residues)):
            if bases[pyrimidine] is None or bases[pyrimidine][1] not in PYRIMIDINES:
                continue
            pair = _evaluate(
                purine, pyrimidine, bases, residues,
                max_hbond_distance, max_plane_angle,
            )
            if pair is not None:
                candidates.append(pair)

    # A base pairs with one partner. Where several fit, the shortest anchor
    # bond wins, which is how a crystallographer would break the tie too.
    candidates.sort(key=lambda pair: pair.hbond_distances[0])
    taken = set()
    chosen = []
    for pair in candidates:
        if pair.purine_index in taken or pair.pyrimidine_index in taken:
            continue
        taken.add(pair.purine_index)
        taken.add(pair.pyrimidine_index)
        chosen.append(pair)
    chosen.sort(key=lambda pair: (pair.purine_index, pair.pyrimidine_index))
    return chosen


def _base_geometry(residue: Dict):
    """``(one-letter code, residue name, ring coordinates, plane normal)``.

    Both names are kept: the hydrogen-bond table is written in one-letter
    codes, while PURINES and PYRIMIDINES list residue names, and thymine is
    only ever "DT" -- there is no lone "T" in either set.
    """
    # The canonical base, so a modified or legacy-named residue still pairs.
    name = residue.get("base") or residue.get("res_name", "")
    if name not in BASE_ATOMS:
        return None
    letter = name[-1] if name.startswith("D") else name
    ring = np.array(
        [residue["atoms"][atom] for atom in BASE_ATOMS[name] if atom in residue["atoms"]],
        dtype=FLOAT,
    )
    if len(ring) < 3:
        return None
    # The base is planar, so the smallest principal axis of its ring atoms is
    # the normal.
    centred = ring - ring.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    return letter, name, ring, normal


def _evaluate(purine, pyrimidine, bases, residues, max_distance, max_angle):
    """The pair these two would make, or None if they do not make one."""
    purine_letter = bases[purine][0]
    pyrimidine_letter = bases[pyrimidine][0]
    if WC_BASE_PAIRS.get(purine_letter) != pyrimidine_letter:
        return None
    bonds = WC_HBOND_ATOMS.get((purine_letter, pyrimidine_letter))
    if bonds is None:
        return None

    distances = []
    for purine_atom, pyrimidine_atom in bonds:
        left = residues[purine]["atoms"].get(purine_atom)
        right = residues[pyrimidine]["atoms"].get(pyrimidine_atom)
        if left is None or right is None:
            distances.append(float("inf"))
            continue
        distances.append(float(np.linalg.norm(np.asarray(left) - np.asarray(right))))

    # The anchor has to be there, and so does one of the others: a single
    # close contact is a contact, not a pair.
    if distances[0] > max_distance:
        return None
    if not any(distance <= max_distance for distance in distances[1:]):
        return None

    cosine = abs(float(np.dot(bases[purine][3], bases[pyrimidine][3])))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    if angle > max_angle:
        return None

    left_c1 = residues[purine]["atoms"].get("C1'")
    right_c1 = residues[pyrimidine]["atoms"].get("C1'")
    c1_distance = (
        float(np.linalg.norm(np.asarray(left_c1) - np.asarray(right_c1)))
        if left_c1 is not None and right_c1 is not None
        else float("inf")
    )

    return BasePair(
        purine_index=purine,
        pyrimidine_index=pyrimidine,
        kind=purine_letter + pyrimidine_letter,
        hbond_distances=tuple(distances),
        plane_angle=angle,
        c1_distance=c1_distance,
    )


def base_pair_arrays(pairs: Sequence[BasePair], num_residues: int) -> Dict[str, np.ndarray]:
    """Pairs as the arrays a graph carries alongside its edges.

    Returns:
        ``pair_index`` ``(2, P)`` purine then pyrimidine, ``pair_kind`` ``(P,)``
        as 0 for AT, 1 for AU, 2 for GC, ``pair_c1_distance`` ``(P,)``,
        ``pair_plane_angle`` ``(P,)``, and ``is_paired`` ``(N,)``.
    """
    kinds = {"AT": 0, "AU": 1, "GC": 2}
    index = np.array(
        [[pair.purine_index for pair in pairs], [pair.pyrimidine_index for pair in pairs]],
        dtype=np.int64,
    ).reshape(2, len(pairs))
    is_paired = np.zeros(num_residues, dtype=FLOAT)
    for pair in pairs:
        is_paired[pair.purine_index] = 1.0
        is_paired[pair.pyrimidine_index] = 1.0
    return {
        "pair_index": index,
        "pair_kind": np.array([kinds[pair.kind] for pair in pairs], dtype=np.int64),
        "pair_c1_distance": np.array(
            [pair.c1_distance for pair in pairs], dtype=FLOAT
        ),
        "pair_plane_angle": np.array(
            [pair.plane_angle for pair in pairs], dtype=FLOAT
        ),
        "is_paired": is_paired,
    }
