"""
NucleicFeaturizer — residue-level and atom-level features for nucleic acids.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..protein.utils import PDBParser, ParsedAtom
from ..utils import dihedral_angles
from ..constants import (
    NUCLEOTIDE_TOKEN,
    NUCLEOTIDE_TYPES,
    NUM_NUCLEOTIDE_TYPES,
    STANDARD_NUCLEOTIDE_ATOMS,
    NUCLEOTIDE_MAX_SASA,
    NUCLEOTIDE_PROPERTIES,
    PURINES,
    PYRIMIDINES,
    DNA_RESIDUES,
    RNA_RESIDUES,
)


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector from a to b."""
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _dihedral(p0: np.ndarray, p1: np.ndarray,
              p2: np.ndarray, p3: np.ndarray) -> float:
    """Dihedral angle (radians) defined by four points."""
    stacked = [np.asarray(p, dtype=np.float64).reshape(1, 3) for p in (p0, p1, p2, p3)]
    return float(dihedral_angles(*stacked)[0])


def _angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Bond angle (radians) at p1."""
    v1 = p0 - p1
    v2 = p2 - p1
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)))


class NucleicFeaturizer:
    """
    Featurizes nucleic acid structures from a parsed PDB file.

    Provides:
    - Sequence features (tokens, GC content, purine/pyrimidine)
    - Residue-level graph with backbone torsions and base properties
    - Backbone geometry (sugar-phosphate)
    - Atom-level graph with token-based features
    """

    def __init__(self, pdb_path: str, chain_id: Optional[str] = None):
        self.pdb_path = pdb_path
        self.chain_id = chain_id
        self._parser = PDBParser(pdb_path, include_nucleic_acids=True)
        self._residues: Optional[List[Dict]] = None

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_na_residues(self) -> List[Dict]:
        """Return ordered list of nucleotide residue dicts from parsed atoms."""
        if self._residues is not None:
            return self._residues

        # Group nucleic-acid atoms by (chain, resnum, resname)
        residue_map: Dict[Tuple, Dict] = {}
        for atom in self._parser.protein_atoms:
            if atom.res_name not in (DNA_RESIDUES | RNA_RESIDUES):
                continue
            if self.chain_id is not None and atom.chain_id != self.chain_id:
                continue
            key = (atom.chain_id, atom.res_num, atom.insertion_code, atom.res_name)
            if key not in residue_map:
                residue_map[key] = {
                    "chain_id": atom.chain_id,
                    "res_num": atom.res_num,
                    "res_name": atom.res_name,
                    "atoms": {},
                }
            residue_map[key]["atoms"][atom.atom_name] = atom.coords

        # Sort by (chain, res_num, insertion_code)
        self._residues = [
            v for _, v in sorted(residue_map.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
        ]
        return self._residues

    def _coord(self, res: Dict, atom_name: str) -> Optional[np.ndarray]:
        c = res["atoms"].get(atom_name)
        return np.array(c, dtype=np.float32) if c is not None else None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_sequence(self) -> str:
        """Return the nucleotide sequence as a string of 1-letter codes."""
        from ..constants import NUCLEOTIDE_3TO1
        residues = self._get_na_residues()
        return "".join(NUCLEOTIDE_3TO1.get(r["res_name"], "N") for r in residues)

    def get_sequence_features(self) -> Dict[str, Any]:
        """
        Sequence-level features.

        Returns:
            tokens: (N,) int array — NUCLEOTIDE_TOKEN index
            is_purine: (N,) float array
            is_pyrimidine: (N,) float array
            gc_content: float scalar
            res_names: list of residue names
        """
        residues = self._get_na_residues()
        n = len(residues)

        tokens = np.zeros(n, dtype=np.int64)
        is_purine = np.zeros(n, dtype=np.float32)
        is_pyrimidine = np.zeros(n, dtype=np.float32)
        gc_count = 0

        for i, res in enumerate(residues):
            rname = res["res_name"]
            tokens[i] = NUCLEOTIDE_TOKEN.get(rname, NUCLEOTIDE_TOKEN["UNK_NT"])
            is_purine[i] = float(rname in PURINES)
            is_pyrimidine[i] = float(rname in PYRIMIDINES)
            if rname in ("DG", "DC", "G", "C"):
                gc_count += 1

        gc_content = gc_count / n if n > 0 else 0.0

        return {
            "tokens": tokens,
            "is_purine": is_purine,
            "is_pyrimidine": is_pyrimidine,
            "gc_content": float(gc_content),
            "res_names": [r["res_name"] for r in residues],
        }

    def _compute_torsions(self, residues: List[Dict]) -> np.ndarray:
        """
        Compute backbone torsion angles for each residue.

        Returns (N, 7) float32 array:
          [alpha, beta, gamma, delta, epsilon, zeta, chi]
        All angles in radians; 0.0 if atoms missing.
        """
        n = len(residues)
        torsions = np.zeros((n, 7), dtype=np.float32)

        def c(res: Dict, name: str) -> Optional[np.ndarray]:
            return self._coord(res, name)

        for i, res in enumerate(residues):
            prev = residues[i - 1] if i > 0 else None
            nxt = residues[i + 1] if i < n - 1 else None

            # alpha: O3'(i-1) - P(i) - O5'(i) - C5'(i)
            if prev is not None:
                p0 = c(prev, "O3'")
                p1 = c(res, "P")
                p2 = c(res, "O5'")
                p3 = c(res, "C5'")
                if all(x is not None for x in [p0, p1, p2, p3]):
                    torsions[i, 0] = _dihedral(p0, p1, p2, p3)

            # beta: P - O5' - C5' - C4'
            p0 = c(res, "P"); p1 = c(res, "O5'"); p2 = c(res, "C5'"); p3 = c(res, "C4'")
            if all(x is not None for x in [p0, p1, p2, p3]):
                torsions[i, 1] = _dihedral(p0, p1, p2, p3)

            # gamma: O5' - C5' - C4' - C3'
            p0 = c(res, "O5'"); p1 = c(res, "C5'"); p2 = c(res, "C4'"); p3 = c(res, "C3'")
            if all(x is not None for x in [p0, p1, p2, p3]):
                torsions[i, 2] = _dihedral(p0, p1, p2, p3)

            # delta: C5' - C4' - C3' - O3'
            p0 = c(res, "C5'"); p1 = c(res, "C4'"); p2 = c(res, "C3'"); p3 = c(res, "O3'")
            if all(x is not None for x in [p0, p1, p2, p3]):
                torsions[i, 3] = _dihedral(p0, p1, p2, p3)

            # epsilon: C4' - C3' - O3' - P(i+1)
            if nxt is not None:
                p0 = c(res, "C4'"); p1 = c(res, "C3'"); p2 = c(res, "O3'"); p3 = c(nxt, "P")
                if all(x is not None for x in [p0, p1, p2, p3]):
                    torsions[i, 4] = _dihedral(p0, p1, p2, p3)

            # zeta: C3' - O3' - P(i+1) - O5'(i+1)
            if nxt is not None:
                p0 = c(res, "C3'"); p1 = c(res, "O3'"); p2 = c(nxt, "P"); p3 = c(nxt, "O5'")
                if all(x is not None for x in [p0, p1, p2, p3]):
                    torsions[i, 5] = _dihedral(p0, p1, p2, p3)

            # chi (glycosidic): O4' - C1' - N9/N1 - C4(purine) or C2(pyrimidine)
            rname = res["res_name"]
            if rname in PURINES:
                chi_atoms = ("O4'", "C1'", "N9", "C4")
            else:
                chi_atoms = ("O4'", "C1'", "N1", "C2")
            p0 = c(res, chi_atoms[0]); p1 = c(res, chi_atoms[1])
            p2 = c(res, chi_atoms[2]); p3 = c(res, chi_atoms[3])
            if all(x is not None for x in [p0, p1, p2, p3]):
                torsions[i, 6] = _dihedral(p0, p1, p2, p3)

        return torsions

    def _compute_sugar_pucker(self, residues: List[Dict]) -> np.ndarray:
        """
        Estimate sugar pucker mode per residue.

        Returns (N,) float32: 1.0 = C3'-endo, 0.0 = C2'-endo, 0.5 = unknown.

        Determined by the pseudo-rotation phase angle P. C3'-endo: P near 0°,
        C2'-endo: P near 160°. We approximate by the C4'-O4'-C1' reference.
        """
        n = len(residues)
        pucker = np.full(n, 0.5, dtype=np.float32)

        for i, res in enumerate(residues):
            # Use delta torsion as a proxy: delta < 115° → C3'-endo
            c5 = self._coord(res, "C5'")
            c4 = self._coord(res, "C4'")
            c3 = self._coord(res, "C3'")
            o3 = self._coord(res, "O3'")
            if any(x is None for x in [c5, c4, c3, o3]):
                continue
            delta = _dihedral(c5, c4, c3, o3)
            delta_deg = np.degrees(delta)
            # C3'-endo: delta ~ 85°, C2'-endo: delta ~ 145°
            pucker[i] = 1.0 if delta_deg < 115.0 else 0.0

        return pucker

    def get_graph(self, distance_cutoff: float = 8.0) -> Dict[str, Any]:
        """
        Residue-level graph with backbone torsion angles and base properties.

        Node features (per residue):
          - nucleotide_type: (N,) int token
          - one_hot: (N, NUM_NUCLEOTIDE_TYPES) float
          - is_purine: (N,) float
          - is_pyrimidine: (N,) float
          - is_dna: (N,) float
          - torsions: (N, 7) float — alpha, beta, gamma, delta, epsilon, zeta, chi
          - sugar_pucker: (N,) float — C3'-endo=1, C2'-endo=0
          - mol_weight: (N,) float (normalized)
          - n_hbond_donors: (N,) float
          - n_hbond_acceptors: (N,) float
          - coords: (N, 3) float — C1' atom coordinates (or centroid fallback)

        Edge features:
          - edge_index: (2, E) int
          - edge_attr: (E, 3) float — [is_sequential, distance, 1/distance]
        """
        residues = self._get_na_residues()
        n = len(residues)

        # Node features
        tokens = np.zeros(n, dtype=np.int64)
        one_hot = np.zeros((n, NUM_NUCLEOTIDE_TYPES), dtype=np.float32)
        is_purine = np.zeros(n, dtype=np.float32)
        is_pyrimidine = np.zeros(n, dtype=np.float32)
        is_dna = np.zeros(n, dtype=np.float32)
        mol_weight = np.zeros(n, dtype=np.float32)
        n_donors = np.zeros(n, dtype=np.float32)
        n_acceptors = np.zeros(n, dtype=np.float32)
        coords = np.zeros((n, 3), dtype=np.float32)

        for i, res in enumerate(residues):
            rname = res["res_name"]
            tok = NUCLEOTIDE_TOKEN.get(rname, NUCLEOTIDE_TOKEN["UNK_NT"])
            tokens[i] = tok
            one_hot[i, tok] = 1.0
            is_purine[i] = float(rname in PURINES)
            is_pyrimidine[i] = float(rname in PYRIMIDINES)
            is_dna[i] = float(rname in DNA_RESIDUES)

            props = NUCLEOTIDE_PROPERTIES.get(rname, NUCLEOTIDE_PROPERTIES.get("UNK_NT", (320.0, 1, 3, 0.5)))
            mol_weight[i] = props[0] / 400.0  # normalize ~[0,1]
            n_donors[i] = float(props[1])
            n_acceptors[i] = float(props[2])

            # C1' as representative position; fallback to centroid
            c1 = self._coord(res, "C1'")
            if c1 is not None:
                coords[i] = c1
            elif res["atoms"]:
                coords[i] = np.mean(list(res["atoms"].values()), axis=0)

        torsions = self._compute_torsions(residues)
        sugar_pucker = self._compute_sugar_pucker(residues)

        # Build edges: sequential + spatial within distance_cutoff
        src, dst, attrs = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                is_seq = 1.0 if j == i + 1 else 0.0
                if is_seq or dist <= distance_cutoff:
                    inv_dist = 1.0 / (dist + 1e-8)
                    src += [i, j]
                    dst += [j, i]
                    attrs += [[is_seq, dist, inv_dist], [is_seq, dist, inv_dist]]

        edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.array(attrs, dtype=np.float32) if attrs else np.zeros((0, 3), dtype=np.float32)

        return {
            "nucleotide_type": tokens,
            "one_hot": one_hot,
            "is_purine": is_purine,
            "is_pyrimidine": is_pyrimidine,
            "is_dna": is_dna,
            "torsions": torsions,
            "sugar_pucker": sugar_pucker,
            "mol_weight": mol_weight,
            "n_hbond_donors": n_donors,
            "n_hbond_acceptors": n_acceptors,
            "coords": coords,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": n,
        }

    def get_backbone(self) -> Dict[str, Any]:
        """
        Sugar-phosphate backbone geometry.

        Returns per-residue arrays of backbone atom coordinates.
        Missing atoms are filled with NaN.
        """
        residues = self._get_na_residues()
        n = len(residues)

        backbone_atoms = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'"]
        n_atoms = len(backbone_atoms)
        backbone_coords = np.full((n, n_atoms, 3), np.nan, dtype=np.float32)

        for i, res in enumerate(residues):
            for j, aname in enumerate(backbone_atoms):
                c = self._coord(res, aname)
                if c is not None:
                    backbone_coords[i, j] = c

        return {
            "backbone_coords": backbone_coords,
            "backbone_atom_names": backbone_atoms,
            "num_residues": n,
        }

    def get_atom_graph(self, distance_cutoff: float = 4.5) -> Dict[str, Any]:
        """
        Atom-level graph with token-based features.

        Node features:
          - atom_tokens: (A,) int — (residue_idx * max_atoms + atom_idx_in_residue)
          - residue_token: (A,) int — NUCLEOTIDE_TOKEN for parent residue
          - coords: (A, 3) float
          - atom_to_residue: (A,) int — which residue each atom belongs to

        Edge features:
          - edge_index: (2, E) int
          - edge_distances: (E,) float
        """
        residues = self._get_na_residues()

        all_coords: List[np.ndarray] = []
        residue_tokens: List[int] = []
        atom_to_residue: List[int] = []
        residue_atom_indices: List[List[int]] = []

        atom_idx = 0
        for res_idx, res in enumerate(residues):
            rname = res["res_name"]
            rtok = NUCLEOTIDE_TOKEN.get(rname, NUCLEOTIDE_TOKEN["UNK_NT"])
            standard_order = STANDARD_NUCLEOTIDE_ATOMS.get(rname, list(res["atoms"].keys()))
            res_atoms = []

            for aname in standard_order:
                c = self._coord(res, aname)
                if c is None:
                    continue
                all_coords.append(c)
                residue_tokens.append(rtok)
                atom_to_residue.append(res_idx)
                res_atoms.append(atom_idx)
                atom_idx += 1

            residue_atom_indices.append(res_atoms)

        n_atoms = len(all_coords)
        if n_atoms == 0:
            return {
                "coords": np.zeros((0, 3), dtype=np.float32),
                "residue_token": np.zeros(0, dtype=np.int64),
                "atom_to_residue": np.zeros(0, dtype=np.int64),
                "residue_atom_indices": residue_atom_indices,
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "edge_distances": np.zeros(0, dtype=np.float32),
                "num_atoms": 0,
            }

        coords_arr = np.stack(all_coords, axis=0)

        # Build spatial edges within cutoff
        src, dst, dists = [], [], []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
                if d <= distance_cutoff:
                    src += [i, j]
                    dst += [j, i]
                    dists += [d, d]

        edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
        edge_distances = np.array(dists, dtype=np.float32) if dists else np.zeros(0, dtype=np.float32)

        return {
            "coords": coords_arr,
            "residue_token": np.array(residue_tokens, dtype=np.int64),
            "atom_to_residue": np.array(atom_to_residue, dtype=np.int64),
            "residue_atom_indices": residue_atom_indices,
            "edge_index": edge_index,
            "edge_distances": edge_distances,
            "num_atoms": n_atoms,
        }
