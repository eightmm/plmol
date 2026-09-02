"""
Edge (bond and pair) feature extraction for molecular graphs.

Provides EdgeFeatureMixin with methods for computing bond-level and
pairwise features including topological bond context, pair features,
distance matrices, and distance bounds.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from typing import Dict

from ..errors import InputError
from ..constants import (
    BOND_TYPES, BOND_STEREOS, BOND_DIRS,
    NORM_CONSTANTS,
)
from ..utils import knn_mask_torch


#: Width of the bond block at the front of the dense adjacency.
BOND_FEATURE_DIM = 27

#: Width of the pair block that follows it.
PAIR_FEATURE_DIM = 10

#: Channels of the dense adjacency worth keeping once it is unrolled into a
#: bond edge list.
#:
#: The pair block describes *arbitrary* atom pairs. On a bonded pair 8 of its
#: 10 channels carry nothing new: the six shortest-path-distance bins are
#: always d=1 and ``same_fragment`` is always 1, because that is what being
#: bonded means, and ``euclid`` is the interatomic distance that channel 20
#: already gives (measured correlation 1.000000, differing only by the
#: normalising constant). ``same_ring`` and ``same_aromatic_system`` are the
#: two that say something a bond channel does not.
#:
#: The dense adjacency itself keeps all 37 channels -- they are meaningful
#: there, where non-bonded pairs exist.
BOND_VIEW_CHANNELS = (*range(BOND_FEATURE_DIM), 34, 36)

#: Channels dropped by :data:`BOND_VIEW_CHANNELS`, for documentation and tests.
BOND_VIEW_DROPPED_CHANNELS = (27, 28, 29, 30, 31, 32, 33, 35)


def bond_view_channels(width: int) -> "list[int]":
    """Channels of a dense adjacency of the given width to keep in a bond view.

    A caller may pass a narrower adjacency than the standard 37-channel one, so
    the selection is clipped rather than assumed.
    """
    return [c for c in BOND_VIEW_CHANNELS if c < width]


class EdgeFeatureMixin:
    """Mixin providing edge-level feature extraction methods.

    Requires the host class to provide:
        - self._rotatable_pattern: compiled SMARTS pattern
        - self._cache: dict for per-molecule caching
        - self.one_hot(value, allowable_set) -> list
        - self.normalize(value, min_val, max_val, clip) -> float
        - self._get_distance_matrix(mol) -> np.ndarray
    """

    def get_bond_features(self, mol) -> torch.Tensor:
        """
        Extract bond features as adjacency tensor.

        Edge Features (~27 dimensions):
            - Bond type one-hot (4): SINGLE, DOUBLE, TRIPLE, AROMATIC
            - Bond stereo one-hot (6)
            - Bond direction one-hot (5)
            - Basic bond properties (5):
                - is_aromatic, is_conjugated, is_in_ring, is_rotatable, bond_order
            - Basic pair distance (1): normalized bond length
            - Topological bond context (6):
                - betweenness, bridge-like, ring-fusion, dist-to-hetero,
                  dist-to-ring, graph-distance

        Returns:
            Tensor of shape [num_atoms, num_atoms, ~27]
        """
        num_atoms = mol.GetNumAtoms()
        num_edge_features = BOND_FEATURE_DIM

        # Precompute rotatable bonds
        rotatable_bonds = set()
        if self._rotatable_pattern:
            matches = mol.GetSubstructMatches(self._rotatable_pattern)
            for match in matches:
                if len(match) >= 2:
                    pair = tuple(sorted((match[0], match[1])))
                    rotatable_bonds.add(pair)

        coords = self._cache.get('coords')
        if coords is None:
            coords = self.get_3d_coordinates(mol)
            self._cache['coords'] = coords
        dm = self._get_distance_matrix(mol)
        ring_info = mol.GetRingInfo()

        # Precompute once for all bonds (avoid per-bond recomputation)
        hetero_indices = [i for i, atom in enumerate(mol.GetAtoms())
                        if atom.GetAtomicNum() not in [1, 6]]
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)

        norm = NORM_CONSTANTS

        adj = torch.zeros(num_atoms, num_atoms, num_edge_features)
        bond_indices = torch.triu(
            torch.tensor(Chem.GetAdjacencyMatrix(mol))
        ).nonzero()

        for src, dst in bond_indices:
            src_idx, dst_idx = src.item(), dst.item()
            bond = mol.GetBondBetweenAtoms(src_idx, dst_idx)
            features = []

            # ===== Bond Type One-hot (4) =====
            features.extend(self.one_hot(bond.GetBondType(), BOND_TYPES))
            features.extend(self.one_hot(bond.GetStereo(), BOND_STEREOS))
            features.extend(self.one_hot(bond.GetBondDir(), BOND_DIRS))

            # ===== Basic Bond Properties (5) =====
            features.extend([
                bond.GetIsAromatic(),
                bond.GetIsConjugated(),
                bond.IsInRing(),
                tuple(sorted((src_idx, dst_idx))) in rotatable_bonds,
                bond.GetBondTypeAsDouble() / 3.0,  # normalized bond order
            ])

            # ===== Basic Pair Distance (1) =====
            if coords is not None and coords.shape[0] == num_atoms:
                if torch.any(coords):
                    dist = torch.dist(coords[src_idx], coords[dst_idx]).item()
                else:
                    dist = 0.0
            else:
                dist = 0.0
            dist_norm = self.normalize(
                dist,
                min_val=norm['bond_length_min'],
                max_val=norm['bond_length_min'] + norm['bond_length_range']
            )
            features.append(dist_norm)
            features.extend(
                self._get_bond_topological_features(
                    mol, bond, src_idx, dst_idx, dm, ring_info,
                    hetero_indices, ring_atoms
                )
            )

            adj[src_idx, dst_idx] = torch.tensor(features, dtype=torch.float32)

        # Make symmetric
        return adj + adj.transpose(0, 1)

    def _get_bond_topological_features(
        self, mol, bond, src_idx: int, dst_idx: int,
        dm: np.ndarray, ring_info,
        hetero_indices: list = None, ring_atoms: set = None,
    ) -> list:
        """
        Compute topological features for a bond (6 dimensions).

        Features:
            - bond_betweenness: fraction of shortest paths using this bond
            - is_bridge: bond removal disconnects the graph
            - ring_fusion_bond: bond shared by multiple rings
            - shortest_path_to_hetero: min distance from bond to heteroatom
            - shortest_path_to_ring: min distance from bond to ring atom
            - graph_distance_normalized: position in molecular graph
        """
        num_atoms = mol.GetNumAtoms()
        features = []
        norm = NORM_CONSTANTS

        # Bond betweenness (vectorized)
        betweenness = 0.0
        if dm is not None and num_atoms > 2:
            # For each (s,t) pair, check if shortest path goes through this bond
            d_via_fwd = dm[:, src_idx:src_idx+1] + 1 + dm[dst_idx:dst_idx+1, :]  # (N, N)
            d_via_rev = dm[:, dst_idx:dst_idx+1] + 1 + dm[src_idx:src_idx+1, :]  # (N, N)
            d_via_bond = np.minimum(d_via_fwd, d_via_rev)
            on_path = np.abs(d_via_bond - dm) < 0.01
            # Only upper triangle, exclude inf pairs
            valid = np.isfinite(dm)
            count = np.sum(np.triu(on_path & valid, k=1))
            max_pairs = (num_atoms - 1) * (num_atoms - 2) / 2
            betweenness = count / max_pairs if max_pairs > 0 else 0
        features.append(min(betweenness, 1.0))

        # Is bridge bond (removal disconnects graph) - approximation
        is_bridge = not bond.IsInRing()
        features.append(float(is_bridge))

        # Ring fusion bond (shared by multiple rings)
        n_rings = ring_info.NumBondRings(bond.GetIdx())
        features.append(min(n_rings / 3.0, 1.0))

        # Shortest path to heteroatom (use precomputed indices)
        if hetero_indices is None:
            hetero_indices = [i for i, atom in enumerate(mol.GetAtoms())
                            if atom.GetAtomicNum() not in [1, 6]]
        if hetero_indices and dm is not None:
            if src_idx in hetero_indices or dst_idx in hetero_indices:
                dist_to_hetero = 0
            else:
                hetero_arr = np.array(hetero_indices)
                dist_to_hetero = min(dm[src_idx, hetero_arr].min(),
                                    dm[dst_idx, hetero_arr].min())
            features.append(min(dist_to_hetero / norm['path_length_max'], 1.0))
        else:
            features.append(1.0)

        # Shortest path to ring atom (use precomputed set)
        if ring_atoms is None:
            ring_atoms = set()
            for ring in ring_info.AtomRings():
                ring_atoms.update(ring)
        if ring_atoms and dm is not None:
            if src_idx in ring_atoms or dst_idx in ring_atoms:
                dist_to_ring = 0
            else:
                ring_arr = np.array(list(ring_atoms))
                dist_to_ring = min(dm[src_idx, ring_arr].min(),
                                  dm[dst_idx, ring_arr].min())
            features.append(min(dist_to_ring / norm['path_length_max'], 1.0))
        else:
            features.append(1.0)

        # Graph distance (normalized position)
        if dm is not None:
            avg_dist = (np.mean(dm[src_idx]) + np.mean(dm[dst_idx])) / 2
            features.append(min(avg_dist / norm['path_length_max'], 1.0))
        else:
            features.append(0.0)

        return features

    def get_pair_features(self, mol, coords: torch.Tensor) -> torch.Tensor:
        """
        Build complementary pairwise features for all atom pairs [N, N, C].

        These channels are designed to complement bond-level edge features
        instead of duplicating bond type/stereo/direction information.

        Channels (C=10):
            0-5: SPD (shortest path distance) one-hot bins
                - d==1, d==2, d==3, d==4, d==5, d>=6 or disconnected
            6: euclidean distance (normalized, if 3D else zeros)
            7: same ring membership (binary)
            8: same molecular fragment (binary)
            9: same aromatic system membership (binary)
        """
        num_atoms = mol.GetNumAtoms()
        norm = NORM_CONSTANTS

        topo_dm = self._get_distance_matrix(mol)
        spd_bins = np.zeros((num_atoms, num_atoms, 6), dtype=np.float32)
        finite_dm = np.where(np.isfinite(topo_dm), topo_dm, 1e9)
        # 0: d==1, 1: d==2, 2: d==3, 3: d==4, 4: d==5, 5: d>=6/disconnected
        spd_bins[..., 0] = (finite_dm == 1).astype(np.float32)
        spd_bins[..., 1] = (finite_dm == 2).astype(np.float32)
        spd_bins[..., 2] = (finite_dm == 3).astype(np.float32)
        spd_bins[..., 3] = (finite_dm == 4).astype(np.float32)
        spd_bins[..., 4] = (finite_dm == 5).astype(np.float32)
        spd_bins[..., 5] = (finite_dm >= 6).astype(np.float32)
        for i in range(num_atoms):
            spd_bins[i, i, :] = 0.0

        euclid_norm = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        if coords is not None and coords.shape[0] == num_atoms and torch.any(coords):
            euclid = torch.cdist(coords, coords).cpu().numpy()
            euclid_norm = np.clip(
                euclid / max(norm['dist_to_special'], 1.0),
                0.0,
                1.0,
            ).astype(np.float32)

        same_ring = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            ring_atoms = list(ring)
            for i in ring_atoms:
                for j in ring_atoms:
                    if i != j:
                        same_ring[i, j] = 1.0

        same_fragment = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        for frag in Chem.GetMolFrags(mol):
            frag_atoms = list(frag)
            for i in frag_atoms:
                for j in frag_atoms:
                    if i != j:
                        same_fragment[i, j] = 1.0

        same_aromatic_system = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        aromatic_atom_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
        if aromatic_atom_indices:
            # Build aromatic-system groups directly from original molecule graph
            # (connected components induced by aromatic atoms).
            aromatic_set = set(aromatic_atom_indices)
            visited = set()
            for start in aromatic_atom_indices:
                if start in visited:
                    continue
                stack = [start]
                comp = []
                visited.add(start)
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    atom = mol.GetAtomWithIdx(cur)
                    for nb in atom.GetNeighbors():
                        nb_idx = nb.GetIdx()
                        if nb_idx in aromatic_set and nb_idx not in visited:
                            visited.add(nb_idx)
                            stack.append(nb_idx)
                for i in comp:
                    for j in comp:
                        if i != j:
                            same_aromatic_system[i, j] = 1.0

        stacked = np.stack(
            [
                spd_bins[..., 0],
                spd_bins[..., 1],
                spd_bins[..., 2],
                spd_bins[..., 3],
                spd_bins[..., 4],
                spd_bins[..., 5],
                euclid_norm,
                same_ring,
                same_fragment,
                same_aromatic_system,
            ],
            axis=-1,
        )
        return torch.from_numpy(stacked)

    def get_distance_matrix(self, mol, coords: torch.Tensor) -> torch.Tensor:
        """
        Build pairwise Euclidean distance matrix [N, N].

        If valid 3D coordinates are unavailable, returns an all-zero matrix.
        """
        num_atoms = mol.GetNumAtoms()
        if coords is None or coords.shape[0] != num_atoms or not torch.any(coords):
            return torch.zeros((num_atoms, num_atoms), dtype=torch.float32)
        return torch.cdist(coords, coords).to(torch.float32)

    def get_distance_bounds(self, mol, coords: torch.Tensor) -> torch.Tensor:
        """
        Build pairwise distance lower/upper bounds [N, N, 2].

        Uses RDKit Distance Geometry bounds matrix:
            - lower(i,j): bounds[j,i] for i<j
            - upper(i,j): bounds[i,j] for i<j
        and symmetrizes to [N, N] matrices.
        """
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return torch.zeros((0, 0, 2), dtype=torch.float32)

        try:
            bounds = np.asarray(rdDistGeom.GetMoleculeBoundsMatrix(mol), dtype=np.float32)
        except Exception as exc:
            raise InputError("Failed to compute RDKit distance-geometry bounds matrix.") from exc

        upper = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        lower = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        iu, ju = np.triu_indices(num_atoms, 1)

        # RDKit convention:
        # - upper bound in upper triangle (i<j)
        # - lower bound in lower triangle (j>i)
        upper_vals = bounds[iu, ju]
        lower_vals = bounds[ju, iu]

        upper[iu, ju] = upper_vals
        upper[ju, iu] = upper_vals
        lower[iu, ju] = lower_vals
        lower[ju, iu] = lower_vals

        return torch.from_numpy(np.stack([lower, upper], axis=-1))
