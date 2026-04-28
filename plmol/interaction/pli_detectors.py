"""
Protein-Ligand Interaction Detectors.

Detects various types of protein-ligand interactions (hydrogen bonds,
salt bridges, pi-stacking, etc.) from precomputed pharmacophore and
coordinate data.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from rdkit import Chem

from ..constants import (
    INTERACTION_TYPES,
    PI_STACK_PARALLEL_MAX,
    PI_STACK_PARALLEL_MIN_ANTI,
    PI_STACK_PERP_MIN,
    PI_STACK_PERP_MAX,
    CATION_PI_MAX_OFFSET_ANGLE,
)
from ..utils import knn_mask_bipartite_numpy
from .pli_featurizer import Interaction


class InteractionDetector:
    """Detects protein-ligand interactions from pharmacophore annotations.

    Args:
        protein_coords: (N_p, 3) heavy atom coordinates for protein.
        ligand_coords: (N_l, 3) heavy atom coordinates for ligand.
        distance_matrix: (N_p, N_l) pairwise distance matrix.
        protein_pharmacophores: {category: set of heavy atom indices}.
        ligand_pharmacophores: {category: set of heavy atom indices}.
        num_protein_atoms: Number of protein heavy atoms.
        num_ligand_atoms: Number of ligand heavy atoms.
        protein_with_h: RDKit mol with explicit H (for angle calculations).
        ligand_with_h: RDKit mol with explicit H.
        protein_coords_with_h: Full coordinates including H.
        ligand_coords_with_h: Full coordinates including H.
        protein_heavy_to_full: heavy_idx -> full_idx mapping.
        ligand_heavy_to_full: heavy_idx -> full_idx mapping.
        protein_full_to_heavy: full_idx -> heavy_idx mapping.
        ligand_full_to_heavy: full_idx -> heavy_idx mapping.
        protein_h_neighbors: {full_idx: [h_full_idx, ...]}.
        ligand_h_neighbors: {full_idx: [h_full_idx, ...]}.
        distance_cutoff: Maximum distance for interaction detection.
        knn_cutoff: Optional k-nearest neighbors cutoff.
    """

    def __init__(
        self,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
        distance_matrix: np.ndarray,
        protein_pharmacophores: Dict[str, Set[int]],
        ligand_pharmacophores: Dict[str, Set[int]],
        num_protein_atoms: int,
        num_ligand_atoms: int,
        protein_with_h: Chem.Mol,
        ligand_with_h: Chem.Mol,
        protein_coords_with_h: np.ndarray,
        ligand_coords_with_h: np.ndarray,
        protein_heavy_to_full: Dict[int, int],
        ligand_heavy_to_full: Dict[int, int],
        protein_full_to_heavy: Dict[int, int],
        ligand_full_to_heavy: Dict[int, int],
        protein_h_neighbors: Dict[int, List[int]],
        ligand_h_neighbors: Dict[int, List[int]],
        distance_cutoff: float = 4.5,
        knn_cutoff: Optional[int] = None,
    ):
        self._protein_coords = protein_coords
        self._ligand_coords = ligand_coords
        self._distance_matrix = distance_matrix
        self._protein_pharmacophores = protein_pharmacophores
        self._ligand_pharmacophores = ligand_pharmacophores
        self.num_protein_atoms = num_protein_atoms
        self.num_ligand_atoms = num_ligand_atoms
        self._protein_with_h = protein_with_h
        self._ligand_with_h = ligand_with_h
        self._protein_coords_with_h = protein_coords_with_h
        self._ligand_coords_with_h = ligand_coords_with_h
        self._protein_heavy_to_full = protein_heavy_to_full
        self._ligand_heavy_to_full = ligand_heavy_to_full
        self._protein_full_to_heavy = protein_full_to_heavy
        self._ligand_full_to_heavy = ligand_full_to_heavy
        self._protein_h_neighbors = protein_h_neighbors
        self._ligand_h_neighbors = ligand_h_neighbors
        self.distance_cutoff = distance_cutoff
        self.knn_cutoff = knn_cutoff
        self._cache: Dict[str, Any] = {}

        # Precompute aromatic rings (used by pi_stacking and cation_pi)
        self._protein_rings: Optional[List[Dict[str, Any]]] = None
        self._ligand_rings: Optional[List[Dict[str, Any]]] = None

    # =========================================================================
    # Aromatic Ring Geometry (cached)
    # =========================================================================

    @property
    def protein_aromatic_rings(self) -> List[Dict[str, Any]]:
        if self._protein_rings is None:
            self._protein_rings = self._get_aromatic_rings(
                self._protein_with_h, self._protein_coords_with_h,
                self._protein_full_to_heavy,
            )
        return self._protein_rings

    @property
    def ligand_aromatic_rings(self) -> List[Dict[str, Any]]:
        if self._ligand_rings is None:
            self._ligand_rings = self._get_aromatic_rings(
                self._ligand_with_h, self._ligand_coords_with_h,
                self._ligand_full_to_heavy,
            )
        return self._ligand_rings

    # =========================================================================
    # Geometric Angle Calculations (Using H positions)
    # =========================================================================

    def _calculate_angle(
        self, coord1: np.ndarray, coord2: np.ndarray, coord3: np.ndarray
    ) -> float:
        """Calculate angle at coord2 (in degrees)."""
        v1 = coord1 - coord2
        v2 = coord3 - coord2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle

    def _calculate_dha_angle_heavy(
        self,
        donor_heavy_idx: int,
        acceptor_coord: np.ndarray,
        is_protein_donor: bool
    ) -> Tuple[Optional[float], bool]:
        """
        Calculate D-H...A angle for hydrogen bond.

        Args:
            donor_heavy_idx: Heavy atom index of donor
            acceptor_coord: 3D coordinate of acceptor
            is_protein_donor: True if donor is from protein, False if from ligand

        Returns:
            Tuple of (angle, has_valid_angle)
        """
        if is_protein_donor:
            full_idx = self._protein_heavy_to_full[donor_heavy_idx]
            h_list = self._protein_h_neighbors.get(full_idx, [])
            coords = self._protein_coords_with_h
        else:
            full_idx = self._ligand_heavy_to_full[donor_heavy_idx]
            h_list = self._ligand_h_neighbors.get(full_idx, [])
            coords = self._ligand_coords_with_h

        if not h_list:
            return None, False

        # Calculate angle with first hydrogen (or best one)
        d_coord = coords[full_idx]
        best_angle = None

        for h_idx in h_list:
            h_coord = coords[h_idx]
            angle = self._calculate_angle(d_coord, h_coord, acceptor_coord)
            if best_angle is None or angle > best_angle:
                best_angle = angle

        return best_angle, True

    def _calculate_halogen_angle_heavy(
        self,
        halogen_heavy_idx: int,
        acceptor_coord: np.ndarray,
        is_protein_halogen: bool
    ) -> Tuple[Optional[float], bool]:
        """
        Calculate C-X...A angle for halogen bond.

        Returns:
            Tuple of (angle, has_valid_angle)
        """
        if is_protein_halogen:
            mol = self._protein_with_h
            full_idx = self._protein_heavy_to_full[halogen_heavy_idx]
            coords = self._protein_coords_with_h
        else:
            mol = self._ligand_with_h
            full_idx = self._ligand_heavy_to_full[halogen_heavy_idx]
            coords = self._ligand_coords_with_h

        # Find attached heavy atom (usually carbon)
        atom = mol.GetAtomWithIdx(full_idx)
        c_idx = None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() > 1:
                c_idx = neighbor.GetIdx()
                break

        if c_idx is None:
            return None, False

        c_coord = coords[c_idx]
        x_coord = coords[full_idx]

        angle = self._calculate_angle(c_coord, x_coord, acceptor_coord)
        return angle, True

    # =========================================================================
    # Interaction Detection Methods
    # =========================================================================

    def _get_close_pairs(self, distance_cutoff: Optional[float] = None,
                         knn_cutoff: Optional[int] = None) -> List[Tuple[int, int, float]]:
        """Get heavy atom pairs within distance cutoff, optionally unioned with bipartite kNN."""
        distance_cutoff = distance_cutoff or self.distance_cutoff
        dm = self._distance_matrix

        # Distance-based mask
        mask = dm < distance_cutoff

        # Bipartite kNN: protein->ligand + ligand->protein
        knn_cutoff = knn_cutoff or self.knn_cutoff
        if knn_cutoff is not None:
            mask = mask | knn_mask_bipartite_numpy(dm, knn_cutoff)

        pairs = []
        protein_idxs, ligand_idxs = np.where(mask)
        for p_idx, l_idx in zip(protein_idxs, ligand_idxs):
            dist = dm[p_idx, l_idx]
            pairs.append((int(p_idx), int(l_idx), float(dist)))
        return pairs

    def detect_hydrogen_bonds(self) -> List[Interaction]:
        """Detect hydrogen bonds with D-H-A angle calculation."""
        if 'hydrogen_bonds' in self._cache:
            return self._cache['hydrogen_bonds']

        interactions = []
        cutoff = INTERACTION_TYPES['hydrogen_bond']['distance_cutoff']
        angle_cutoff = INTERACTION_TYPES['hydrogen_bond'].get('angle_cutoff', 120.0)

        # Protein donor - Ligand acceptor
        for p_idx in self._protein_pharmacophores['hbond_donor']:
            for l_idx in self._ligand_pharmacophores['hbond_acceptor']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        dha_angle, has_valid = self._calculate_dha_angle_heavy(
                            p_idx, self._ligand_coords[l_idx], is_protein_donor=True
                        )

                        # Filter by angle if valid
                        if has_valid and dha_angle is not None and dha_angle < angle_cutoff:
                            continue

                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='hydrogen_bond',
                            distance=dist,
                            dha_angle=dha_angle,
                            has_valid_angle=has_valid,
                            metadata={'donor': 'protein', 'acceptor': 'ligand'}
                        ))

        # Protein acceptor - Ligand donor
        for p_idx in self._protein_pharmacophores['hbond_acceptor']:
            for l_idx in self._ligand_pharmacophores['hbond_donor']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        dha_angle, has_valid = self._calculate_dha_angle_heavy(
                            l_idx, self._protein_coords[p_idx], is_protein_donor=False
                        )

                        if has_valid and dha_angle is not None and dha_angle < angle_cutoff:
                            continue

                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='hydrogen_bond',
                            distance=dist,
                            dha_angle=dha_angle,
                            has_valid_angle=has_valid,
                            metadata={'donor': 'ligand', 'acceptor': 'protein'}
                        ))

        self._cache['hydrogen_bonds'] = interactions
        return interactions

    def detect_salt_bridges(self) -> List[Interaction]:
        """Detect salt bridges (ionic interactions)."""
        if 'salt_bridges' in self._cache:
            return self._cache['salt_bridges']

        interactions = []
        cutoff = INTERACTION_TYPES['salt_bridge']['distance_cutoff']

        for p_idx in self._protein_pharmacophores['positive_charge']:
            for l_idx in self._ligand_pharmacophores['negative_charge']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='salt_bridge',
                            distance=dist,
                            has_valid_angle=True,  # No angle needed for salt bridge
                            metadata={'protein_charge': 'positive', 'ligand_charge': 'negative'}
                        ))

        for p_idx in self._protein_pharmacophores['negative_charge']:
            for l_idx in self._ligand_pharmacophores['positive_charge']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='salt_bridge',
                            distance=dist,
                            has_valid_angle=True,
                            metadata={'protein_charge': 'negative', 'ligand_charge': 'positive'}
                        ))

        self._cache['salt_bridges'] = interactions
        return interactions

    def detect_pi_stacking(self) -> List[Interaction]:
        """Detect pi-stacking with ring angle calculation."""
        if 'pi_stacking' in self._cache:
            return self._cache['pi_stacking']

        interactions = []
        cutoff = INTERACTION_TYPES['pi_stacking']['distance_cutoff']

        for p_ring in self.protein_aromatic_rings:
            for l_ring in self.ligand_aromatic_rings:
                dist = np.linalg.norm(p_ring['center'] - l_ring['center'])
                if dist < cutoff:
                    cos_angle = abs(np.dot(p_ring['normal'], l_ring['normal']))
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                    is_parallel = angle < PI_STACK_PARALLEL_MAX or angle > PI_STACK_PARALLEL_MIN_ANTI
                    is_perpendicular = PI_STACK_PERP_MIN < angle < PI_STACK_PERP_MAX

                    if is_parallel or is_perpendicular:
                        interactions.append(Interaction(
                            protein_atom_idx=p_ring['heavy_atoms'][0],
                            ligand_atom_idx=l_ring['heavy_atoms'][0],
                            interaction_type='pi_stacking',
                            distance=dist,
                            angle=angle,
                            has_valid_angle=True,
                            metadata={
                                'stacking_type': 'parallel' if is_parallel else 'T-shaped',
                                'protein_ring_atoms': p_ring['heavy_atoms'],
                                'ligand_ring_atoms': l_ring['heavy_atoms'],
                            }
                        ))

        self._cache['pi_stacking'] = interactions
        return interactions

    def detect_cation_pi(self) -> List[Interaction]:
        """Detect cation-pi interactions."""
        if 'cation_pi' in self._cache:
            return self._cache['cation_pi']

        interactions = []
        cutoff = INTERACTION_TYPES['cation_pi']['distance_cutoff']

        # Protein cation - Ligand aromatic
        for p_idx in self._protein_pharmacophores['positive_charge']:
            if p_idx >= self.num_protein_atoms:
                continue
            p_coord = self._protein_coords[p_idx]
            for l_ring in self.ligand_aromatic_rings:
                dist = np.linalg.norm(p_coord - l_ring['center'])
                if dist < cutoff:
                    vec_to_center = l_ring['center'] - p_coord
                    vec_to_center = vec_to_center / (np.linalg.norm(vec_to_center) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(
                        abs(np.dot(vec_to_center, l_ring['normal'])), -1, 1
                    )))
                    if angle > CATION_PI_MAX_OFFSET_ANGLE:
                        continue

                    interactions.append(Interaction(
                        protein_atom_idx=p_idx,
                        ligand_atom_idx=l_ring['heavy_atoms'][0],
                        interaction_type='cation_pi',
                        distance=dist,
                        angle=angle,
                        has_valid_angle=True,
                        metadata={'cation': 'protein', 'pi': 'ligand'}
                    ))

        # Ligand cation - Protein aromatic
        for l_idx in self._ligand_pharmacophores['positive_charge']:
            if l_idx >= self.num_ligand_atoms:
                continue
            l_coord = self._ligand_coords[l_idx]
            for p_ring in self.protein_aromatic_rings:
                dist = np.linalg.norm(l_coord - p_ring['center'])
                if dist < cutoff:
                    vec_to_center = p_ring['center'] - l_coord
                    vec_to_center = vec_to_center / (np.linalg.norm(vec_to_center) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(
                        abs(np.dot(vec_to_center, p_ring['normal'])), -1, 1
                    )))
                    if angle > CATION_PI_MAX_OFFSET_ANGLE:
                        continue

                    interactions.append(Interaction(
                        protein_atom_idx=p_ring['heavy_atoms'][0],
                        ligand_atom_idx=l_idx,
                        interaction_type='cation_pi',
                        distance=dist,
                        angle=angle,
                        has_valid_angle=True,
                        metadata={'cation': 'ligand', 'pi': 'protein'}
                    ))

        self._cache['cation_pi'] = interactions
        return interactions

    def detect_hydrophobic(self) -> List[Interaction]:
        """Detect hydrophobic contacts."""
        if 'hydrophobic' in self._cache:
            return self._cache['hydrophobic']

        interactions = []
        cutoff = INTERACTION_TYPES['hydrophobic']['distance_cutoff']

        for p_idx in self._protein_pharmacophores['hydrophobic']:
            for l_idx in self._ligand_pharmacophores['hydrophobic']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='hydrophobic',
                            distance=dist,
                            has_valid_angle=True,
                        ))

        self._cache['hydrophobic'] = interactions
        return interactions

    def detect_halogen_bonds(self) -> List[Interaction]:
        """Detect halogen bonds with C-X-A angle calculation."""
        if 'halogen_bonds' in self._cache:
            return self._cache['halogen_bonds']

        interactions = []
        cutoff = INTERACTION_TYPES['halogen_bond']['distance_cutoff']
        angle_cutoff = INTERACTION_TYPES['halogen_bond'].get('angle_cutoff', 140.0)

        # Ligand halogen - Protein acceptor
        for l_idx in self._ligand_pharmacophores['halogen_bond']:
            for p_idx in self._protein_pharmacophores['hbond_acceptor']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        cxa_angle, has_valid = self._calculate_halogen_angle_heavy(
                            l_idx, self._protein_coords[p_idx], is_protein_halogen=False
                        )

                        if has_valid and cxa_angle is not None and cxa_angle < angle_cutoff:
                            continue

                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='halogen_bond',
                            distance=dist,
                            cxa_angle=cxa_angle,
                            has_valid_angle=has_valid,
                            metadata={'halogen_source': 'ligand'}
                        ))

        self._cache['halogen_bonds'] = interactions
        return interactions

    def detect_metal_coordination(self) -> List[Interaction]:
        """Detect metal coordination interactions.

        Uses element-based detection for protein metal ions (Zn, Fe, Mg, etc.)
        and SMARTS-based detection for ligand coordinating atoms (N, O, S lone pairs).
        """
        if 'metal_coordination' in self._cache:
            return self._cache['metal_coordination']

        interactions = []
        cutoff = INTERACTION_TYPES['metal_coordination']['distance_cutoff']

        # Find protein metal atoms by element type
        metal_symbols = {'Zn', 'Fe', 'Mg', 'Mn', 'Cu', 'Co', 'Ni', 'Ca', 'Na', 'K'}
        protein_metals = set()
        for atom in self._protein_with_h.GetAtoms():
            full_idx = atom.GetIdx()
            if full_idx in self._protein_full_to_heavy:
                if atom.GetSymbol() in metal_symbols:
                    protein_metals.add(self._protein_full_to_heavy[full_idx])

        # Ligand coordinating atoms: metal_coord + h_acceptor + negative_charge
        ligand_coordinators = set()
        ligand_coordinators.update(self._ligand_pharmacophores.get('metal_coord', set()))
        ligand_coordinators.update(self._ligand_pharmacophores.get('hbond_acceptor', set()))
        ligand_coordinators.update(self._ligand_pharmacophores.get('negative_charge', set()))

        for p_idx in protein_metals:
            for l_idx in ligand_coordinators:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='metal_coordination',
                            distance=dist,
                            has_valid_angle=True,
                        ))

        self._cache['metal_coordination'] = interactions
        return interactions

    def detect_all_interactions(self) -> List[Interaction]:
        """Detect all types of interactions."""
        if 'all_interactions' in self._cache:
            return self._cache['all_interactions']

        all_interactions = []
        all_interactions.extend(self.detect_hydrogen_bonds())
        all_interactions.extend(self.detect_salt_bridges())
        all_interactions.extend(self.detect_pi_stacking())
        all_interactions.extend(self.detect_cation_pi())
        all_interactions.extend(self.detect_hydrophobic())
        all_interactions.extend(self.detect_halogen_bonds())
        all_interactions.extend(self.detect_metal_coordination())

        self._cache['all_interactions'] = all_interactions
        return all_interactions

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_aromatic_rings(
        self, mol: Chem.Mol, coords: np.ndarray, full_to_heavy: Dict
    ) -> List[Dict[str, Any]]:
        """Get aromatic ring info with heavy atom indices."""
        ring_info = mol.GetRingInfo()
        rings = []

        for ring_atoms in ring_info.AtomRings():
            # Check if all ring atoms are aromatic and heavy
            is_aromatic = True
            heavy_atoms = []
            for idx in ring_atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() == 1:  # Skip if H in ring (shouldn't happen)
                    continue
                if not atom.GetIsAromatic():
                    is_aromatic = False
                    break
                if idx in full_to_heavy:
                    heavy_atoms.append(full_to_heavy[idx])

            if not is_aromatic or not heavy_atoms:
                continue

            # Get ring coordinates (heavy atoms only in full coords)
            ring_coords = coords[list(ring_atoms)]
            center = np.mean(ring_coords, axis=0)

            if len(ring_coords) >= 3:
                v1 = ring_coords[1] - ring_coords[0]
                v2 = ring_coords[2] - ring_coords[0]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 1e-8:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
            else:
                normal = np.array([0, 0, 1])

            rings.append({
                'heavy_atoms': heavy_atoms,  # Heavy atom indices
                'center': center,
                'normal': normal,
            })

        return rings
