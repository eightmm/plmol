"""
Graph-based molecular featurization for GNN models.

This module provides atom (node) and bond (edge) feature extraction
for molecular graph representations.
"""

import logging
import warnings
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, rdPartialCharges, AllChem, rdDistGeom
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Suppress RDKit C++ warnings (e.g., "Molecule does not have explicit Hs")
RDLogger.DisableLog('rdApp.*')

from ..constants import (
    ATOM_TYPES, PERIODS, GROUPS, DEGREES, HEAVY_DEGREES, VALENCES,
    TOTAL_HS, HYBRIDIZATION_TYPES as HYBRIDIZATIONS, BOND_TYPES, BOND_STEREOS, BOND_DIRS,
    PERIODIC_TABLE, ELECTRONEGATIVITY,
    VDW_RADIUS, COVALENT_RADIUS, IONIZATION_ENERGY, POLARIZABILITY, VALENCE_ELECTRONS,
    CHEMICAL_SMARTS, ROTATABLE_BOND_SMARTS,
    DEFAULT_VDW_RADIUS, DEFAULT_COVALENT_RADIUS, DEFAULT_IONIZATION_ENERGY,
    DEFAULT_POLARIZABILITY, DEFAULT_VALENCE_ELECTRONS, DEFAULT_ELECTRONEGATIVITY,
    NORM_CONSTANTS,
)


class MoleculeGraphFeaturizer:
    """
    Extracts graph-level features (node and edge) from molecules.

    This class provides methods to convert molecules into graph representations
    suitable for Graph Neural Networks (GNNs).

    Node Features (~78 dimensions):
        - Atom identity (symbol one-hot)
        - Period/group one-hot and electronegativity
        - Formal charge (scalar + compact one-hot)
        - Hybridization (one-hot)
        - Aromaticity and ring membership
        - Radical electron count (scalar)
        - Total Hs (one-hot + scalar) and degree (one-hot + scalar)
        - Essential physical properties (mass, VdW radius)
        - Partial charges (Gasteiger)
        - Stereochemistry context
        - Physical properties (atomic context)
        - Topological context
        - SMARTS functional group matches

    Edge Features (~27 dimensions):
        - Bond type (one-hot, 4)
        - Bond stereo (one-hot, 6)
        - Bond direction (one-hot, 5)
        - Aromaticity, conjugation, ring membership, rotatability, bond order (5)
        - Basic pair distance (1)
        - Topological bond context (6)
    """

    def __init__(self):
        """Initialize the graph featurizer."""
        self._smarts_patterns = {
            k: Chem.MolFromSmarts(v) for k, v in CHEMICAL_SMARTS.items()
        }
        self._rotatable_pattern = Chem.MolFromSmarts(ROTATABLE_BOND_SMARTS)
        # Cache for per-molecule computed values
        self._cache = {}

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def one_hot(value, allowable_set: list) -> list:
        """
        Create one-hot encoding for a value.

        If value is not in allowable_set, maps to the last element (UNK).
        """
        if value not in allowable_set:
            value = allowable_set[-1]
        return [value == s for s in allowable_set]

    @staticmethod
    def normalize(value: float, min_val: float = 0.0, max_val: float = 1.0,
                  clip: bool = True) -> float:
        """Normalize value to [0, 1] range."""
        result = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0
        if clip:
            result = max(0.0, min(1.0, result))
        return result

    def _clear_cache(self):
        """Clear the per-molecule cache."""
        self._cache = {}

    def _get_gasteiger_charges(self, mol) -> dict:
        """
        Compute and cache Gasteiger partial charges.

        Returns:
            Dictionary mapping atom index to charge value (clipped to [-1, 1])
        """
        cache_key = 'gasteiger_charges'
        if cache_key in self._cache:
            return self._cache[cache_key]

        charges = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rdPartialCharges.ComputeGasteigerCharges(mol)

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            try:
                charge = float(atom.GetProp('_GasteigerCharge'))
                if np.isnan(charge) or np.isinf(charge):
                    charge = 0.0
            except (KeyError, ValueError, RuntimeError):
                charge = 0.0
            charges[idx] = max(-1.0, min(1.0, charge))

        self._cache[cache_key] = charges
        return charges

    def _get_distance_matrix(self, mol) -> np.ndarray:
        """
        Compute and cache distance matrix.

        Returns:
            numpy array of shape [num_atoms, num_atoms]
        """
        cache_key = 'distance_matrix'
        if cache_key in self._cache:
            return self._cache[cache_key]

        dm = Chem.GetDistanceMatrix(mol)
        self._cache[cache_key] = dm
        return dm

    # =========================================================================
    # Ring Analysis
    # =========================================================================

    def get_ring_info(self, mol) -> Tuple[Dict, Dict]:
        """
        Get ring membership information for atoms and bonds.

        Returns:
            Tuple of (atom_rings, bond_rings) where each maps index to list of ring sizes
        """
        ring_info = mol.GetRingInfo()
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        atom_rings = {i: [] for i in range(num_atoms)}
        bond_rings = {i: [] for i in range(num_bonds)}

        for ring in ring_info.AtomRings():
            for atom_idx in ring:
                atom_rings[atom_idx].append(len(ring))

        for ring in ring_info.BondRings():
            for bond_idx in ring:
                bond_rings[bond_idx].append(len(ring))

        return atom_rings, bond_rings

    def encode_ring_features(self, ring_sizes: list, is_aromatic: bool) -> list:
        """
        Encode ring membership features (21 dimensions).

        Features:
            - is_in_ring (1)
            - is_aromatic (1)
            - num_rings normalized (1)
            - ring size flags 3-8+ (6)
            - one-hot num_rings 0-4 (5)
            - one-hot smallest ring size (7)
        """
        is_in_ring = len(ring_sizes) > 0
        num_rings = min(len(ring_sizes), 4)
        smallest = min(ring_sizes) if ring_sizes else 0

        # Ring size flags (3-8+)
        size_flags = [False] * 6
        for size in ring_sizes:
            if 3 <= size <= 8:
                size_flags[size - 3] = True
            elif size > 8:
                size_flags[5] = True

        return (
            [is_in_ring, is_aromatic, num_rings / 4.0] +  # normalized
            size_flags +
            self.one_hot(num_rings, [0, 1, 2, 3, 4]) +
            self.one_hot(smallest, [0, 3, 4, 5, 6, 7, 8])
        )

    # =========================================================================
    # Atom Feature Extraction
    # =========================================================================

    def get_degree_info(self, mol) -> Dict[int, Dict]:
        """
        Compute degree-related features for all atoms.

        Returns:
            Dictionary mapping atom_idx to degree statistics
        """
        degree_info = {}
        num_atoms = mol.GetNumAtoms()

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            neighbors = list(atom.GetNeighbors())

            total_degree = atom.GetDegree()
            heavy_degree = sum(1 for n in neighbors if n.GetAtomicNum() > 1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total_valence = atom.GetTotalValence()

            neighbor_degrees = [n.GetDegree() for n in neighbors]
            neighbor_heavy = [
                sum(1 for nn in n.GetNeighbors() if nn.GetAtomicNum() > 1)
                for n in neighbors
            ]

            if neighbor_degrees:
                mean_deg = sum(neighbor_degrees) / len(neighbor_degrees)
                mean_heavy = sum(neighbor_heavy) / len(neighbor_heavy)
                variance = sum((d - mean_deg)**2 for d in neighbor_degrees) / len(neighbor_degrees)
            else:
                mean_deg = mean_heavy = variance = 0

            degree_info[idx] = {
                'total_degree': total_degree,
                'heavy_degree': heavy_degree,
                'valence': total_valence,
                'min_neighbor_deg': min(neighbor_degrees) if neighbor_degrees else 0,
                'max_neighbor_deg': max(neighbor_degrees) if neighbor_degrees else 0,
                'mean_neighbor_deg': mean_deg,
                'min_neighbor_heavy': min(neighbor_heavy) if neighbor_heavy else 0,
                'max_neighbor_heavy': max(neighbor_heavy) if neighbor_heavy else 0,
                'mean_neighbor_heavy': mean_heavy,
                'degree_centrality': total_degree / (num_atoms - 1) if num_atoms > 1 else 0,
                'degree_variance': variance
            }

        return degree_info

    def get_stereochemistry_features(self, mol) -> torch.Tensor:
        """
        Extract stereochemistry features for all atoms (8 dimensions per atom).

        Features:
            - Chiral tag (CW, CCW, unspecified)
            - Potential chiral center
            - Has stereo bond
            - Aromatic / SP2 / SP
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 8)

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            chiral_tag = atom.GetChiralTag()

            # Chiral tags
            if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                features[idx, 0] = 1.0
            elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                features[idx, 1] = 1.0
            elif chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                features[idx, 2] = 1.0

            # Potential chiral center
            if (len(atom.GetNeighbors()) == 4 and
                atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3):
                features[idx, 3] = 1.0

            # Has stereo bond
            for bond in atom.GetBonds():
                if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
                    features[idx, 4] = 1.0
                    break

            # Hybridization-based features
            if atom.GetIsAromatic():
                features[idx, 5] = 1.0
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                features[idx, 6] = 1.0
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
                features[idx, 7] = 1.0

        return features

    def get_partial_charges(self, mol) -> torch.Tensor:
        """
        Compute Gasteiger partial charges (2 dimensions per atom).

        Features:
            - Normalized charge [0, 1]
            - Absolute charge
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 2)

        # Use cached charges
        charges = self._get_gasteiger_charges(mol)

        for idx, charge in charges.items():
            features[idx, 0] = (charge + 1.0) / 2.0
            features[idx, 1] = abs(charge)

        return features

    def get_extended_neighborhood(self, mol) -> torch.Tensor:
        """
        Compute extended neighborhood features (16 dimensions per atom).

        Features for 1-hop and 2-hop neighborhoods (8 each):
            - Neighbor count (normalized)
            - Aromatic ratio
            - Heteroatom ratio (N, O, S)
            - H-bond donor ratio
            - H-bond acceptor ratio
            - Mean partial charge
            - Ring atom ratio
            - Halogen ratio (F, Cl, Br, I)
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 16)
        hetero_symbols = {'N', 'O', 'S'}
        halogen_symbols = {'F', 'Cl', 'Br', 'I'}

        # Use cached charges
        charges = self._get_gasteiger_charges(mol)

        def compute_hop_features(neighbors: list) -> list:
            """Compute 8 features for a set of neighbors."""
            if not neighbors:
                return [0.0] * 8

            n = len(neighbors)

            # 1. Count (normalized)
            count_norm = min(n / 6.0, 1.0) if n <= 6 else min(n / 20.0, 1.0)

            # 2. Aromatic ratio
            aromatic_ratio = sum(a.GetIsAromatic() for a in neighbors) / n

            # 3. Heteroatom ratio (N, O, S)
            hetero_ratio = sum(1 for a in neighbors if a.GetSymbol() in hetero_symbols) / n

            # 4. H-bond donor ratio (N-H, O-H)
            h_donor_count = 0
            for a in neighbors:
                symbol = a.GetSymbol()
                if symbol in ('N', 'O') and a.GetTotalNumHs() > 0:
                    h_donor_count += 1
            h_donor_ratio = h_donor_count / n

            # 5. H-bond acceptor ratio (N, O with lone pairs)
            h_acceptor_count = 0
            for a in neighbors:
                symbol = a.GetSymbol()
                if symbol == 'N' and a.GetDegree() < 4:  # N with lone pair
                    h_acceptor_count += 1
                elif symbol == 'O':  # O always has lone pairs
                    h_acceptor_count += 1
            h_acceptor_ratio = h_acceptor_count / n

            # 6. Mean partial charge (normalized to [0, 1])
            neighbor_charges = [charges.get(a.GetIdx(), 0.0) for a in neighbors]
            mean_charge = sum(neighbor_charges) / n
            mean_charge_norm = (mean_charge + 1.0) / 2.0  # [-1, 1] -> [0, 1]

            # 7. Ring atom ratio
            ring_ratio = sum(a.IsInRing() for a in neighbors) / n

            # 8. Halogen ratio (F, Cl, Br, I)
            halogen_ratio = sum(1 for a in neighbors if a.GetSymbol() in halogen_symbols) / n

            return [count_norm, aromatic_ratio, hetero_ratio, h_donor_ratio,
                    h_acceptor_ratio, mean_charge_norm, ring_ratio, halogen_ratio]

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            neighbors_1 = list(atom.GetNeighbors())

            # 2-hop neighbors (excluding self)
            neighbors_2 = set()
            for n1 in neighbors_1:
                for n2 in n1.GetNeighbors():
                    if n2.GetIdx() != idx:
                        neighbors_2.add(n2)
            neighbors_2 = list(neighbors_2)

            # 1-hop features (0-7)
            hop1_feats = compute_hop_features(neighbors_1)
            features[idx, 0:8] = torch.tensor(hop1_feats)

            # 2-hop features (8-15)
            hop2_feats = compute_hop_features(neighbors_2)
            features[idx, 8:16] = torch.tensor(hop2_feats)

        return features

    def get_physical_properties(self, mol) -> torch.Tensor:
        """
        Compute physical property features (6 dimensions per atom).

        Features:
            - Atomic mass
            - Van der Waals radius
            - Covalent radius
            - Ionization energy
            - Polarizability
            - Lone pairs
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 6)
        norm = NORM_CONSTANTS

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            anum = atom.GetAtomicNum()

            # Atomic mass
            features[idx, 0] = min(atom.GetMass() / norm['atomic_mass'], 1.0)

            # Van der Waals radius
            vdw = VDW_RADIUS.get(anum, DEFAULT_VDW_RADIUS)
            features[idx, 1] = (vdw - norm['vdw_radius_min']) / norm['vdw_radius_range']

            # Covalent radius
            cov = COVALENT_RADIUS.get(anum, DEFAULT_COVALENT_RADIUS)
            features[idx, 2] = cov / norm['covalent_radius']

            # Ionization energy
            ie = IONIZATION_ENERGY.get(anum, DEFAULT_IONIZATION_ENERGY)
            features[idx, 3] = (ie - norm['ionization_energy_min']) / norm['ionization_energy_range']

            # Polarizability (log scale)
            pol = POLARIZABILITY.get(anum, DEFAULT_POLARIZABILITY)
            features[idx, 4] = min(np.log1p(pol) / norm['polarizability_log_scale'], 1.0)

            # Lone pairs
            valence_e = VALENCE_ELECTRONS.get(anum, DEFAULT_VALENCE_ELECTRONS)
            bonds = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
            num_h = atom.GetTotalNumHs()
            lone_pairs = max(0, (valence_e - bonds - num_h) / 2.0)
            features[idx, 5] = min(lone_pairs / norm['lone_pairs'], 1.0)

        return features

    def get_crippen_contributions(self, mol) -> torch.Tensor:
        """
        Compute Crippen logP and MR contributions (2 dimensions per atom).
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 2)
        norm = NORM_CONSTANTS

        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        for idx, (logp, mr) in enumerate(contribs):
            features[idx, 0] = (logp + norm['logp_shift']) / norm['logp_range']
            features[idx, 1] = min(mr / norm['mr_max'], 1.0)

        return features

    def get_tpsa_contributions(self, mol) -> torch.Tensor:
        """Compute TPSA contributions (1 dimension per atom)."""
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 1)

        contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        for idx, tpsa in enumerate(contribs):
            features[idx, 0] = min(tpsa / NORM_CONSTANTS['tpsa_max'], 1.0)

        return features

    def get_labute_asa_contributions(self, mol) -> torch.Tensor:
        """Compute Labute ASA contributions (1 dimension per atom)."""
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 1)

        contribs, _ = rdMolDescriptors._CalcLabuteASAContribs(mol)
        for idx, asa in enumerate(contribs):
            features[idx, 0] = min(asa / NORM_CONSTANTS['asa_max'], 1.0)

        return features

    def get_topological_features(self, mol) -> torch.Tensor:
        """
        Compute topological features based on distance matrix (5 dimensions per atom).

        Features:
            - Eccentricity
            - Closeness centrality
            - Betweenness centrality
            - Distance to nearest heteroatom
            - Distance to nearest ring atom
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 5)

        if num_atoms == 1:
            return features

        # Use cached distance matrix
        dm = self._get_distance_matrix(mol)
        norm = NORM_CONSTANTS

        # Identify special atoms
        hetero_indices = [
            i for i, atom in enumerate(mol.GetAtoms())
            if atom.GetAtomicNum() not in [1, 6]
        ]

        ring_info = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)

        # Vectorized eccentricity and closeness
        dm_masked = np.where(np.isinf(dm), 0, dm)
        valid_mask = np.isfinite(dm)

        # Eccentricity: max finite distance per atom
        dm_for_max = np.where(valid_mask, dm, -np.inf)
        max_dists = np.max(dm_for_max, axis=1)
        max_dists[max_dists < 0] = 0
        features[:, 0] = torch.tensor(
            np.clip(max_dists / norm['eccentricity'], 0, 1), dtype=torch.float32
        )

        # Closeness centrality: (N-1) / sum(finite distances)
        dist_sums = dm_masked.sum(axis=1)
        closeness = np.where(dist_sums > 0, (num_atoms - 1) / dist_sums, 0)
        features[:, 1] = torch.tensor(np.clip(closeness, 0, 1), dtype=torch.float32)

        # Distance to nearest heteroatom (vectorized)
        if hetero_indices:
            hetero_arr = np.array(hetero_indices)
            min_hetero_dist = dm[:, hetero_arr].min(axis=1)
            hetero_set = set(hetero_indices)
            for idx in range(num_atoms):
                if idx in hetero_set:
                    features[idx, 3] = 0.0
                else:
                    features[idx, 3] = min(min_hetero_dist[idx] / norm['dist_to_special'], 1.0)
        else:
            features[:, 3] = 1.0

        # Distance to nearest ring atom (vectorized)
        if ring_atoms:
            ring_arr = np.array(list(ring_atoms))
            min_ring_dist = dm[:, ring_arr].min(axis=1)
            for idx in range(num_atoms):
                if idx in ring_atoms:
                    features[idx, 4] = 0.0
                else:
                    features[idx, 4] = min(min_ring_dist[idx] / norm['dist_to_special'], 1.0)
        else:
            features[:, 4] = 1.0

        # Betweenness centrality (vectorized)
        features[:, 2] = torch.tensor(
            self._calc_betweenness(dm, num_atoms), dtype=torch.float32
        )

        return features

    def _calc_betweenness(self, dm: np.ndarray, num_atoms: int) -> np.ndarray:
        """Calculate betweenness centrality for all atoms (vectorized)."""
        betweenness = np.zeros(num_atoms)

        if num_atoms <= 2:
            return betweenness

        # For each intermediate node v, count how many (s,t) pairs
        # have shortest path through v: dm[s,v] + dm[v,t] == dm[s,t]
        for v in range(num_atoms):
            # dm[s, v] + dm[v, t] for all (s, t) pairs
            path_via_v = dm[:, v:v+1] + dm[v:v+1, :]  # (N, N) broadcast
            on_shortest = np.abs(path_via_v - dm) < 0.01  # (N, N) bool
            # Exclude pairs where v == s or v == t, and only upper triangle
            on_shortest[v, :] = False
            on_shortest[:, v] = False
            # Count upper triangle only (s < t)
            betweenness[v] = np.sum(np.triu(on_shortest, k=1))

        # Normalize
        max_pairs = (num_atoms - 1) * (num_atoms - 2) / 2
        if max_pairs > 0:
            betweenness /= max_pairs

        return np.clip(betweenness, 0, 1)

    def get_extended_neighbor_stats(self, mol) -> torch.Tensor:
        """
        Compute extended neighbor statistics (6 dimensions per atom).

        Features:
            - Sum of neighbor electronegativities
            - Electronegativity difference (max - min)
            - Sum of neighbor masses
            - Sum of neighbor formal charges
            - Aromatic neighbor ratio
            - Ring neighbor ratio
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 6)
        norm = NORM_CONSTANTS

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            neighbors = list(atom.GetNeighbors())

            if not neighbors:
                continue

            en_values = []
            masses = []
            charges = []
            n_aromatic = 0
            n_ring = 0

            for n in neighbors:
                # Electronegativity
                period, group = PERIODIC_TABLE.get(n.GetSymbol(), (5, 18))
                en_values.append(ELECTRONEGATIVITY.get((period, group), DEFAULT_ELECTRONEGATIVITY))

                masses.append(n.GetMass())
                charges.append(n.GetFormalCharge())

                if n.GetIsAromatic():
                    n_aromatic += 1
                if n.IsInRing():
                    n_ring += 1

            n_neighbors = len(neighbors)
            features[idx, 0] = min(sum(en_values) / norm['neighbor_en_sum'], 1.0)

            if len(en_values) > 1:
                features[idx, 1] = (max(en_values) - min(en_values)) / norm['neighbor_en_diff']

            features[idx, 2] = min(sum(masses) / norm['neighbor_mass_sum'], 1.0)
            features[idx, 3] = (sum(charges) + norm['neighbor_charge_shift']) / norm['neighbor_charge_range']
            features[idx, 4] = n_aromatic / n_neighbors
            features[idx, 5] = n_ring / n_neighbors

        return features

    def get_extended_ring_features(self, mol) -> torch.Tensor:
        """
        Compute extended ring features (4 dimensions per atom).

        Features:
            - Number of aromatic bonds
            - Ring fusion degree
            - Is bridgehead
            - Is spiro
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, 4)

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        # Ring membership count
        ring_membership = {i: 0 for i in range(num_atoms)}
        for ring in atom_rings:
            for atom_idx in ring:
                ring_membership[atom_idx] += 1

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()

            # Aromatic bonds
            aromatic_bonds = sum(1 for b in atom.GetBonds() if b.GetIsAromatic())
            features[idx, 0] = min(aromatic_bonds / 3.0, 1.0)

            # Ring fusion degree
            features[idx, 1] = min(ring_membership[idx] / 3.0, 1.0)

            # Bridgehead approximation
            if ring_membership[idx] >= 2 and atom.GetDegree() >= 3:
                features[idx, 2] = 1.0

            # Spiro approximation
            if ring_membership[idx] == 2 and atom.GetDegree() == 4:
                features[idx, 3] = 1.0

        return features

    def get_smarts_features(self, mol) -> torch.Tensor:
        """
        Compute SMARTS pattern matching features (5 dimensions per atom).
        """
        num_atoms = mol.GetNumAtoms()
        features = torch.zeros(num_atoms, len(self._smarts_patterns))

        for i, (name, pattern) in enumerate(self._smarts_patterns.items()):
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                matched_atoms = set(sum(matches, ()))
                for atom_idx in matched_atoms:
                    if atom_idx < num_atoms:
                        features[atom_idx, i] = 1.0

        return features

    # =========================================================================
    # Coordinate Extraction
    # =========================================================================

    def get_3d_coordinates(self, mol, generate_if_missing: bool = True) -> torch.Tensor:
        """
        Extract or generate 3D coordinates.

        If coordinates exist, uses them directly.
        If not and generate_if_missing=True, generates on a copy (doesn't modify input mol).

        Args:
            mol: RDKit mol object
            generate_if_missing: Whether to generate coordinates if not present

        Returns:
            Tensor of shape [num_atoms, 3]
        """
        num_atoms = mol.GetNumAtoms()

        # If coordinates exist, use them
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer(0)
            coords = []
            for i in range(num_atoms):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            return torch.tensor(coords, dtype=torch.float32)

        # No coordinates exist
        if not generate_if_missing:
            return torch.zeros((num_atoms, 3), dtype=torch.float32)

        # Generate coordinates on a copy to avoid modifying original
        try:
            from .descriptors import MoleculeFeaturizer
            mol_3d = MoleculeFeaturizer._ensure_3d_conformer(mol)
            if mol_3d is not None and mol_3d.GetNumConformers() > 0:
                conf = mol_3d.GetConformer(0)
                coords = []
                for i in range(num_atoms):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                return torch.tensor(coords, dtype=torch.float32)
        except (RuntimeError, ValueError, ImportError):
            logger.debug("3D coordinate generation failed, using zero coordinates")

        return torch.zeros((num_atoms, 3), dtype=torch.float32)

    # =========================================================================
    # Main Feature Extraction
    # =========================================================================

    def get_atom_features(self, mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract all atom features.

        Returns:
            Tuple of (node_features, coordinates)
            - node_features: [num_atoms, 98]
            - coordinates: [num_atoms, 3]
        """
        norm = NORM_CONSTANTS

        # Per-atom list features
        basic_features = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            formal_charge = atom.GetFormalCharge()
            degree = atom.GetDegree()
            period, group = PERIODIC_TABLE.get(symbol, (PERIODS[-1], GROUPS[-1]))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total_hs = atom.GetTotalNumHs()

            # Essential properties
            mass = min(atom.GetMass() / norm['atomic_mass'], 1.0)
            vdw = VDW_RADIUS.get(atom.GetAtomicNum(), DEFAULT_VDW_RADIUS)
            vdw_norm = (vdw - norm['vdw_radius_min']) / norm['vdw_radius_range']
            electronegativity = ELECTRONEGATIVITY.get((period, group), DEFAULT_ELECTRONEGATIVITY)
            electronegativity_norm = min(electronegativity / 4.0, 1.0)
            radical_electrons_norm = min(atom.GetNumRadicalElectrons() / 2.0, 1.0)

            formal_charge_one_hot = self.one_hot(
                formal_charge, [-2, -1, 0, 1, 2, 'UNK']
            )

            basic = (
                self.one_hot(symbol, ATOM_TYPES) +
                formal_charge_one_hot +
                self.one_hot(atom.GetHybridization(), HYBRIDIZATIONS) +
                [
                    atom.GetIsAromatic(),
                    atom.IsInRing(),
                    radical_electrons_norm,
                ] +
                self.one_hot(total_hs, TOTAL_HS) +
                self.one_hot(degree, DEGREES) +
                [mass, vdw_norm, electronegativity_norm]
            )

            basic_features.append(basic)

        atom_feat = torch.tensor(basic_features, dtype=torch.float32)

        # Advanced atom features
        stereochemistry_features = self.get_stereochemistry_features(mol)
        partial_charges = self.get_partial_charges(mol)
        physical_properties = self.get_physical_properties(mol)
        topological_features = self.get_topological_features(mol)
        smarts_features = self.get_smarts_features(mol)

        # Per-atom contribution features
        extended_neighborhood = self.get_extended_neighborhood(mol)
        crippen_contributions = self.get_crippen_contributions(mol)
        tpsa_contributions = self.get_tpsa_contributions(mol)
        labute_asa_contributions = self.get_labute_asa_contributions(mol)

        node_features = torch.cat(
            [
                atom_feat,
                stereochemistry_features,
                partial_charges,
                physical_properties,
                topological_features,
                smarts_features,
                extended_neighborhood,
                crippen_contributions,
                tpsa_contributions,
                labute_asa_contributions,
            ],
            dim=1,
        )

        coords = self.get_3d_coordinates(mol)
        self._cache['coords'] = coords

        return node_features, coords

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
        num_edge_features = 27

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

    def _precompute_atom_properties(self, mol) -> Dict[int, Dict]:
        """Precompute atom properties for efficient pair feature calculation."""
        props = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            period, group = PERIODIC_TABLE.get(symbol, (5, 18))
            en = ELECTRONEGATIVITY.get((period, group), DEFAULT_ELECTRONEGATIVITY)

            props[idx] = {
                'en': en,
                'mass': atom.GetMass(),
                'charge': atom.GetFormalCharge(),
                'hybrid': atom.GetHybridization(),
            }
        return props

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
            raise ValueError("Failed to compute RDKit distance-geometry bounds matrix.") from exc

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

    def featurize(self, mol, distance_cutoff: Optional[float] = None,
                  knn_cutoff: Optional[int] = None) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        Extract complete graph representation with separate bond and distance edges.

        Args:
            mol: RDKit mol object
            distance_cutoff: Optional 3D distance cutoff for spatial edges.
            knn_cutoff: Optional k-nearest neighbors cutoff for spatial edges.

        Returns:
            Tuple of (node_dict, edge_dict, adjacency_matrix):
            - node_dict: {'node_feats': [N, ~78], 'coords': [N, 3]}
            - edge_dict: {
                'bond_edges': [2, Eb], 'bond_edge_feats': [Eb, ~27],
                'dist_edges': [2, Ed], 'dist_edge_feats': [Ed, 1]
                'pair_features': [N, N, 10],
                'distance_matrix': [N, N],
                'distance_bounds': [N, N, 2],
              }
            - adjacency_matrix: [N, N, ~27] (Bond-based)
        """
        # Clear cache for new molecule
        self._clear_cache()

        # Suppress RDKit warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node_features, coords = self.get_atom_features(mol)
            bond_adj = self.get_bond_features(mol)

        # 1. Bond edges (RDKit Bond 기반)
        src_b, dst_b = torch.where(bond_adj.sum(dim=-1) > 0)
        bond_edge_features = bond_adj[src_b, dst_b]

        # 2. Distance edges (3D Cutoff 기반)
        dist_edge_index = torch.empty((2, 0), dtype=torch.long)
        dist_edge_features = torch.empty((0, 1), dtype=torch.float32)
        pair_features = self.get_pair_features(mol, coords)
        distance_matrix = self.get_distance_matrix(mol, coords)
        distance_bounds = self.get_distance_bounds(mol, coords)

        has_spatial = (distance_cutoff is not None or knn_cutoff is not None) and coords is not None
        if has_spatial:
            dist_matrix = torch.cdist(coords, coords)
            mask = torch.zeros(coords.size(0), coords.size(0), dtype=torch.bool)

            if distance_cutoff is not None:
                mask = mask | ((dist_matrix <= distance_cutoff) & (~torch.eye(coords.size(0), dtype=torch.bool)))

            if knn_cutoff is not None and coords.size(0) > 1:
                dm_knn = dist_matrix.clone()
                dm_knn.fill_diagonal_(float('inf'))
                k = min(knn_cutoff, dm_knn.size(0) - 1)
                _, topk_idx = torch.topk(dm_knn, k, dim=1, largest=False)
                knn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
                knn_mask.scatter_(1, topk_idx, True)
                mask = mask | knn_mask

            src_d, dst_d = torch.where(mask)
            dist_edge_index = torch.stack([src_d, dst_d], dim=0)
            dist_edge_features = dist_matrix[src_d, dst_d].unsqueeze(-1)

        node_dict = {
            'node_feats': node_features,
            'coords': coords
        }

        edge_dict = {
            'edges': torch.stack([src_b, dst_b], dim=0),  # Legacy compatibility
            'edge_feats': bond_edge_features,              # Legacy compatibility
            'bond_edges': torch.stack([src_b, dst_b], dim=0),
            'bond_edge_feats': bond_edge_features,
            'dist_edges': dist_edge_index,
            'dist_edge_feats': dist_edge_features,
            'pair_features': pair_features,
            'distance_matrix': distance_matrix,
            'distance_bounds': distance_bounds,
        }

        return node_dict, edge_dict, bond_adj
