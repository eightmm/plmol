"""
Graph-based molecular featurization for GNN models.

This module provides atom (node) and bond (edge) feature extraction
for molecular graph representations.
"""

import warnings
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdPartialCharges, AllChem
from typing import Dict, Tuple

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

    Node Features (157 dimensions):
        - Basic atom properties (39)
        - Degree features (41)
        - Ring features (21)
        - SMARTS patterns (5)
        - Stereochemistry (8)
        - Partial charges (2)
        - Extended neighborhood (16): 1-hop & 2-hop stats including H-bond, charge, halogen
        - Physical properties (6)
        - Crippen contributions (2)
        - TPSA/ASA contributions (2)
        - Topological features (5)
        - Extended neighbor stats (6)
        - Extended ring features (4)

    Edge Features (66 dimensions):
        - Bond type one-hot (4)
        - Bond stereo one-hot (6)
        - Bond direction one-hot (5)
        - Basic bond properties (5): is_aromatic, is_conjugated, is_in_ring, is_rotatable, bond_order
        - Atom pair properties (8): EN diff, mass diff/sum, charge diff, hybridization match, etc.
        - Ring features (21)
        - Topological features (6): shortest path features, bridge bond, etc.
        - Degree-based features (11)
    """

    def __init__(self):
        """Initialize the graph featurizer."""
        self._smarts_patterns = {
            k: Chem.MolFromSmarts(v) for k, v in CHEMICAL_SMARTS.items()
        }
        self._rotatable_pattern = Chem.MolFromSmarts(ROTATABLE_BOND_SMARTS)

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rdPartialCharges.ComputeGasteigerCharges(mol)

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            try:
                charge = float(atom.GetProp('_GasteigerCharge'))
                if np.isnan(charge):
                    charge = 0.0
            except:
                charge = 0.0

            charge = max(-1.0, min(1.0, charge))
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

        # Precompute partial charges
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = {}
            for atom in mol.GetAtoms():
                charge = atom.GetDoubleProp('_GasteigerCharge')
                if np.isnan(charge) or np.isinf(charge):
                    charge = 0.0
                charges[atom.GetIdx()] = max(-1.0, min(1.0, charge))
        except:
            charges = {atom.GetIdx(): 0.0 for atom in mol.GetAtoms()}

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

        dm = Chem.GetDistanceMatrix(mol)
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

        for idx in range(num_atoms):
            distances = dm[idx]
            valid_dist = distances[distances < np.inf]

            # Eccentricity
            if len(valid_dist) > 1:
                features[idx, 0] = min(np.max(valid_dist) / norm['eccentricity'], 1.0)

            # Closeness centrality
            dist_sum = np.sum(valid_dist)
            if dist_sum > 0:
                features[idx, 1] = min((num_atoms - 1) / dist_sum, 1.0)

            # Distance to nearest heteroatom
            if hetero_indices:
                if idx in hetero_indices:
                    features[idx, 3] = 0.0
                else:
                    min_dist = min(dm[idx][j] for j in hetero_indices)
                    features[idx, 3] = min(min_dist / norm['dist_to_special'], 1.0)
            else:
                features[idx, 3] = 1.0

            # Distance to nearest ring atom
            if ring_atoms:
                if idx in ring_atoms:
                    features[idx, 4] = 0.0
                else:
                    min_dist = min(dm[idx][j] for j in ring_atoms)
                    features[idx, 4] = min(min_dist / norm['dist_to_special'], 1.0)
            else:
                features[idx, 4] = 1.0

        # Betweenness centrality
        features[:, 2] = torch.tensor(
            self._calc_betweenness(dm, num_atoms), dtype=torch.float32
        )

        return features

    def _calc_betweenness(self, dm: np.ndarray, num_atoms: int) -> np.ndarray:
        """Calculate betweenness centrality for all atoms."""
        betweenness = np.zeros(num_atoms)

        if num_atoms <= 2:
            return betweenness

        for s in range(num_atoms):
            for t in range(s + 1, num_atoms):
                if dm[s, t] == np.inf:
                    continue
                for v in range(num_atoms):
                    if v == s or v == t:
                        continue
                    if abs(dm[s, v] + dm[v, t] - dm[s, t]) < 0.01:
                        betweenness[v] += 1

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

        Args:
            mol: RDKit mol object
            generate_if_missing: Whether to generate coordinates if not present

        Returns:
            Tensor of shape [num_atoms, 3]
        """
        num_atoms = mol.GetNumAtoms()

        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer(0)
            coords = []
            for i in range(num_atoms):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            return torch.tensor(coords, dtype=torch.float32)

        if not generate_if_missing:
            return torch.zeros((num_atoms, 3), dtype=torch.float32)

        try:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if mol.GetNumConformers() > 0:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                conf = mol.GetConformer(0)
                coords = []
                for i in range(num_atoms):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                return torch.tensor(coords, dtype=torch.float32)
        except:
            pass

        return torch.zeros((num_atoms, 3), dtype=torch.float32)

    # =========================================================================
    # Main Feature Extraction
    # =========================================================================

    def get_atom_features(self, mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract all atom features.

        Returns:
            Tuple of (node_features, coordinates)
            - node_features: [num_atoms, 157]
            - coordinates: [num_atoms, 3]
        """
        atom_rings, _ = self.get_ring_info(mol)
        degree_info = self.get_degree_info(mol)

        # Tensor features
        stereo_feat = self.get_stereochemistry_features(mol)
        charge_feat = self.get_partial_charges(mol)
        ext_neighbor = self.get_extended_neighborhood(mol)
        physical_feat = self.get_physical_properties(mol)
        crippen_feat = self.get_crippen_contributions(mol)
        tpsa_feat = self.get_tpsa_contributions(mol)
        asa_feat = self.get_labute_asa_contributions(mol)
        topo_feat = self.get_topological_features(mol)
        ext_stats = self.get_extended_neighbor_stats(mol)
        ext_ring = self.get_extended_ring_features(mol)
        smarts_feat = self.get_smarts_features(mol)

        # Per-atom list features
        basic_features = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            period, group = PERIODIC_TABLE.get(symbol, (5, 18))
            en = ELECTRONEGATIVITY.get((period, group), 0.0)
            deg = degree_info[idx]

            # Basic features (39)
            basic = (
                self.one_hot(symbol, ATOM_TYPES) +
                self.one_hot(period, PERIODS) +
                self.one_hot(group, GROUPS) +
                [
                    atom.GetIsAromatic(),
                    atom.IsInRing(),
                    min(atom.GetNumRadicalElectrons() / 3.0, 1.0),
                    (atom.GetFormalCharge() + 3) / 6.0,
                    (en - 0.8) / 3.2
                ]
            )

            # Degree features (41)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total_valence = atom.GetTotalValence()
                total_hs = atom.GetTotalNumHs()

            degree_feat = (
                self.one_hot(deg['total_degree'], DEGREES) +
                self.one_hot(deg['heavy_degree'], HEAVY_DEGREES) +
                self.one_hot(total_valence, VALENCES) +
                self.one_hot(total_hs, TOTAL_HS) +
                self.one_hot(atom.GetHybridization(), HYBRIDIZATIONS) +
                [
                    deg['min_neighbor_deg'] / 6,
                    deg['max_neighbor_deg'] / 6,
                    deg['mean_neighbor_deg'] / 6,
                    deg['min_neighbor_heavy'] / 6,
                    deg['max_neighbor_heavy'] / 6,
                    deg['mean_neighbor_heavy'] / 6,
                    deg['degree_centrality'],
                    deg['degree_variance'] / 10
                ]
            )

            # Ring features (21)
            ring_feat = self.encode_ring_features(
                atom_rings[idx], atom.GetIsAromatic()
            )

            basic_features.append(basic + degree_feat + ring_feat)

        atom_feat = torch.tensor(basic_features, dtype=torch.float32)

        # Concatenate all features
        node_features = torch.cat([
            atom_feat,      # Basic + degree + ring (101)
            smarts_feat,    # SMARTS patterns (5)
            stereo_feat,    # Stereochemistry (8)
            charge_feat,    # Partial charges (2)
            ext_neighbor,   # Extended neighborhood: 1-hop & 2-hop stats (16)
            physical_feat,  # Physical properties (6)
            crippen_feat,   # Crippen logP/MR (2)
            tpsa_feat,      # TPSA (1)
            asa_feat,       # Labute ASA (1)
            topo_feat,      # Topological (5)
            ext_stats,      # Extended neighbor stats (6)
            ext_ring,       # Extended ring features (4)
        ], dim=-1)

        coords = self.get_3d_coordinates(mol)

        return node_features, coords

    def get_bond_features(self, mol) -> torch.Tensor:
        """
        Extract bond features as adjacency tensor.

        Edge Features (66 dimensions):
            - Bond type one-hot (4): SINGLE, DOUBLE, TRIPLE, AROMATIC
            - Bond stereo one-hot (6): stereo configurations
            - Bond direction one-hot (5): wedge/dash for chirality
            - Basic bond properties (5):
                - is_aromatic, is_conjugated, is_in_ring, is_rotatable, bond_order
            - Atom pair properties (8):
                - electronegativity_diff, mass_diff, mass_sum, charge_diff
                - same_hybridization, both_aromatic, both_in_ring, hetero_bond
            - Ring features (21): same as atom ring features
            - Topological features (6):
                - bond_betweenness, is_bridge, ring_fusion_bond
                - shortest_path_to_hetero, shortest_path_to_ring, graph_distance
            - Degree-based features (11):
                - degree diff/sum/min/max, centrality diff/sum, valence stats

        Returns:
            Tensor of shape [num_atoms, num_atoms, 66]
        """
        num_atoms = mol.GetNumAtoms()
        num_edge_features = 66

        _, bond_rings = self.get_ring_info(mol)
        degree_info = self.get_degree_info(mol)

        # Precompute rotatable bonds
        rotatable_atoms = set()
        if self._rotatable_pattern:
            matches = mol.GetSubstructMatches(self._rotatable_pattern)
            rotatable_atoms = set(sum(matches, ()))

        # Precompute distance matrix for topological features
        dm = Chem.GetDistanceMatrix(mol) if num_atoms > 1 else None

        # Precompute atom properties for pair features
        atom_props = self._precompute_atom_properties(mol)

        # Precompute ring info
        ring_info = mol.GetRingInfo()

        adj = torch.zeros(num_atoms, num_atoms, num_edge_features)
        bond_indices = torch.triu(
            torch.tensor(Chem.GetAdjacencyMatrix(mol))
        ).nonzero()

        for src, dst in bond_indices:
            src_idx, dst_idx = src.item(), dst.item()
            bond = mol.GetBondBetweenAtoms(src_idx, dst_idx)
            src_atom = mol.GetAtomWithIdx(src_idx)
            dst_atom = mol.GetAtomWithIdx(dst_idx)
            src_deg = degree_info[src_idx]
            dst_deg = degree_info[dst_idx]

            features = []

            # ===== Bond Type One-hot (4) =====
            features.extend(self.one_hot(bond.GetBondType(), BOND_TYPES))

            # ===== Bond Stereo One-hot (6) =====
            features.extend(self.one_hot(bond.GetStereo(), BOND_STEREOS))

            # ===== Bond Direction One-hot (5) =====
            features.extend(self.one_hot(bond.GetBondDir(), BOND_DIRS))

            # ===== Basic Bond Properties (5) =====
            features.extend([
                bond.GetIsAromatic(),
                bond.GetIsConjugated(),
                bond.IsInRing(),
                (src_idx, dst_idx) in rotatable_atoms or (dst_idx, src_idx) in rotatable_atoms,
                bond.GetBondTypeAsDouble() / 3.0,  # normalized bond order
            ])

            # ===== Atom Pair Properties (8) =====
            src_props = atom_props[src_idx]
            dst_props = atom_props[dst_idx]
            norm = NORM_CONSTANTS

            # Electronegativity difference (bond polarity)
            en_diff = abs(src_props['en'] - dst_props['en'])
            features.append(en_diff / norm['en_diff_max'])

            # Mass difference and sum
            mass_diff = abs(src_props['mass'] - dst_props['mass'])
            mass_sum = src_props['mass'] + dst_props['mass']
            features.append(mass_diff / norm['mass_diff_max'])
            features.append(min(mass_sum / norm['mass_sum_max'], 1.0))

            # Formal charge difference
            charge_diff = abs(src_props['charge'] - dst_props['charge'])
            features.append(charge_diff / norm['charge_diff_max'])

            # Hybridization match
            features.append(float(src_props['hybrid'] == dst_props['hybrid']))

            # Both aromatic
            features.append(float(src_atom.GetIsAromatic() and dst_atom.GetIsAromatic()))

            # Both in ring
            features.append(float(src_atom.IsInRing() and dst_atom.IsInRing()))

            # Hetero bond (involves non-C, non-H)
            is_hetero = (src_atom.GetAtomicNum() not in [1, 6] or
                        dst_atom.GetAtomicNum() not in [1, 6])
            features.append(float(is_hetero))

            # ===== Ring Features (21) =====
            ring_feat = self.encode_ring_features(
                bond_rings[bond.GetIdx()], bond.GetIsAromatic()
            )
            features.extend(ring_feat)

            # ===== Topological Features (6) =====
            topo_feat = self._get_bond_topological_features(
                mol, bond, src_idx, dst_idx, dm, ring_info
            )
            features.extend(topo_feat)

            # ===== Degree-based Features (11) =====
            degree_feat = [
                abs(src_deg['total_degree'] - dst_deg['total_degree']) / 6,
                abs(src_deg['heavy_degree'] - dst_deg['heavy_degree']) / 6,
                abs(src_deg['valence'] - dst_deg['valence']) / 8,
                (src_deg['total_degree'] + dst_deg['total_degree']) / 12,
                (src_deg['heavy_degree'] + dst_deg['heavy_degree']) / 12,
                (src_deg['valence'] + dst_deg['valence']) / 16,
                abs(src_deg['degree_centrality'] - dst_deg['degree_centrality']),
                (src_deg['degree_centrality'] + dst_deg['degree_centrality']) / 2,
                min(src_deg['total_degree'], dst_deg['total_degree']) / 6,
                max(src_deg['total_degree'], dst_deg['total_degree']) / 6,
                abs(src_deg['degree_variance'] - dst_deg['degree_variance']) / 10,
            ]
            features.extend(degree_feat)

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
        dm: np.ndarray, ring_info
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

        # Bond betweenness (simplified)
        betweenness = 0.0
        if dm is not None and num_atoms > 2:
            count = 0
            for s in range(num_atoms):
                for t in range(s + 1, num_atoms):
                    if dm[s, t] == np.inf:
                        continue
                    # Check if this bond is on shortest path s->t
                    d_via_bond = min(dm[s, src_idx] + 1 + dm[dst_idx, t],
                                    dm[s, dst_idx] + 1 + dm[src_idx, t])
                    if abs(d_via_bond - dm[s, t]) < 0.01:
                        count += 1
            max_pairs = (num_atoms - 1) * (num_atoms - 2) / 2
            betweenness = count / max_pairs if max_pairs > 0 else 0
        features.append(min(betweenness, 1.0))

        # Is bridge bond (removal disconnects graph) - approximation
        # A bond is likely a bridge if it's not in any ring
        is_bridge = not bond.IsInRing()
        features.append(float(is_bridge))

        # Ring fusion bond (shared by multiple rings)
        n_rings = ring_info.NumBondRings(bond.GetIdx())
        features.append(min(n_rings / 3.0, 1.0))

        # Shortest path to heteroatom
        hetero_indices = [i for i, atom in enumerate(mol.GetAtoms())
                        if atom.GetAtomicNum() not in [1, 6]]
        if hetero_indices and dm is not None:
            if src_idx in hetero_indices or dst_idx in hetero_indices:
                dist_to_hetero = 0
            else:
                dist_src = min(dm[src_idx][j] for j in hetero_indices)
                dist_dst = min(dm[dst_idx][j] for j in hetero_indices)
                dist_to_hetero = min(dist_src, dist_dst)
            features.append(min(dist_to_hetero / norm['path_length_max'], 1.0))
        else:
            features.append(1.0)

        # Shortest path to ring atom
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)
        if ring_atoms and dm is not None:
            if src_idx in ring_atoms or dst_idx in ring_atoms:
                dist_to_ring = 0
            else:
                dist_src = min(dm[src_idx][j] for j in ring_atoms)
                dist_dst = min(dm[dst_idx][j] for j in ring_atoms)
                dist_to_ring = min(dist_src, dist_dst)
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

    def featurize(self, mol) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        Extract complete graph representation.

        Args:
            mol: RDKit mol object (with hydrogens already processed)

        Returns:
            Tuple of (node_dict, edge_dict, adjacency_matrix):
            - node_dict: {'node_feats': [N, 157], 'coords': [N, 3]}
            - edge_dict: {'edges': [2, E], 'edge_feats': [E, 66]}
            - adjacency_matrix: [N, N, 66]
        """
        node_features, coords = self.get_atom_features(mol)
        bond_features = self.get_bond_features(mol)

        # Extract edge list from adjacency
        src, dst = torch.where(bond_features.sum(dim=-1) > 0)
        edge_features = bond_features[src, dst]

        node_dict = {
            'node_feats': node_features,
            'coords': coords
        }

        edge_dict = {
            'edges': torch.stack([src, dst], dim=0),
            'edge_feats': edge_features
        }

        return node_dict, edge_dict, bond_features
