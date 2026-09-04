"""
Atom (node) feature extraction for molecular graphs.

Provides AtomFeatureMixin with methods for computing per-atom features
including ring analysis, stereochemistry, partial charges, physical properties,
topological features, SMARTS matching, and 3D coordinates.
"""

import logging
import warnings
import numpy as np

from ..arrays import FLOAT
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typing import Dict, Tuple

from ..constants import (
    ATOM_TYPES, PERIODS, GROUPS, DEGREES, TOTAL_HS,
    HYBRIDIZATION_TYPES as HYBRIDIZATIONS,
    PERIODIC_TABLE, ELECTRONEGATIVITY,
    VDW_RADIUS, COVALENT_RADIUS, IONIZATION_ENERGY, POLARIZABILITY, VALENCE_ELECTRONS,
    DEFAULT_VDW_RADIUS, DEFAULT_COVALENT_RADIUS, DEFAULT_IONIZATION_ENERGY,
    DEFAULT_POLARIZABILITY, DEFAULT_VALENCE_ELECTRONS, DEFAULT_ELECTRONEGATIVITY,
    NORM_CONSTANTS,
)
from ..rdkit_utils import ensure_3d_conformer, has_3d, substructure_matches

logger = logging.getLogger(__name__)


def _conformer_coords(conf, num_atoms: int) -> np.ndarray:
    """First ``num_atoms`` conformer positions as a float32 (num_atoms, 3) tensor."""
    positions = np.asarray(conf.GetPositions(), dtype=np.float32).reshape(-1, 3)
    return positions[:num_atoms].copy()


class AtomFeatureMixin:
    """Mixin providing atom-level feature extraction methods.

    Requires the host class to provide:
        - self._smarts_patterns: dict of compiled SMARTS patterns
        - self._cache: dict for per-molecule caching
        - self.one_hot(value, allowable_set) -> list
        - self.normalize(value, min_val, max_val, clip) -> float
        - self._get_gasteiger_charges(mol) -> dict
        - self._get_distance_matrix(mol) -> np.ndarray
    """

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

    def get_stereochemistry_features(self, mol) -> np.ndarray:
        """
        Extract stereochemistry features for all atoms (6 dimensions per atom).

        Until 0.4.0 there were 8. The dropped two repeated channels the vector
        already carried, bit for bit on every atom of 29 molecules covering B,
        Se, P, S, halogens, charges and stereocentres: is_aromatic was the
        aromatic flag at [34], and is_SP was the SP column of the hybridization
        one-hot at [28]. is_SP2 stays, because the branch below only reaches it
        for a non-aromatic atom and that is not what [29] says.

        Features:
            - Chiral tag (CW, CCW, unspecified)
            - Potential chiral center
            - Has stereo bond
            - Aromatic / SP2 / SP
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 6), dtype=FLOAT)

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

            # SP2 only when the atom is not aromatic, which is what makes this
            # column say something the hybridization one-hot does not.
            if atom.GetIsAromatic():
                pass
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                features[idx, 5] = 1.0

        return features

    def get_partial_charges(self, mol) -> np.ndarray:
        """
        Compute Gasteiger partial charges (2 dimensions per atom).

        Features:
            - Normalized charge [0, 1]
            - Absolute charge
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 2), dtype=FLOAT)

        # Use cached charges
        charges = self._get_gasteiger_charges(mol)

        for idx, charge in charges.items():
            features[idx, 0] = (charge + 1.0) / 2.0
            features[idx, 1] = abs(charge)

        return features

    def get_extended_neighborhood(self, mol) -> np.ndarray:
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
        features = np.zeros((num_atoms, 16), dtype=FLOAT)
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
            features[idx, 0:8] = np.asarray(hop1_feats, dtype=FLOAT)

            # 2-hop features (8-15)
            hop2_feats = compute_hop_features(neighbors_2)
            features[idx, 8:16] = np.asarray(hop2_feats, dtype=FLOAT)

        return features

    def get_physical_properties(self, mol) -> np.ndarray:
        """
        Compute physical property features (4 dimensions per atom).

        Until 0.4.0 there were 6, starting with atomic mass and van der Waals
        radius. Both were bit-identical to the atom-property channels at [49]
        and [50].

        Features:
            - Covalent radius
            - Ionization energy
            - Polarizability
            - Lone pairs
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 4), dtype=FLOAT)
        norm = NORM_CONSTANTS

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            anum = atom.GetAtomicNum()

            # Covalent radius
            cov = COVALENT_RADIUS.get(anum, DEFAULT_COVALENT_RADIUS)
            features[idx, 0] = cov / norm['covalent_radius']

            # Ionization energy
            ie = IONIZATION_ENERGY.get(anum, DEFAULT_IONIZATION_ENERGY)
            features[idx, 1] = (ie - norm['ionization_energy_min']) / norm['ionization_energy_range']

            # Polarizability (log scale)
            pol = POLARIZABILITY.get(anum, DEFAULT_POLARIZABILITY)
            features[idx, 2] = min(np.log1p(pol) / norm['polarizability_log_scale'], 1.0)

            # Lone pairs
            valence_e = VALENCE_ELECTRONS.get(anum, DEFAULT_VALENCE_ELECTRONS)
            bonds = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
            num_h = atom.GetTotalNumHs()
            lone_pairs = max(0, (valence_e - bonds - num_h) / 2.0)
            features[idx, 3] = min(lone_pairs / norm['lone_pairs'], 1.0)

        return features

    def get_crippen_contributions(self, mol) -> np.ndarray:
        """
        Compute Crippen logP and MR contributions (2 dimensions per atom).
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 2), dtype=FLOAT)
        norm = NORM_CONSTANTS

        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        for idx, (logp, mr) in enumerate(contribs):
            features[idx, 0] = (logp + norm['logp_shift']) / norm['logp_range']
            features[idx, 1] = min(mr / norm['mr_max'], 1.0)

        return features

    def get_tpsa_contributions(self, mol) -> np.ndarray:
        """Compute TPSA contributions (1 dimension per atom)."""
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 1), dtype=FLOAT)

        contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        for idx, tpsa in enumerate(contribs):
            features[idx, 0] = min(tpsa / NORM_CONSTANTS['tpsa_max'], 1.0)

        return features

    def get_labute_asa_contributions(self, mol) -> np.ndarray:
        """Compute Labute ASA contributions (1 dimension per atom)."""
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 1), dtype=FLOAT)

        contribs, _ = rdMolDescriptors._CalcLabuteASAContribs(mol)
        for idx, asa in enumerate(contribs):
            features[idx, 0] = min(asa / NORM_CONSTANTS['asa_max'], 1.0)

        return features

    def get_topological_features(self, mol) -> np.ndarray:
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
        features = np.zeros((num_atoms, 5), dtype=FLOAT)

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
        features[:, 0] = np.asarray(
            np.clip(max_dists / norm['eccentricity'], 0, 1), dtype=FLOAT
        )

        # Closeness centrality: (N-1) / sum(finite distances)
        dist_sums = dm_masked.sum(axis=1)
        closeness = np.where(dist_sums > 0, (num_atoms - 1) / dist_sums, 0)
        features[:, 1] = np.asarray(np.clip(closeness, 0, 1), dtype=FLOAT)

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
        features[:, 2] = np.asarray(
            self._calc_betweenness(dm, num_atoms), dtype=FLOAT
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

    def get_extended_neighbor_stats(self, mol) -> np.ndarray:
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
        features = np.zeros((num_atoms, 6), dtype=FLOAT)
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

    def get_extended_ring_features(self, mol) -> np.ndarray:
        """
        Compute extended ring features (4 dimensions per atom).

        Features:
            - Number of aromatic bonds
            - Ring fusion degree
            - Is bridgehead
            - Is spiro
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 4), dtype=FLOAT)

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

    def get_smarts_features(self, mol) -> np.ndarray:
        """
        Compute SMARTS pattern matching features (5 dimensions per atom).
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, len(self._smarts_patterns)), dtype=FLOAT)

        for i, (name, pattern) in enumerate(self._smarts_patterns.items()):
            if pattern is None:
                continue
            matches = substructure_matches(mol, pattern)
            if matches:
                matched_atoms = set(sum(matches, ()))
                for atom_idx in matched_atoms:
                    if atom_idx < num_atoms:
                        features[atom_idx, i] = 1.0

        return features

    # =========================================================================
    # Coordinate Extraction
    # =========================================================================

    def get_3d_coordinates(self, mol, generate_if_missing: bool = True) -> np.ndarray:
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
        if has_3d(mol):
            conf = mol.GetConformer(0)
            return _conformer_coords(conf, num_atoms)

        # No coordinates exist
        if not generate_if_missing:
            return np.zeros((num_atoms, 3), dtype=FLOAT)

        # Generate coordinates on a copy to avoid modifying original
        try:
            mol_3d = ensure_3d_conformer(mol)
            if mol_3d is not None and has_3d(mol_3d):
                conf = mol_3d.GetConformer(0)
                # ensure_3d_conformer adds hydrogens, so keep the leading
                # heavy-atom rows that line up with the input molecule.
                return _conformer_coords(conf, num_atoms)
        except (RuntimeError, ValueError, ImportError):
            logger.debug("3D coordinate generation failed, using zero coordinates")

        return np.zeros((num_atoms, 3), dtype=FLOAT)

    # =========================================================================
    # Main Atom Feature Assembly
    # =========================================================================

    def get_atom_features(
        self,
        mol,
        generate_conformer: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract all atom features.

        Returns:
            Tuple of (node_features, coordinates)
            - node_features: [num_atoms, 94]
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

        atom_feat = np.asarray(basic_features, dtype=FLOAT)

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

        node_features = np.concatenate(
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
            axis=1,
        )

        coords = self.get_3d_coordinates(mol, generate_if_missing=generate_conformer)
        self._cache['coords'] = coords

        return node_features, coords

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
