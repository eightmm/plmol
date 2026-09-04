"""
Residue Featurizer Module

This module provides functionality to extract structural features from protein PDB files
for machine learning applications. It includes geometric features, SASA calculations,
and graph-based interaction features.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any

from collections import defaultdict

logger = logging.getLogger(__name__)

import numpy as np

from .geometry import (
    calculate_dihedral,
    calculate_local_frames,
    calculate_backbone_curvature,
    calculate_backbone_torsion,
    calculate_self_distances_vectors,
)
# Import unified PDB parsing utilities from canonical location
from .utils import (
    PDBParser,
    calculate_sidechain_centroid,
    normalize_residue_name,
    residue_label_parts,
)

from ..arrays import FLOAT, INT, one_hot, pairwise_distances, sanitized
from ..utils import (
    DEFAULT_SASA_POINTS,
    PEPTIDE_BOND_MAX,
    dense_to_edges,
    knn_mask,
    residue_chain_breaks,
    sasa_structure_result,
)

# Import amino acid constants from centralized module
from ..constants import (
    AMINO_ACID_3TO1,
    AMINO_ACID_1TO3,
    AMINO_ACID_1_TO_INT,
    AMINO_ACID_3_TO_INT,
    MAX_ATOMS_PER_RESIDUE,
    NUM_RESIDUE_TYPES,
    RESIDUE_PROPERTIES,
    NUM_RESIDUE_PROPERTIES,
    RESIDUE_TYPES,
    RESIDUE_MAX_SASA,
    residue_atom_index,
)


# =============================================================================
# Chi Angle Constants (cached at module level)
# =============================================================================
# Residue indices that have each chi angle
CHI_ANGLE_RESIDUE_INDICES = {
    'chi1': np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    'chi2': np.array([2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19]),
    'chi3': np.array([3, 8, 10, 13, 14]),
    'chi4': np.array([8, 14]),
    'chi5': np.array([14]),
}

# ILE residue index for special handling
ILE_RESIDUE_INDEX = np.array([7])


class ResidueFeaturizer:
    """
    A class for extracting structural features from protein PDB files.

    This class provides methods to compute various features including:
    - Geometric features (distances, angles, dihedrals)
    - Solvent accessible surface area (SASA)
    - Residue-level and interaction features
    - Graph representations of protein structure

    Note:
        All preprocessing (metal ion exclusion, water removal, etc.) is handled
        by PDBParser. This class focuses only on feature extraction.
    """

    def __init__(self, pdb_file: str, sasa_points: int = DEFAULT_SASA_POINTS):
        """
        Initialize the featurizer with a PDB file.

        Args:
            pdb_file: Path to the PDB file
            sasa_points: Shrake-Rupley sample points per atom. Raising it
                narrows the orientation dependence documented on
                :func:`plmol.sasa.shrake_rupley`, at proportional cost. It is
                fixed per featurizer rather than per call, so a cached area is
                never served to a request that asked for a different count.

        Note:
            Uses PDBParser internally for consistent preprocessing.
        """
        self.pdb_file = pdb_file
        self.sasa_points = sasa_points

        # Use PDBParser for consistent preprocessing (metal/water/hydrogen exclusion)
        parser = PDBParser(pdb_file)
        self._init_from_parser(parser)

    def _init_from_parser(self, pdb_parser: 'PDBParser'):
        """
        Initialize internal data structures from PDBParser.

        Builds two lookup structures directly (no pandas):
        - _coord_cache: (chain, res_num, res_type, icode) → np.ndarray of all atom coords
        - _atom_coords:  (chain, res_num, res_type, icode) → {atom_name: np.ndarray}
        - _residues: sorted list of unique (chain, res_num, res_type, icode) tuples,
          ordered by (chain, res_num, icode) so 100, 100A, 100B follow each other

        Args:
            pdb_parser: Pre-initialized PDBParser instance
        """
        # Collect atoms grouped by residue key
        residue_atoms: dict[tuple, list] = defaultdict(list)
        atom_coords: dict[tuple, dict] = defaultdict(dict)

        for atom in pdb_parser.protein_atoms:
            norm_res = normalize_residue_name(atom.res_name, atom.atom_name)
            res_type = AMINO_ACID_3_TO_INT.get(norm_res, 20)

            if res_type == 20:
                if atom.atom_name not in ['N', 'CA', 'C', 'O', 'CB']:
                    continue

            # res_type stays at index 2: callers read residue[2] and
            # np.array(residues)[:, 2] for it. The insertion code goes last and
            # the sort below uses it explicitly, so 100, 100A and 100B are three
            # residues in sequence order rather than one pile of atoms.
            res_key = (atom.chain_id, atom.res_num, res_type, atom.insertion_code)
            coord = np.array(atom.coords, dtype=np.float32)
            # Everything downstream reads these rows positionally -- the CA is
            # row 1, the side chain starts at row 4 -- so they are stored in the
            # residue's own atom order rather than the file's. A PDB is only
            # conventionally written N, CA, C, O, CB; a file that is not, and
            # that standardize=False leaves alone, used to put some other atom
            # where the CA belongs. File position breaks ties, so a canonical
            # file is unaffected.
            order = (residue_atom_index(norm_res, atom.atom_name),
                     len(residue_atoms[res_key]))
            residue_atoms[res_key].append((order, coord))
            atom_coords[res_key][atom.atom_name] = coord

        # Build coordinate caches
        self._coord_cache = {
            key: np.vstack([coord for _, coord in sorted(atoms, key=lambda a: a[0])])
            for key, atoms in residue_atoms.items()
        }
        self._atom_coords = dict(atom_coords)
        self._residues = sorted(residue_atoms.keys(), key=lambda k: (k[0], k[1], k[3]))

    @classmethod
    def from_parser(cls, pdb_parser: 'PDBParser', pdb_file: str = None,
                    sasa_points: int = DEFAULT_SASA_POINTS) -> 'ResidueFeaturizer':
        """
        Create ResidueFeaturizer from pre-parsed PDBParser data.

        Avoids re-parsing the PDB file when PDBParser is already available.

        Args:
            pdb_parser: Pre-initialized PDBParser instance
            pdb_file: Optional path to PDB file (for SASA calculation)

        Returns:
            ResidueFeaturizer instance with cached data

        Note:
            PDBParser handles all preprocessing (metal/water/hydrogen exclusion).
        """
        instance = cls.__new__(cls)
        instance.pdb_file = pdb_file or pdb_parser.pdb_path
        instance.sasa_points = sasa_points
        instance._init_from_parser(pdb_parser)
        return instance

    def get_residues(self) -> List[Tuple]:
        """
        Get list of all protein residues.

        Returns:
            List of (chain, residue_number, residue_type) tuples
        """
        return list(self._residues)

    def get_sequence_by_chain(self) -> Dict[str, str]:
        """
        Get amino acid sequences in one-letter code separated by chain.

        Returns:
            Dictionary mapping chain IDs to one-letter amino acid sequences
        """
        residues = self.get_residues()
        sequences_by_chain = {}

        # Reverse mapping from int to 3-letter code
        int_to_3letter = {v: k for k, v in AMINO_ACID_3_TO_INT.items()}

        for chain, res_num, res_type, _icode in residues:
            if chain not in sequences_by_chain:
                sequences_by_chain[chain] = []

            three_letter = int_to_3letter.get(res_type, 'UNK')
            one_letter = AMINO_ACID_3TO1.get(three_letter, 'X')
            sequences_by_chain[chain].append(one_letter)

        # Convert lists to strings
        return {chain: ''.join(seq) for chain, seq in sequences_by_chain.items()}

    def get_residue_coordinates(self, residue_index: Tuple) -> Dict[str, np.ndarray]:
        """
        Get coordinates for a specific residue, keyed by atom name.

        Args:
            residue_index: Tuple of (chain, residue_number, residue_type)

        Returns:
            Dict mapping atom_name → coordinate np.ndarray (3,)
        """
        return self._atom_coords.get(residue_index, {})

    def get_residue_coordinates_numpy(self, residue_index: Tuple) -> np.ndarray:
        """
        Get coordinates for a specific residue as numpy array (faster).

        Uses pre-built cache for O(1) dict lookup.

        Args:
            residue_index: Tuple of (chain, residue_number, residue_type)

        Returns:
            Coordinates as numpy array [num_atoms, 3]
        """
        return self._coord_cache.get(residue_index, np.zeros((1, 3), dtype=np.float32))

    def get_terminal_flags(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify N-terminal and C-terminal residues.

        Returns:
            Tuple of (n_terminal_flags, c_terminal_flags) as boolean tensors
        """
        residues = self.get_residues()

        residue_array = np.array(residues, dtype=object)
        chains = residue_array[:, 0]
        res_nums = residue_array[:, 1].astype(int)

        unique_chains = np.unique(chains)
        n_terminal = np.zeros(len(residues), dtype=bool)
        c_terminal = np.zeros(len(residues), dtype=bool)

        chain_masks = chains[:, None] == unique_chains[None, :]

        for i, chain in enumerate(unique_chains):
            mask = chain_masks[:, i]
            chain_indices = np.where(mask)[0]
            chain_res_nums = res_nums[mask]

            min_idx = chain_indices[np.argmin(chain_res_nums)]
            max_idx = chain_indices[np.argmax(chain_res_nums)]

            n_terminal[min_idx] = True
            c_terminal[max_idx] = True

        return n_terminal, c_terminal

    def get_relative_position(self, cutoff: int = 32, onehot: bool = True) -> np.ndarray:
        """
        Calculate relative position encoding for residue pairs.

        Args:
            cutoff: Maximum relative position to consider
            onehot: Whether to return one-hot encoded positions

        Returns:
            Relative position tensor
        """
        residues = self.get_residues()
        num_residues = len(residues)

        chain_indices = {}
        for idx, residue in enumerate(residues):
            chain = residue[0]
            if chain not in chain_indices:
                chain_indices[chain] = []
            chain_indices[chain].append(idx)

        relative_positions = np.full((num_residues, num_residues), -1, dtype=INT)

        for chain, indices in chain_indices.items():
            if len(indices) <= 1:
                continue

            indices_tensor = np.array(indices, dtype=INT)
            num_chain_residues = len(indices)

            arrange = np.arange(num_chain_residues, dtype=INT)
            chain_relative_positions = np.abs(arrange[:, None] - arrange[None, :])
            chain_relative_positions = np.minimum(chain_relative_positions, cutoff + 1)
            chain_relative_positions = np.where(chain_relative_positions > cutoff, 33, chain_relative_positions)

            relative_positions[indices_tensor[:, None], indices_tensor[None, :]] = chain_relative_positions

        if onehot:
            relative_positions_mapped = np.where(relative_positions == -1, 34, relative_positions)
            relative_positions_onehot = one_hot(relative_positions_mapped, 35, dtype=FLOAT)
            return relative_positions_onehot

        return relative_positions

    def calculate_sasa(self) -> np.ndarray:
        """
        Calculate Solvent Accessible Surface Area (SASA) for each residue.

        Returns:
            SASA features array of shape [num_residues, 11]:
                - polar, apolar, mainChain, sideChain over RESIDUE_MAX_SASA:
                  what fraction of the residue's whole surface each class
                  accounts for. The total is relativeTotal below, which is the
                  same quantity under its proper name.
                - relativeTotal, relativePolar, relativeApolar,
                  relativeMainChain, relativeSideChain, each over
                  RESIDUE_MAX_CLASS_SASA: how exposed that class is against
                  how exposed it could be
                - burial_index (1.0 - relativeTotal)
                - polar_apolar_ratio (polar / (polar + apolar))

            The two groups answer different questions. They were the same
            numbers in 0.3.x and briefly in 0.4.0, because the per-class
            references did not exist and everything was divided by the
            residue's total.
        """
        num_residues = len(self.get_residues())
        sasa_dim = 11  # Number of SASA features per residue


        # Build reverse mapping: res_type_int -> 3-letter code for RESIDUE_MAX_SASA lookup
        int_to_3letter = {v: k for k, v in AMINO_ACID_3_TO_INT.items()}

        try:
            _, result = sasa_structure_result(self.pdb_file, n_points=self.sasa_points)
            residue_areas = result.residueAreas()

            # residueAreas() comes back in the order the file lists residues;
            # self._residues is sorted by (chain, number, insertion code).
            # Zipping the two positionally, as this did until 0.4.x, put a
            # residue's SASA on whichever row happened to sit at the same
            # index and normalised it by another residue's reference area.
            # Any file whose chains are not in alphabetical order, or whose
            # residues are not ascending, was affected.
            row_of = {
                (key[0], key[1], key[3]): index
                for index, key in enumerate(self._residues)
            }
            sasa_tensor = np.zeros((num_residues, sasa_dim), dtype=FLOAT)
            matched = 0
            for chain, residues in residue_areas.items():
                for label, values in residues.items():
                    res_num, insertion_code = residue_label_parts(label)
                    index = row_of.get((chain, res_num, insertion_code))
                    if index is None:
                        index = row_of.get((chain.strip(), res_num, insertion_code))
                    if index is None:
                        continue
                    matched += 1

                    res_name_3 = int_to_3letter.get(self._residues[index][2], 'UNK')
                    max_sasa = RESIDUE_MAX_SASA.get(res_name_3, 200.0)

                    # relativeTotal is a fraction of the residue's reference
                    # area, not a percentage.
                    burial_index = 1.0 - values.relativeTotal
                    polar_apolar_ratio = values.polar / (values.polar + values.apolar + 1e-8)

                    sasa_tensor[index] = [
                        values.polar / max_sasa,
                        values.apolar / max_sasa,
                        values.mainChain / max_sasa,
                        values.sideChain / max_sasa,
                        values.relativeTotal,
                        values.relativePolar,
                        values.relativeApolar,
                        values.relativeMainChain,
                        values.relativeSideChain,
                        burial_index,
                        polar_apolar_ratio,
                    ]

            if matched != num_residues:
                logger.warning(
                    f"SASA covered {matched} of {num_residues} residues; the rest "
                    "keep zeros. This usually means two residues were grouped as one."
                )

            return np.nan_to_num(sasa_tensor)

        except Exception as e:
            logger.warning(f"FreeSASA calculation failed: {e}. Returning zeros for SASA features.")
            return np.zeros((num_residues, sasa_dim), dtype=FLOAT)

    def _residue_breaks(self) -> np.ndarray:
        """``(L - 1,)`` bool: residue pairs that are not joined by a peptide bond.

        A chain boundary is always a break, and so is a gap in the numbering
        that a disordered loop left behind. In the example protein, chain A
        ends 46 A from where chain B starts, and until 0.4.x the two were read
        as consecutive.
        """
        residues = self.get_residues()
        missing = np.full(3, np.nan)
        carbon = np.array(
            [self._atom_coords.get(key, {}).get('C', missing) for key in residues],
            dtype=np.float64,
        ).reshape(-1, 3)
        nitrogen = np.array(
            [self._atom_coords.get(key, {}).get('N', missing) for key in residues],
            dtype=np.float64,
        ).reshape(-1, 3)
        return residue_chain_breaks(
            [key[0] for key in residues], carbon, nitrogen, PEPTIDE_BOND_MAX
        )

    def get_dihedral_angles(self, coords: np.ndarray, res_types: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate backbone and sidechain dihedral angles.

        Args:
            coords: Residue coordinates tensor (num_residues, 15, 3)
            res_types: Residue type indices tensor (num_residues,)

        Returns:
            Tuple of (dihedral_angles, has_chi_angles)
        """
        # Use cached chi angle constants
        is_ILE = np.isin(res_types, ILE_RESIDUE_INDEX).astype(FLOAT)[:, None, None]
        is_not_ILE = 1 - is_ILE

        has_chi = np.stack([
            np.isin(res_types, CHI_ANGLE_RESIDUE_INDICES[f'chi{i}']).astype(np.int32)
            for i in range(1, 6)
        ], axis=1)

        # Backbone dihedrals. The rows are a sorted residue list, not a chain:
        # psi and omega of a residue and phi of the next one read across the
        # join, and the join is only a peptide bond when the structure says so.
        N_CA_C = coords[:, :3, :]
        backbone_dihedrals = calculate_dihedral(N_CA_C, breaks=self._residue_breaks())

        # Sidechain dihedrals
        N_A_B_G_D_E_Z_ILE = np.concatenate([coords[:, :2, :], coords[:, 4:6, :], coords[:, 7:11, :]], axis=1) * is_ILE
        N_A_B_G_D_E_Z_no_ILE = np.concatenate([coords[:, :2, :], coords[:, 4:10, :]], axis=1) * is_not_ILE
        N_A_B_G_D_E_Z = N_A_B_G_D_E_Z_ILE + N_A_B_G_D_E_Z_no_ILE

        side_chain_dihedrals = calculate_dihedral(N_A_B_G_D_E_Z)[:, 1:-2] * has_chi.astype(FLOAT)

        dihedrals = np.concatenate([backbone_dihedrals, side_chain_dihedrals], axis=1)

        return dihedrals, has_chi

    def _calculate_forward_reverse(self, coord: np.ndarray, terminal_flags: Tuple[np.ndarray, np.ndarray]) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate forward and reverse residue connection features.

        Args:
            coord: Residue coordinates
            terminal_flags: N and C terminal flags

        Returns:
            Tuple of ((forward_vectors, forward_distances), (reverse_vectors, reverse_distances))
        """
        ca_coords = coord[:, 1, :]  # CA coordinates
        sc_coords = coord[:, -1, :]  # SC coordinates

        n_terminal, c_terminal = terminal_flags

        forward_vector = np.zeros((coord.shape[0], 4, 3), dtype=FLOAT)
        forward_distance = np.zeros((coord.shape[0], 4), dtype=FLOAT)
        reverse_vector = np.zeros((coord.shape[0], 4, 3), dtype=FLOAT)
        reverse_distance = np.zeros((coord.shape[0], 4), dtype=FLOAT)

        if coord.shape[0] > 1:
            ca_diff = ca_coords[1:] - ca_coords[:-1]
            sc_diff = sc_coords[1:] - sc_coords[:-1]
            ca_sc_diff = sc_coords[1:] - ca_coords[:-1]
            sc_ca_diff = ca_coords[1:] - sc_coords[:-1]

            forward_vector[:-1] = np.stack([ca_diff, sc_diff, ca_sc_diff, sc_ca_diff], axis=1)
            forward_distance[:-1] = np.linalg.norm(forward_vector[:-1], axis=-1)

            c_mask = ~c_terminal[:-1]
            forward_vector[:-1] *= c_mask[:, None, None]
            forward_distance[:-1] *= c_mask[:, None]

            reverse_vector[1:] = np.stack([-ca_diff, -sc_diff, ca_coords[:-1] - sc_coords[1:],
                                           sc_coords[:-1] - ca_coords[1:]], axis=1)
            reverse_distance[1:] = np.linalg.norm(reverse_vector[1:], axis=-1)

            n_mask = (~n_terminal[1:])
            reverse_vector[1:] *= n_mask[:, None, None]
            reverse_distance[1:] *= n_mask[:, None]

        forward_vector = sanitized(forward_vector, ca_coords, sc_coords)
        reverse_vector = sanitized(reverse_vector, ca_coords, sc_coords)
        forward_distance = sanitized(forward_distance, ca_coords, sc_coords)
        reverse_distance = sanitized(reverse_distance, ca_coords, sc_coords)

        return (forward_vector, forward_distance), (reverse_vector, reverse_distance)

    def _calculate_interaction_features(self, coords: np.ndarray, distance_cutoff: float = 8,
                                        knn_cutoff: Optional[int] = None) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate inter-residue interaction features.

        Args:
            coords: Residue coordinates
            distance_cutoff: Distance cutoff for interactions
            knn_cutoff: Optional k-nearest neighbors cutoff. If given, union
                distance-based and kNN-based adjacency for connectivity.

        Returns:
            Tuple of (distances, adjacency_matrix, interaction_vectors)
        """
        coord_CA = coords[:, 1, :]
        coord_SC = coords[:, -1, :]
        mask = (1 - np.eye(coords.shape[0], dtype=FLOAT)).astype(np.int32)

        dm_CA_CA = pairwise_distances(coord_CA, coord_CA)
        dm_SC_SC = pairwise_distances(coord_SC, coord_SC)
        dm_CA_SC = pairwise_distances(coord_CA, coord_SC)
        # |SC_i - CA_j| is |CA_j - SC_i|, and a norm is exactly the norm of the
        # negated vector, so the fourth matrix is the third transposed.
        dm_SC_CA = dm_CA_SC.T

        adj_CA_CA = (dm_CA_CA < distance_cutoff) * mask
        adj_SC_SC = (dm_SC_SC < distance_cutoff) * mask
        adj_CA_SC = (dm_CA_SC < distance_cutoff) * mask
        adj_SC_CA = (dm_SC_CA < distance_cutoff) * mask

        adj = adj_CA_CA | adj_SC_SC | adj_CA_SC | adj_SC_CA

        if knn_cutoff is not None and coords.shape[0] > 1:
            min_dm = np.minimum(np.minimum(dm_CA_CA, dm_SC_SC),
                                np.minimum(dm_CA_SC, dm_SC_CA))
            knn_adj = knn_mask(min_dm, knn_cutoff).astype(np.int32)
            knn_adj = (knn_adj * mask).astype(np.int32)
            adj = adj | knn_adj

        dm_all = np.stack((dm_CA_CA, dm_SC_SC, dm_CA_SC, dm_SC_CA), axis=-1)
        dm_select = dm_all * adj[:, :, None].astype(FLOAT)

        # Calculate interaction vectors
        coord_CA_SC = np.concatenate([coords[:, 1:2, :], coords[:, -1:, :]], axis=1)
        coord_SC_CA = np.concatenate([coords[:, -1:, :], coords[:, 1:2, :]], axis=1)

        vector1 = coord_CA_SC[:, None, :] - coord_CA_SC[:, :, :]
        vector3 = coord_CA_SC[:, None, :] - coord_SC_CA[:, :, :]
        vectors = np.concatenate([vector1, -vector1, vector3, -vector3], axis=2)
        vectors = sanitized(vectors, coord_CA, coord_SC)
        vectors = vectors * adj[:, :, None, None].astype(FLOAT)

        return sanitized(dm_select, coord_CA, coord_SC), adj, vectors

    def _extract_residue_features(self, coords: np.ndarray, residue_types: np.ndarray) -> \
            Tuple[Tuple, Tuple]:
        """
        Extract all residue-level features.

        Args:
            coords: Residue coordinates
            residue_types: Residue type indices

        Returns:
            Tuple of (scalar_features, vector_features)
        """
        # One-hot encoding of residue types (with bounds checking)
        residue_types_clamped = np.clip(residue_types, 0, NUM_RESIDUE_TYPES - 1)
        residue_one_hot = one_hot(residue_types_clamped, NUM_RESIDUE_TYPES, dtype=INT)
        terminal_flags = self.get_terminal_flags()

        # Local self features
        self_distance, self_vector = calculate_self_distances_vectors(coords)

        # Local frames
        local_frames = calculate_local_frames(coords)

        # Dihedral angles and curvature
        dihedrals, has_chi_angles = self.get_dihedral_angles(coords, residue_types)
        backbone_curvature = calculate_backbone_curvature(coords, terminal_flags)
        backbone_torsion = calculate_backbone_torsion(coords, terminal_flags)

        degree = np.concatenate([dihedrals, backbone_curvature[:, None], backbone_torsion[:, None]], axis=1)
        degree_feature = np.concatenate([np.cos(degree), np.sin(degree)], axis=1)

        # SASA features
        sasa = self.calculate_sasa()

        # Forward/reverse features
        forward, reverse = self._calculate_forward_reverse(coords, terminal_flags)
        forward_vector, forward_distance = forward
        reverse_vector, reverse_distance = reverse

        rf_vector = np.concatenate([forward_vector, reverse_vector], axis=1)
        rf_distance = np.concatenate([forward_distance, reverse_distance], axis=1)

        # Physicochemical properties (5-dim per residue, vectorized via lookup table)
        default_props = RESIDUE_PROPERTIES.get('Other', [0.0] * NUM_RESIDUE_PROPERTIES)
        property_rows = []
        for res_name in RESIDUE_TYPES:
            property_rows.append(RESIDUE_PROPERTIES.get(res_name, default_props))
        property_rows.append(default_props)  # for out-of-range indices
        property_table = np.array(property_rows, dtype=FLOAT)
        idx_clamped = np.clip(residue_types.astype(INT), 0, len(RESIDUE_TYPES))
        physchem = property_table[idx_clamped]

        # Collect all features
        scalar_features = (
            residue_one_hot,
            np.stack(terminal_flags, axis=1),
            self_distance,
            degree_feature,
            has_chi_angles,
            sasa,
            rf_distance,
            physchem,
        )

        vector_features = (
            self_vector,
            rf_vector,
            local_frames,
        )

        return scalar_features, vector_features

    def _extract_interaction_features(self, coords: np.ndarray, distance_cutoff: float = 8,
                                     relative_position_cutoff: int = 32,
                                     knn_cutoff: Optional[int] = None) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple, Tuple]:
        """
        Extract interaction features between residues.

        Args:
            coords: Residue coordinates
            distance_cutoff: Distance cutoff for interactions
            relative_position_cutoff: Cutoff for relative position encoding
            knn_cutoff: Optional k-nearest neighbors cutoff

        Returns:
            Tuple of (edges, scalar_features, vector_features)
        """
        relative_position = self.get_relative_position(cutoff=relative_position_cutoff, onehot=True)
        distance_adj, adj, interaction_vectors = self._calculate_interaction_features(
            coords, distance_cutoff=distance_cutoff, knn_cutoff=knn_cutoff
        )

        # Dense adjacency to edge list
        src, dst, distance = dense_to_edges(distance_adj)

        relative_position = relative_position[src, dst]
        vectors = interaction_vectors[src, dst, :]

        edges = (src, dst)
        edge_scalar_features = (distance, relative_position)
        edge_vector_features = (vectors,)

        return edges, edge_scalar_features, edge_vector_features

    def get_features(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract all features for the protein.

        Returns:
            Tuple of (node_features, edge_features) dictionaries
        """
        residues = self.get_residues()
        coords = np.zeros((len(residues), MAX_ATOMS_PER_RESIDUE, 3), dtype=FLOAT)
        residue_types = np.array(residues)[:, 2].astype(int)

        # Build coordinate tensor using cached coordinates (O(1) lookup)
        for idx, residue in enumerate(residues):
            residue_coord_np = self.get_residue_coordinates_numpy(residue)
            coords[idx, :residue_coord_np.shape[0], :] = residue_coord_np
            # Sidechain centroid (using calculate_sidechain_centroid logic)
            coords[idx, -1, :] = calculate_sidechain_centroid(residue_coord_np)

        # Extract CA and SC coordinates
        coords_CA = coords[:, 1:2, :]
        coords_SC = coords[:, -1:, :]
        coord = np.concatenate([coords_CA, coords_SC], axis=1)

        # Extract features
        node_scalar_features, node_vector_features = self._extract_residue_features(coords, residue_types)
        edges, edge_scalar_features, edge_vector_features = self._extract_interaction_features(
            coords, distance_cutoff=8, relative_position_cutoff=32
        )

        # Package features
        node = {
            'coords': coord,
            'node_scalar_features': node_scalar_features,
            'node_vector_features': node_vector_features
        }

        edge = {
            'edges': edges,
            'edge_scalar_features': edge_scalar_features,
            'edge_vector_features': edge_vector_features
        }

        return node, edge
