"""
Efficient Protein Featurizer with one-time parsing.

This module provides a high-level API for protein feature extraction
with efficient caching of parsed PDB data.
"""

import os
import tempfile
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np

from .pdb_standardizer import PDBStandardizer
from .residue_featurizer import ResidueFeaturizer
from .utils import calculate_sidechain_centroid, normalize_residue_name
from .geometry import (
    calculate_backbone_curvature,
    calculate_backbone_torsion,
    calculate_self_distances_vectors,
)
from .backbone_featurizer import compute_backbone_features
from ..constants import (
    MAX_ATOMS_PER_RESIDUE,
    NUM_RESIDUE_TYPES,
    VDW_RADIUS,
    DEFAULT_VDW_RADIUS,
    ELEMENT_SYMBOL_TO_ATOMIC_NUMBER,
    DEFAULT_BACKBONE_KNN_NEIGHBORS,
    SURFACE_DEFAULT_GRID_DENSITY,
    SURFACE_DEFAULT_THRESHOLD,
    SURFACE_DEFAULT_SHARPNESS,
    SURFACE_DEFAULT_MAX_MEMORY_GB,
    SURFACE_DEFAULT_POINTS_PER_ATOM,
    SURFACE_DEFAULT_PROBE_RADIUS,
    VOXEL_DEFAULT_RESOLUTION,
    VOXEL_DEFAULT_BOX_SIZE,
    VOXEL_DEFAULT_PADDING,
    VOXEL_DEFAULT_SIGMA_SCALE,
    VOXEL_DEFAULT_CUTOFF_SIGMA,
)
from ..featurizers.surface import build_protein_surface
from ..featurizers.voxel import build_protein_voxel
from .utils import PDBParser


class ProteinFeaturizer:
    """
    Efficient protein featurizer that parses PDB once and caches results.

    Examples:
        >>> # Parse once, extract multiple features efficiently
        >>> featurizer = ProteinFeaturizer("protein.pdb")
        >>> sequence = featurizer.get_sequence_features()
        >>> geometry = featurizer.get_geometric_features()
        >>> sasa = featurizer.get_sasa_features()
    """

    def __init__(self, pdb_file: str, standardize: bool = True,
                 keep_hydrogens: bool = False):
        """
        Initialize and parse PDB file once.

        Args:
            pdb_file: Path to PDB file
            standardize: Whether to standardize the PDB first
            keep_hydrogens: Whether to keep hydrogens during standardization
        """
        self.input_file = pdb_file
        self.standardize = standardize
        self.keep_hydrogens = keep_hydrogens

        # Check if file exists
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        # Standardize if requested
        if standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                self.tmp_pdb = tmp_file.name

            standardizer = PDBStandardizer(remove_hydrogens=not keep_hydrogens)
            standardizer.standardize(pdb_file, self.tmp_pdb)
            pdb_to_process = self.tmp_pdb
        else:
            self.tmp_pdb = None
            pdb_to_process = pdb_file

        # Parse PDB once
        self._featurizer = ResidueFeaturizer(pdb_to_process)
        self._parse_structure()

        # Cache for computed features
        self._cache = {}

    def _parse_structure(self):
        """Parse structure and cache basic data."""
        # Get residues
        self.residues = self._featurizer.get_residues()
        self.num_residues = len(self.residues)

        # Build coordinate tensor
        self.coords = torch.zeros(self.num_residues, MAX_ATOMS_PER_RESIDUE, 3)
        self.residue_types = torch.from_numpy(
            np.array(self.residues)[:, 2].astype(int)
        )

        for idx, residue in enumerate(self.residues):
            # For unknown residues (type 20), only use backbone + CB atoms
            res_type = residue[2]
            if res_type == 20:  # UNK residue
                atom_coord_map = self._featurizer.get_residue_coordinates(residue)
                standard_unk_atoms = ['N', 'CA', 'C', 'O', 'CB']
                filtered_coords = [
                    atom_coord_map.get(name, np.zeros(3, dtype=np.float32))
                    for name in standard_unk_atoms
                ]
                residue_coord_np = np.vstack(filtered_coords).astype(np.float32)
            else:
                # Use cached coordinates (O(1) dict lookup)
                residue_coord_np = self._featurizer.get_residue_coordinates_numpy(residue)

            residue_coord = torch.from_numpy(residue_coord_np)
            self.coords[idx, :residue_coord.shape[0], :] = residue_coord
            # Sidechain centroid (using unified calculate_sidechain_centroid)
            self.coords[idx, -1, :] = torch.from_numpy(
                calculate_sidechain_centroid(residue_coord_np)
            )

        # Extract CA and SC coordinates
        self.coords_CA = self.coords[:, 1:2, :]
        self.coords_SC = self.coords[:, -1:, :]
        self.coord = torch.cat([self.coords_CA, self.coords_SC], dim=1)

    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'tmp_pdb') and self.tmp_pdb and os.path.exists(self.tmp_pdb):
            os.unlink(self.tmp_pdb)

    def get_sequence_features(self) -> Dict[str, Any]:
        """
        Get amino acid sequence and position features.

        Returns:
            Dictionary with residue types and one-hot encoding
        """
        if 'sequence' not in self._cache:
            # Bounds checking for one-hot encoding
            residue_types_clamped = torch.clamp(self.residue_types, 0, NUM_RESIDUE_TYPES - 1)
            residue_one_hot = torch.nn.functional.one_hot(
                residue_types_clamped, num_classes=NUM_RESIDUE_TYPES
            )

            self._cache['sequence'] = {
                'residue_types': self.residue_types,
                'residue_one_hot': residue_one_hot,
                'num_residues': self.num_residues
            }

        return self._cache['sequence']

    def get_geometric_features(self) -> Dict[str, Any]:
        """
        Get geometric features including distances, angles, and dihedrals.

        Returns:
            Dictionary with geometric measurements
        """
        if 'geometric' not in self._cache:
            # Get geometric features
            dihedrals, has_chi = self._featurizer.get_dihedral_angles(
                self.coords, self.residue_types
            )
            terminal_flags = self.get_terminal_flags()
            curvature = calculate_backbone_curvature(
                self.coords, (terminal_flags['n_terminal'], terminal_flags['c_terminal'])
            )
            torsion = calculate_backbone_torsion(
                self.coords, (terminal_flags['n_terminal'], terminal_flags['c_terminal'])
            )
            self_distance, self_vector = calculate_self_distances_vectors(
                self.coords
            )

            self._cache['geometric'] = {
                'dihedrals': dihedrals,
                'has_chi_angles': has_chi,
                'backbone_curvature': curvature,
                'backbone_torsion': torsion,
                'self_distances': self_distance,
                'self_vectors': self_vector,
                'coordinates': self.coords
            }

        return self._cache['geometric']

    def get_sasa_features(self) -> torch.Tensor:
        """
        Get Solvent Accessible Surface Area features.

        Returns:
            SASA tensor with multiple components per residue
        """
        if 'sasa' not in self._cache:
            self._cache['sasa'] = self._featurizer.calculate_sasa()

        return self._cache['sasa']

    def get_contact_map(self, cutoff: float = 8.0) -> Dict[str, Any]:
        """
        Get residue-residue contact map and distances.

        Args:
            cutoff: Distance cutoff for contacts (default: 8.0 Å)

        Returns:
            Dictionary with contact information
        """
        cache_key = f'contact_map_{cutoff}'

        if cache_key not in self._cache:
            distance_adj, adj, vectors = self._featurizer._calculate_interaction_features(
                self.coords, cutoff=cutoff
            )

            # Get sparse representation
            sparse = distance_adj.to_sparse(sparse_dim=2)
            src, dst = sparse.indices()
            distances = sparse.values()

            self._cache[cache_key] = {
                'adjacency_matrix': adj,
                'distance_matrix': distance_adj,
                'edges': (src, dst),
                'edge_distances': distances,
                'interaction_vectors': vectors
            }

        return self._cache[cache_key]

    def get_relative_position(self, cutoff: int = 32) -> torch.Tensor:
        """
        Get relative position encoding between residues.

        Args:
            cutoff: Maximum relative position to consider

        Returns:
            One-hot encoded relative position tensor
        """
        cache_key = f'relative_position_{cutoff}'

        if cache_key not in self._cache:
            self._cache[cache_key] = self._featurizer.get_relative_position(
                cutoff=cutoff, onehot=True
            )

        return self._cache[cache_key]

    def get_node_features(self) -> Dict[str, Any]:
        """
        Get all node (residue-level) features.

        Returns:
            Dictionary with scalar and vector node features
        """
        if 'node_features' not in self._cache:
            scalar_features, vector_features = self._featurizer._extract_residue_features(
                self.coords, self.residue_types
            )

            self._cache['node_features'] = {
                'coordinates': self.coord,
                'scalar_features': scalar_features,
                'vector_features': vector_features
            }

        return self._cache['node_features']

    def get_edge_features(self, distance_cutoff: float = 8.0) -> Dict[str, Any]:
        """
        Get all edge (interaction) features.

        Args:
            distance_cutoff: Distance cutoff for interactions

        Returns:
            Dictionary with edge indices and features
        """
        cache_key = f'edge_features_{distance_cutoff}'

        if cache_key not in self._cache:
            edges, scalar_features, vector_features = \
                self._featurizer._extract_interaction_features(
                    self.coords, distance_cutoff=distance_cutoff
                )

            self._cache[cache_key] = {
                'edges': edges,
                'scalar_features': scalar_features,
                'vector_features': vector_features
            }

        return self._cache[cache_key]

    def get_terminal_flags(self) -> Dict[str, torch.Tensor]:
        """
        Get N-terminal and C-terminal residue flags.

        Returns:
            Dictionary with terminal flags
        """
        if 'terminal_flags' not in self._cache:
            n_terminal, c_terminal = self._featurizer.get_terminal_flags()
            self._cache['terminal_flags'] = {
                'n_terminal': n_terminal,
                'c_terminal': c_terminal
            }

        return self._cache['terminal_flags']

    def get_features(self, distance_cutoff: float = 8.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get node and edge features in standard format.

        Args:
            distance_cutoff: Distance cutoff for residue-residue edges (default: 8.0 Å)

        Returns:
            Tuple of (node, edge) dictionaries with:
            - node: {'coord', 'node_scalar_features', 'node_vector_features'}
            - edge: {'edges', 'edge_scalar_features', 'edge_vector_features'}
        """
        cache_key = f'features_{distance_cutoff}'
        if cache_key not in self._cache:
            # Get edges with the specified cutoff
            edges, edge_scalar_features, edge_vector_features = \
                self._featurizer._extract_interaction_features(
                    self.coords, distance_cutoff=distance_cutoff
                )

            # Get node features
            node_scalar_features, node_vector_features = \
                self._featurizer._extract_residue_features(
                    self.coords, self.residue_types
                )

            node = {
                'coord': self.coord,
                'node_scalar_features': node_scalar_features,
                'node_vector_features': node_vector_features
            }

            edge = {
                'edges': edges,
                'edge_scalar_features': edge_scalar_features,
                'edge_vector_features': edge_vector_features
            }

            self._cache[cache_key] = (node, edge)
        return self._cache[cache_key]

    def get_all_features(self, save_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all features at once.

        Args:
            save_to: Optional path to save features

        Returns:
            Dictionary containing all features
        """
        node_features = self.get_node_features()
        edge_features = self.get_edge_features()

        features = {
            'node': node_features,
            'edge': edge_features,
            'metadata': {
                'input_file': self.input_file,
                'standardized': self.standardize,
                'hydrogens_removed': not self.keep_hydrogens if self.standardize else None,
                'num_residues': self.num_residues
            }
        }

        if save_to:
            torch.save(features, save_to)

        return features

    def get_surface(
        self,
        grid_density: float = SURFACE_DEFAULT_GRID_DENSITY,
        threshold: float = SURFACE_DEFAULT_THRESHOLD,
        sharpness: float = SURFACE_DEFAULT_SHARPNESS,
        include_features: bool = True,
        max_memory_gb: float = SURFACE_DEFAULT_MAX_MEMORY_GB,
        device: Optional[str] = None,
        mode: str = "mesh",
        n_points_per_atom: int = SURFACE_DEFAULT_POINTS_PER_ATOM,
        probe_radius: float = SURFACE_DEFAULT_PROBE_RADIUS,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Create a protein surface from atom coordinates.

        Args:
            mode: "mesh" for marching cubes mesh, "point_cloud" for SAS point cloud.
            n_points_per_atom: Points per atom for point cloud mode (default: 100).
            probe_radius: Solvent probe radius for point cloud mode (default: 1.4).

        Returns:
            Dict with "points", "normals" (and legacy "verts"). "faces" is
            included only in mesh mode. Plus protein-tailored residue/geometric
            features mapped to the surface when include_features is True.
        """
        cache_key = (
            f"surface_{grid_density}_{threshold}_{sharpness}_{include_features}_"
            f"{max_memory_gb}_{device}_{mode}_{n_points_per_atom}_{probe_radius}"
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        pdb_to_use = self.tmp_pdb if self.tmp_pdb else self.input_file
        parser = PDBParser(pdb_to_use)
        atoms = parser.protein_atoms
        if not atoms:
            self._cache[cache_key] = None
            return None

        coords = np.array([atom.coords for atom in atoms], dtype=np.float32)
        radii = np.array(
            [
                VDW_RADIUS.get(
                    ELEMENT_SYMBOL_TO_ATOMIC_NUMBER.get((atom.element or "").upper(), 0),
                    DEFAULT_VDW_RADIUS,
                )
                for atom in atoms
            ],
            dtype=np.float32,
        )

        atom_metadata = [
            {
                "res_name": normalize_residue_name(atom.res_name, atom.atom_name),
                "atom_name": atom.atom_name,
                "element": atom.element or "",
                "b_factor": 0.0,
            }
            for atom in atoms
        ]
        try:
            surface = build_protein_surface(
                coords=coords,
                radii=radii,
                atom_metadata=atom_metadata,
                grid_density=grid_density,
                threshold=threshold,
                sharpness=sharpness,
                include_features=include_features,
                max_memory_gb=max_memory_gb,
                device=device,
                mode=mode,
                n_points_per_atom=n_points_per_atom,
                probe_radius=probe_radius,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Surface featurization requires optional dependencies (open3d, scikit-image, trimesh). "
                "Install them to enable surface features."
            ) from exc
        if surface is None:
            self._cache[cache_key] = None
            return None

        self._cache[cache_key] = surface
        return surface

    def get_voxel(
        self,
        center: Optional[np.ndarray] = None,
        resolution: float = VOXEL_DEFAULT_RESOLUTION,
        box_size: Optional[int] = VOXEL_DEFAULT_BOX_SIZE,
        padding: float = VOXEL_DEFAULT_PADDING,
        sigma_scale: float = VOXEL_DEFAULT_SIGMA_SCALE,
        cutoff_sigma: float = VOXEL_DEFAULT_CUTOFF_SIGMA,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Create a protein voxel representation from atom coordinates.

        Each atom's features are Gaussian-smeared onto a 3D grid,
        producing a multi-channel volume suitable for 3D CNNs.

        Channels (16): occupancy, atom type (6), charge, hydrophobicity,
        HBD, HBA, aromaticity, pos/neg ionizable, backbone, b_factor.

        Args:
            center: Grid center (3,). None = protein centroid.
            resolution: Angstrom per voxel (default: 1.0).
            box_size: Grid dimension per axis (default: 24). None for adaptive.
            padding: Padding in Angstrom when box_size is None.
            sigma_scale: VdW radius multiplier for Gaussian sigma.
            cutoff_sigma: Gaussian cutoff in sigma units.

        Returns:
            Dict with "voxel" (16, D, H, W), "channel_names", "grid_origin",
            "grid_shape", "resolution".
        """
        center_key = "None" if center is None else f"{center[0]:.2f}_{center[1]:.2f}_{center[2]:.2f}"
        cache_key = (
            f"voxel_{center_key}_{resolution}_{box_size}_{padding}_{sigma_scale}_{cutoff_sigma}"
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        pdb_to_use = self.tmp_pdb if self.tmp_pdb else self.input_file
        parser = PDBParser(pdb_to_use)
        atoms = parser.protein_atoms
        if not atoms:
            self._cache[cache_key] = None
            return None

        coords = np.array([atom.coords for atom in atoms], dtype=np.float32)
        radii = np.array(
            [
                VDW_RADIUS.get(
                    ELEMENT_SYMBOL_TO_ATOMIC_NUMBER.get((atom.element or "").upper(), 0),
                    DEFAULT_VDW_RADIUS,
                )
                for atom in atoms
            ],
            dtype=np.float32,
        )

        atom_metadata = [
            {
                "res_name": normalize_residue_name(atom.res_name, atom.atom_name),
                "atom_name": atom.atom_name,
                "element": atom.element or "",
                "b_factor": 0.0,
            }
            for atom in atoms
        ]

        voxel = build_protein_voxel(
            coords=coords,
            radii=radii,
            atom_metadata=atom_metadata,
            center=center,
            resolution=resolution,
            box_size=box_size,
            padding=padding,
            sigma_scale=sigma_scale,
            cutoff_sigma=cutoff_sigma,
        )

        self._cache[cache_key] = voxel
        return voxel

    # Alias for backward compatibility
    extract = get_all_features

    # ============== ATOM-LEVEL FEATURES ==============

    def get_atom_graph(self, distance_cutoff: float = 4.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get atom-level graph representation with node and edge features.

        Args:
            distance_cutoff: Distance cutoff for atom-atom edges (default: 4.0 Å)

        Returns:
            Tuple of (node, edge) dictionaries:
                - node: token-based features + enriched scalar features
                - edge: distance, same_residue, sequence_separation, unit_vector
        """
        cache_key = f'atom_graph_{distance_cutoff}'

        if cache_key not in self._cache:
            from .atom_featurizer import AtomFeaturizer

            atom_featurizer = AtomFeaturizer()
            pdb_to_use = self.tmp_pdb if self.tmp_pdb else self.input_file

            # Get atom features with SASA and enriched features
            atom_features = atom_featurizer.get_all_atom_features(pdb_to_use)

            # Build distance matrix using torch
            coords = atom_features['coord']
            if not isinstance(coords, torch.Tensor):
                coords = torch.tensor(coords, dtype=torch.float32)
            coords = coords.float()
            dist_matrix = torch.cdist(coords, coords, p=2)

            # Create edges based on distance cutoff
            edge_mask = (dist_matrix < distance_cutoff) & (dist_matrix > 0)
            edge_index = edge_mask.nonzero(as_tuple=False)
            edges = (edge_index[:, 0].long(), edge_index[:, 1].long())
            edge_distances = dist_matrix[edge_mask].float()

            # Package node features
            residue_nums = atom_features['metadata']['residue_numbers']
            if isinstance(residue_nums, torch.Tensor):
                residue_number_tensor = residue_nums.clone()
            else:
                residue_number_tensor = torch.tensor(residue_nums, dtype=torch.long)

            # Create residue_count: sequential index starting from 0
            chain_labels = atom_features['metadata']['chain_labels']
            residue_count = torch.zeros_like(residue_number_tensor)
            if len(residue_number_tensor) > 0:
                current_count = 0
                residue_count[0] = current_count
                for i in range(1, len(residue_number_tensor)):
                    residue_changed = (residue_number_tensor[i] != residue_number_tensor[i-1]) or \
                                    (chain_labels[i] != chain_labels[i-1])
                    if residue_changed:
                        current_count += 1
                    residue_count[i] = current_count

            # --- Edge features ---
            src, dst = edges

            # same_residue: 1 if both atoms belong to same residue
            same_residue = (residue_count[src] == residue_count[dst]).float()

            # sequence_separation: |residue_count_i - residue_count_j|, capped at 32
            seq_sep = (residue_count[src] - residue_count[dst]).abs().float()
            seq_sep = torch.clamp(seq_sep, max=32.0)

            # unit_vector: normalized direction from src to dst
            diff = coords[dst] - coords[src]  # (E, 3)
            dist_safe = edge_distances.clamp(min=1e-6).unsqueeze(-1)  # (E, 1)
            unit_vector = diff / dist_safe  # (E, 3)

            node = {
                'coord': atom_features['coord'],
                'node_features': atom_features['token'],
                'atom_tokens': atom_features['token'],
                'sasa': atom_features['sasa'],
                'relative_sasa': atom_features['relative_sasa'],
                'residue_token': atom_features['residue_token'],
                'atom_element': atom_features['atom_element'],
                'residue_number': residue_number_tensor,
                'residue_count': residue_count,
                'b_factor': atom_features['b_factor'],
                'b_factor_zscore': atom_features['b_factor_zscore'],
                'is_backbone': atom_features['is_backbone'],
                'formal_charge': atom_features['formal_charge'],
                'is_hbond_donor': atom_features['is_hbond_donor'],
                'is_hbond_acceptor': atom_features['is_hbond_acceptor'],
                'secondary_structure': atom_features['secondary_structure'],
                'atom_name': atom_features['metadata']['atom_names'],
                'chain_label': chain_labels,
            }

            edge = {
                'edges': edges,
                'edge_distances': edge_distances,
                'same_residue': same_residue,
                'sequence_separation': seq_sep,
                'unit_vector': unit_vector,
                'distance_cutoff': distance_cutoff,
            }

            self._cache[cache_key] = (node, edge)

        return self._cache[cache_key]

    # Primary alias for atom-level graph
    get_atom_features = get_atom_graph

    def get_atom_tokens_and_coords(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom-level tokenized features and coordinates.

        Returns:
            Tuple of (token, coord):
                - token: Atom type tokens (175 types)
                - coord: 3D coordinates
        """
        if 'atom_tokens_coords' not in self._cache:
            from .atom_featurizer import AtomFeaturizer
            atom_featurizer = AtomFeaturizer()

            # Use the standardized PDB if available
            pdb_to_use = self.tmp_pdb if self.tmp_pdb else self.input_file
            token, coord = atom_featurizer.get_protein_atom_features(pdb_to_use)
            self._cache['atom_tokens_coords'] = (token, coord)
        return self._cache['atom_tokens_coords']

    # Alias
    get_atom_tokens = get_atom_tokens_and_coords

    def get_atom_features_with_sasa(self) -> Dict[str, Any]:
        """
        Get comprehensive atom features including SASA.

        Returns:
            Dictionary with atom features and SASA
        """
        if 'atom_features_sasa' not in self._cache:
            from .atom_featurizer import AtomFeaturizer
            atom_featurizer = AtomFeaturizer()

            # Use the standardized PDB if available
            pdb_to_use = self.tmp_pdb if self.tmp_pdb else self.input_file
            features = atom_featurizer.get_all_atom_features(pdb_to_use)
            self._cache['atom_features_sasa'] = features
        return self._cache['atom_features_sasa']

    # Alias
    get_atom_sasa = get_atom_features_with_sasa

    def get_atom_coordinates(self) -> torch.Tensor:
        """
        Get only atom-level 3D coordinates.

        Returns:
            torch.Tensor: [n_atoms, 3] coordinates
        """
        token, coord = self.get_atom_tokens_and_coords()
        return coord

    def get_atom_tokens_only(self) -> torch.Tensor:
        """
        Get only atom-level tokens without coordinates.

        Returns:
            torch.Tensor: [n_atoms] token IDs (0-174)
        """
        token, coord = self.get_atom_tokens_and_coords()
        return token

    # ============== RESIDUE-LEVEL ALIASES ==============
    # Cleaner aliases - removed redundant _level_ variants

    # Sequence
    get_residue_sequence = get_sequence_features

    # Geometry
    get_residue_geometry = get_geometric_features

    # SASA
    get_residue_sasa = get_sasa_features

    # Contact map
    get_residue_contacts = get_contact_map

    # Graph features
    get_residue_features = get_features

    # ============== SEQUENCE FEATURES ==============

    def get_sequence_by_chain(self) -> Dict[str, str]:
        """
        Get amino acid sequences in one-letter code separated by chain.

        Returns:
            Dictionary mapping chain IDs to one-letter amino acid sequences
        """
        return self._featurizer.get_sequence_by_chain()

    def get_backbone(self, k_neighbors: int = DEFAULT_BACKBONE_KNN_NEIGHBORS) -> Dict[str, Any]:
        """Backbone features for inverse folding (ProteinMPNN/ESM-IF/GVP).

        Args:
            k_neighbors: Number of nearest neighbors for kNN graph.

        Returns:
            Dict with backbone_coords, cb_coords, dihedrals, dihedrals_mask,
            orientation_frames, residue_types, chain_ids, residue_mask,
            edge_index, edge_dist, edge_unit_vec, edge_seq_sep,
            edge_same_chain, num_residues, num_chains, k_neighbors.
        """
        cache_key = f"backbone_{k_neighbors}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = compute_backbone_features(
            self.coords, self.residues, self.residue_types, k_neighbors
        )

        self._cache[cache_key] = result
        return result
