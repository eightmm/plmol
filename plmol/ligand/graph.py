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
from rdkit.Chem import rdPartialCharges
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Suppress RDKit C++ warnings (e.g., "Molecule does not have explicit Hs")
RDLogger.DisableLog('rdApp.*')

from ..constants import (
    CHEMICAL_SMARTS, ROTATABLE_BOND_SMARTS,
)
from ..utils import knn_mask_torch
from .graph_atom_features import AtomFeatureMixin
from .graph_edge_features import EdgeFeatureMixin


class MoleculeGraphFeaturizer(AtomFeatureMixin, EdgeFeatureMixin):
    """
    Extracts graph-level features (node and edge) from molecules.

    This class provides methods to convert molecules into graph representations
    suitable for Graph Neural Networks (GNNs).

    Node Features (98 dimensions):
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

    Edge Features (37 dimensions — 27 bond + 10 pair):
        - Bond type (one-hot, 4)
        - Bond stereo (one-hot, 6)
        - Bond direction (one-hot, 5)
        - Aromaticity, conjugation, ring membership, rotatability, bond order (5)
        - Basic pair distance (1)
        - Topological bond context (6)
        - 3D pair features (10)
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
    # Main Entry Point
    # =========================================================================

    def featurize(
        self,
        mol,
        distance_cutoff: Optional[float] = None,
        knn_cutoff: Optional[int] = None,
        generate_conformer: bool = True,
    ) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        Extract complete graph representation with separate bond and distance edges.

        Args:
            mol: RDKit mol object
            distance_cutoff: Optional 3D distance cutoff for spatial edges.
            knn_cutoff: Optional k-nearest neighbors cutoff for spatial edges.
            generate_conformer: Whether to generate 3D coordinates if missing.

        Returns:
            Tuple of (node_dict, edge_dict, adjacency_matrix):
            - node_dict: {'node_feats': [N, 98], 'coords': [N, 3]}
            - edge_dict: {
                'bond_edges': [2, Eb], 'bond_edge_feats': [Eb, ~27],
                'dist_edges': [2, Ed], 'dist_edge_feats': [Ed, 1]
                'pair_features': [N, N, 10],
                'distance_matrix': [N, N],
                'distance_bounds': [N, N, 2],
              }
            - adjacency_matrix: [N, N, 27] (Bond-based)
        """
        # Clear cache for new molecule
        self._clear_cache()

        # Suppress RDKit warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node_features, coords = self.get_atom_features(
                mol,
                generate_conformer=generate_conformer,
            )
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
                mask = mask | knn_mask_torch(dist_matrix, knn_cutoff)

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
