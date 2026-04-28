"""
Protein-Ligand Interaction Graph Encoding.

Builds edge features and interaction graphs from detected interactions.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
import torch

from ..constants import (
    INTERACTION_TYPE_IDX,
    NUM_INTERACTION_TYPES,
    IDEAL_DISTANCES,
    PHARMACOPHORE_IDX,
    HEAVY_ELEMENT_TYPES,
    NUM_HEAVY_ELEMENT_TYPES,
    HYBRIDIZATION_TYPES,
    NUM_HYBRIDIZATION_TYPES,
    RESIDUE_TYPES,
    NUM_RESIDUE_TYPES,
    INTERACTION_STRENGTH_SIGMA,
    IDEAL_DISTANCE_FALLBACK,
    CROSS_CONTACT_DENSITY_CUTOFF,
    CROSS_CONTACT_DENSITY_NORM,
)
from ..utils import knn_mask_bipartite_numpy
from .pli_featurizer import Interaction

# Backward-compat aliases used internally
ELEMENT_TYPES = HEAVY_ELEMENT_TYPES
NUM_ELEMENT_TYPES = NUM_HEAVY_ELEMENT_TYPES


class InteractionGraphBuilder:
    """Builds edge features and graph representations from interactions.

    Args:
        distance_cutoff: Distance cutoff used for interaction detection.
        knn_cutoff: Optional k-nearest neighbors cutoff.
        distance_matrix: (N_p, N_l) pairwise distance matrix.
        num_protein_atoms: Number of protein heavy atoms.
        num_ligand_atoms: Number of ligand heavy atoms.
        protein_atom_features: {heavy_idx: feature_dict}.
        ligand_atom_features: {heavy_idx: feature_dict}.
        protein_residue_info: {heavy_idx: residue_dict}.
    """

    def __init__(
        self,
        distance_cutoff: float,
        knn_cutoff: Optional[int],
        distance_matrix: np.ndarray,
        num_protein_atoms: int,
        num_ligand_atoms: int,
        protein_atom_features: Dict[int, Dict[str, Any]],
        ligand_atom_features: Dict[int, Dict[str, Any]],
        protein_residue_info: Dict[int, Dict[str, Any]],
    ):
        self.distance_cutoff = distance_cutoff
        self.knn_cutoff = knn_cutoff
        self._distance_matrix = distance_matrix
        self.num_protein_atoms = num_protein_atoms
        self.num_ligand_atoms = num_ligand_atoms
        self._protein_atom_features = protein_atom_features
        self._ligand_atom_features = ligand_atom_features
        self._protein_residue_info = protein_residue_info

    def _get_edge_feature_dim(self) -> int:
        """Get total edge feature dimension."""
        return (
            NUM_INTERACTION_TYPES +            # interaction type one-hot (7)
            4 +                                # distance + angles + has_valid (4)
            NUM_ELEMENT_TYPES * 2 +            # element types (20)
            NUM_HYBRIDIZATION_TYPES * 2 +      # hybridization (12)
            2 +                                # formal charges (2)
            2 +                                # aromatic (2)
            4 +                                # in_ring (2) + degree (2)
            NUM_RESIDUE_TYPES +                # residue type (21)
            1 +                                # is_backbone (1)
            1 +                                # interaction_strength (1)
            # Complex-only edge features
            2 +                                # cross-contact density (protein, ligand) (2)
            2 +                                # endpoint min distance to other entity (2)
            1                                  # relative distance in pocket (1)
        )  # Total: 79

    def _build_edge_features(self, interactions: List[Interaction]) -> torch.Tensor:
        """Build comprehensive feature tensor for interactions."""
        num_interactions = len(interactions)
        feature_dim = self._get_edge_feature_dim()
        features = torch.zeros(num_interactions, feature_dim)

        for i, inter in enumerate(interactions):
            offset = 0

            # 1. Interaction type one-hot (7 dims)
            type_idx = INTERACTION_TYPE_IDX.get(inter.interaction_type, 0)
            features[i, offset + type_idx] = 1.0
            offset += NUM_INTERACTION_TYPES

            # 2. Distance and geometric features (4 dims)
            features[i, offset] = inter.distance / self.distance_cutoff
            offset += 1

            # Combined angle (ring/DHA/CXA - use whichever is available)
            angle_val = inter.angle or inter.dha_angle or inter.cxa_angle
            if angle_val is not None:
                features[i, offset] = angle_val / 180.0
            offset += 1

            # Has valid angle flag
            features[i, offset] = float(inter.has_valid_angle)
            offset += 1

            # Angle type indicator (0=none, 0.33=ring, 0.67=dha, 1.0=cxa)
            if inter.angle is not None:
                features[i, offset] = 0.33
            elif inter.dha_angle is not None:
                features[i, offset] = 0.67
            elif inter.cxa_angle is not None:
                features[i, offset] = 1.0
            offset += 1

            # 3. Element types (20 dims)
            p_feats = self._protein_atom_features.get(inter.protein_atom_idx, {})
            l_feats = self._ligand_atom_features.get(inter.ligand_atom_idx, {})

            p_elem_idx = p_feats.get('element_idx', NUM_ELEMENT_TYPES - 1)
            features[i, offset + p_elem_idx] = 1.0
            offset += NUM_ELEMENT_TYPES

            l_elem_idx = l_feats.get('element_idx', NUM_ELEMENT_TYPES - 1)
            features[i, offset + l_elem_idx] = 1.0
            offset += NUM_ELEMENT_TYPES

            # 4. Hybridization (12 dims)
            p_hyb_idx = p_feats.get('hybridization_idx', NUM_HYBRIDIZATION_TYPES - 1)
            features[i, offset + p_hyb_idx] = 1.0
            offset += NUM_HYBRIDIZATION_TYPES

            l_hyb_idx = l_feats.get('hybridization_idx', NUM_HYBRIDIZATION_TYPES - 1)
            features[i, offset + l_hyb_idx] = 1.0
            offset += NUM_HYBRIDIZATION_TYPES

            # 5. Formal charges (2 dims)
            p_charge = p_feats.get('formal_charge', 0)
            l_charge = l_feats.get('formal_charge', 0)
            features[i, offset] = (p_charge + 2) / 4.0
            features[i, offset + 1] = (l_charge + 2) / 4.0
            offset += 2

            # 6. Aromatic (2 dims)
            features[i, offset] = float(p_feats.get('is_aromatic', False))
            features[i, offset + 1] = float(l_feats.get('is_aromatic', False))
            offset += 2

            # 7. Ring membership + degree (4 dims)
            features[i, offset] = float(p_feats.get('is_in_ring', False))
            features[i, offset + 1] = float(l_feats.get('is_in_ring', False))
            features[i, offset + 2] = p_feats.get('degree', 0) / 4.0
            features[i, offset + 3] = l_feats.get('degree', 0) / 4.0
            offset += 4

            # 8. Residue type for protein (21 dims)
            res_info = self._protein_residue_info.get(inter.protein_atom_idx, {})
            res_idx = res_info.get('residue_idx', NUM_RESIDUE_TYPES - 1)
            features[i, offset + res_idx] = 1.0
            offset += NUM_RESIDUE_TYPES

            # 9. Is backbone (1 dim)
            is_backbone = res_info.get('is_backbone', False)
            features[i, offset] = float(is_backbone)
            offset += 1

            # 10. Interaction strength: Gaussian decay from ideal distance (1 dim)
            ideal = IDEAL_DISTANCES.get(inter.interaction_type, IDEAL_DISTANCE_FALLBACK)
            strength = math.exp(-0.5 * ((inter.distance - ideal) / INTERACTION_STRENGTH_SIGMA) ** 2)
            features[i, offset] = strength
            offset += 1

            # --- Complex-only edge features ---

            # 11. Cross-contact density (2 dims)
            # Number of atoms from the other entity within CROSS_CONTACT_DENSITY_CUTOFF of each endpoint
            p_idx = inter.protein_atom_idx
            l_idx = inter.ligand_atom_idx
            if p_idx < self._distance_matrix.shape[0]:
                p_contacts = int((self._distance_matrix[p_idx, :] < CROSS_CONTACT_DENSITY_CUTOFF).sum())
                features[i, offset] = min(p_contacts / CROSS_CONTACT_DENSITY_NORM, 1.0)
            offset += 1
            if l_idx < self._distance_matrix.shape[1]:
                l_contacts = int((self._distance_matrix[:, l_idx] < CROSS_CONTACT_DENSITY_CUTOFF).sum())
                features[i, offset] = min(l_contacts / CROSS_CONTACT_DENSITY_NORM, 1.0)
            offset += 1

            # 12. Endpoint min cross-distance (2 dims)
            # How close is each endpoint to its nearest partner atom
            if p_idx < self._distance_matrix.shape[0]:
                p_min_dist = float(self._distance_matrix[p_idx, :].min())
                features[i, offset] = min(p_min_dist / self.distance_cutoff, 1.0)
            offset += 1
            if l_idx < self._distance_matrix.shape[1]:
                l_min_dist = float(self._distance_matrix[:, l_idx].min())
                features[i, offset] = min(l_min_dist / self.distance_cutoff, 1.0)
            offset += 1

            # 13. Relative distance in pocket (1 dim)
            # How deep this interaction is relative to the pocket boundary
            max_dist = float(self._distance_matrix.max()) if self._distance_matrix.size > 0 else 1.0
            features[i, offset] = inter.distance / max(max_dist, 1e-6)
            offset += 1

        return features

    def _get_contact_edges(
        self, distance_cutoff: float,
        knn_cutoff: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all heavy atom pairs within cutoff as generic contact edges.

        Args:
            distance_cutoff: Distance cutoff in Angstrom.
            knn_cutoff: Optional k-nearest neighbors cutoff for bipartite edges.

        Returns:
            contact_edges: (2, E_contact) protein/ligand heavy atom indices.
            contact_distances: (E_contact,) pairwise distances.
        """
        dm = self._distance_matrix
        mask = dm < distance_cutoff

        knn_cutoff = knn_cutoff or self.knn_cutoff
        if knn_cutoff is not None:
            mask = mask | knn_mask_bipartite_numpy(dm, knn_cutoff)

        p_idx, l_idx = np.where(mask)
        if len(p_idx) == 0:
            return (
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
            )
        edges = torch.tensor(np.stack([p_idx, l_idx]), dtype=torch.long)
        distances = torch.tensor(
            self._distance_matrix[mask], dtype=torch.float32
        )
        return edges, distances

    def get_interaction_edges(
        self, interactions: List[Interaction],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interaction edges as tensors.

        Returns:
            Tuple of (edges, edge_features):
                - edges: [2, num_interactions] tensor of heavy atom indices
                - edge_features: [num_interactions, feature_dim] tensor
        """
        if not interactions:
            return (
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, self._get_edge_feature_dim(), dtype=torch.float32)
            )

        edges = torch.tensor([
            [inter.protein_atom_idx for inter in interactions],
            [inter.ligand_atom_idx for inter in interactions]
        ], dtype=torch.long)

        edge_features = self._build_edge_features(interactions)

        return edges, edge_features

    def get_interaction_graph(
        self,
        interactions: List[Interaction],
        edges: torch.Tensor,
        edge_features: torch.Tensor,
        include_contacts: bool = False,
        contact_cutoff: Optional[float] = None,
        knn_cutoff: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get complete interaction graph data."""
        type_counts = {}
        for inter in interactions:
            type_counts[inter.interaction_type] = type_counts.get(inter.interaction_type, 0) + 1

        knn_cutoff = knn_cutoff or self.knn_cutoff
        graph = {
            'edges': edges,
            'edge_features': edge_features,
            'interactions': interactions,
            'num_interactions': len(interactions),
            'interaction_counts': type_counts,
            'num_protein_atoms': self.num_protein_atoms,
            'num_ligand_atoms': self.num_ligand_atoms,
            'distance_cutoff': self.distance_cutoff,
            'knn_cutoff': knn_cutoff,
            'feature_dim': self._get_edge_feature_dim(),
            'metadata': {
                'interaction_type_indices': INTERACTION_TYPE_IDX,
                'pharmacophore_indices': PHARMACOPHORE_IDX,
                'element_types': ELEMENT_TYPES,
                'residue_types': RESIDUE_TYPES,
                'heavy_atom_only': True,
            }
        }

        if include_contacts:
            c_cutoff = contact_cutoff or self.distance_cutoff
            contact_edges, contact_distances = self._get_contact_edges(
                c_cutoff, knn_cutoff=knn_cutoff
            )
            graph['contact_edges'] = contact_edges
            graph['contact_distances'] = contact_distances
            graph['num_contacts'] = contact_edges.shape[1]

        return graph

    def get_interaction_summary(
        self, interactions: List[Interaction],
    ) -> str:
        """Get text summary of detected interactions."""
        type_counts = {}
        angles_available = 0
        for inter in interactions:
            type_counts[inter.interaction_type] = type_counts.get(inter.interaction_type, 0) + 1
            if inter.has_valid_angle:
                angles_available += 1

        lines = [
            f"PLI Summary (Heavy Atom Only)",
            f"  Protein heavy atoms: {self.num_protein_atoms}",
            f"  Ligand heavy atoms: {self.num_ligand_atoms}",
            f"  Distance cutoff: {self.distance_cutoff} \u00c5",
            f"  Total interactions: {len(interactions)}",
            f"  Valid angle calculations: {angles_available}/{len(interactions)}",
            f"  Feature dimension: {self._get_edge_feature_dim()}",
            "",
            "Interaction counts:"
        ]

        for itype in INTERACTION_TYPE_IDX:
            count = type_counts.get(itype, 0)
            lines.append(f"  {itype}: {count}")

        return '\n'.join(lines)

    def get_feature_description(self) -> Dict[str, Any]:
        """Get description of feature dimensions."""
        offset = 0
        breakdown = {}

        breakdown['interaction_type'] = (offset, NUM_INTERACTION_TYPES)
        offset += NUM_INTERACTION_TYPES

        breakdown['distance'] = (offset, 1)
        offset += 1
        breakdown['angle'] = (offset, 1)
        offset += 1
        breakdown['has_valid_angle'] = (offset, 1)
        offset += 1
        breakdown['angle_type'] = (offset, 1)
        offset += 1

        breakdown['protein_element'] = (offset, NUM_ELEMENT_TYPES)
        offset += NUM_ELEMENT_TYPES
        breakdown['ligand_element'] = (offset, NUM_ELEMENT_TYPES)
        offset += NUM_ELEMENT_TYPES

        breakdown['protein_hybridization'] = (offset, NUM_HYBRIDIZATION_TYPES)
        offset += NUM_HYBRIDIZATION_TYPES
        breakdown['ligand_hybridization'] = (offset, NUM_HYBRIDIZATION_TYPES)
        offset += NUM_HYBRIDIZATION_TYPES

        breakdown['formal_charges'] = (offset, 2)
        offset += 2

        breakdown['aromatic'] = (offset, 2)
        offset += 2

        breakdown['ring_and_degree'] = (offset, 4)
        offset += 4

        breakdown['residue_type'] = (offset, NUM_RESIDUE_TYPES)
        offset += NUM_RESIDUE_TYPES

        breakdown['is_backbone'] = (offset, 1)
        offset += 1

        breakdown['interaction_strength'] = (offset, 1)
        offset += 1

        breakdown['cross_contact_density'] = (offset, 2)
        offset += 2

        breakdown['endpoint_min_distance'] = (offset, 2)
        offset += 2

        breakdown['relative_pocket_distance'] = (offset, 1)
        offset += 1

        return {
            'total_dim': offset,
            'breakdown': breakdown
        }
