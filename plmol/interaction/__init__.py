"""
Interaction Featurizer Module.

Provides protein-ligand interaction feature extraction for GNN models.

Classes:
    - PLInteractionFeaturizer: Protein-ligand edge feature extraction
    - PocketExtractor: Binding pocket extraction from protein
"""

from .pli_featurizer import PLInteractionFeaturizer, Interaction
from .pli_detectors import InteractionDetector
from .pli_encoding import InteractionGraphBuilder
from .pocket_extractor import extract_pocket, PocketExtractor, PocketInfo, ParsedProtein
from .metal_coordination import MetalSite, detect_metal_sites, classify_coordination_geometry, encode_metal_features
from ..constants import (
    # Primary pharmacophore patterns
    PHARMACOPHORE_SMARTS,
    PHARMACOPHORE_CATEGORIES,
    # Backward compatibility
    CHEMICAL_SMARTS,
    # Interaction definitions
    INTERACTION_TYPES,
    INTERACTION_TYPE_IDX,
    NUM_INTERACTION_TYPES,
    IDEAL_DISTANCES,
    PHARMACOPHORE_IDX,
    NUM_PHARMACOPHORE_TYPES,
)

__all__ = [
    'PLInteractionFeaturizer',
    'Interaction',
    'PocketExtractor',
    'PocketInfo',
    'ParsedProtein',
    'extract_pocket',
    'MetalSite',
    'detect_metal_sites',
    'classify_coordination_geometry',
    'encode_metal_features',
]
