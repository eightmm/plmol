"""
Interaction Featurizer Module.

Provides protein-ligand interaction feature extraction for GNN models.

Classes:
    - PLInteractionFeaturizer: Protein-ligand edge feature extraction
    - PocketExtractor: Binding pocket extraction from protein
"""

from .pli_featurizer import PLInteractionFeaturizer, Interaction
from .pocket_extractor import extract_pocket, PocketExtractor, PocketInfo, ParsedProtein
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
    # Main classes
    'PLInteractionFeaturizer',
    'Interaction',
    'PocketExtractor',
    'PocketInfo',
    'ParsedProtein',
    # Functions
    'extract_pocket',
    # SMARTS patterns
    'PHARMACOPHORE_SMARTS',
    'PHARMACOPHORE_CATEGORIES',
    'CHEMICAL_SMARTS',
    # Interaction definitions
    'INTERACTION_TYPES',
    'INTERACTION_TYPE_IDX',
    'NUM_INTERACTION_TYPES',
    'IDEAL_DISTANCES',
    'PHARMACOPHORE_IDX',
    'NUM_PHARMACOPHORE_TYPES',
]
