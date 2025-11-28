"""
PLFeature Constants Module.

Centralized constants for protein-ligand featurization including:
- Element and periodic table data
- Amino acid mappings and tokens
- SMARTS patterns for chemical features
- Interaction type definitions
- Physical properties (radii, energies, etc.)
"""

# =============================================================================
# Element Constants
# =============================================================================
from .elements import (
    # Atom type lists
    ATOM_TYPES,
    HEAVY_ELEMENT_TYPES,
    NUM_HEAVY_ELEMENT_TYPES,
    PROTEIN_ELEMENT_TYPES,
    ATOM_NAME_TO_ELEMENT,

    # Simplified element types for hierarchical models
    SIMPLIFIED_ELEMENT_TYPES,
    NUM_SIMPLIFIED_ELEMENT_TYPES,
    METAL_ELEMENTS,

    # Periodic table
    PERIODIC_TABLE,
    PERIODS,
    GROUPS,
    ELECTRONEGATIVITY,
    DEFAULT_ELECTRONEGATIVITY,

    # RDKit types
    HYBRIDIZATION_TYPES,
    NUM_HYBRIDIZATION_TYPES,
    BOND_TYPES,
    BOND_STEREOS,
    BOND_DIRS,

    # Degree/valence
    DEGREES,
    HEAVY_DEGREES,
    VALENCES,
    TOTAL_HS,
)

# =============================================================================
# Amino Acid Constants
# =============================================================================
from .amino_acids import (
    # Amino acid mappings
    AMINO_ACID_3TO1,
    AMINO_ACID_1TO3,
    AMINO_ACID_1_TO_INT,
    AMINO_ACID_3_TO_INT,
    AMINO_ACID_LETTERS,

    # Residue tokens
    RESIDUE_TYPES,
    NUM_RESIDUE_TYPES,
    MAX_ATOMS_PER_RESIDUE,
    RESIDUE_TOKEN,
    RESIDUE_ATOM_TOKEN,
    UNK_TOKEN,

    # Variants
    HISTIDINE_VARIANTS,
    CYSTEINE_VARIANTS,

    # Backbone
    BACKBONE_ATOMS,
    BACKBONE_ATOMS_WITH_CB,

    # Standard atoms per residue (for standardization)
    STANDARD_ATOMS,
    STANDARD_ATOMS_PTM,

    # Residue name normalization
    RESIDUE_NAME_MAPPING,
    PTM_RESIDUES,
    NUCLEIC_ACID_RESIDUES,
    METAL_RESIDUES,
)

# =============================================================================
# SMARTS Pattern Constants
# =============================================================================
from .smarts_patterns import (
    # Primary pharmacophore patterns (recommended)
    PHARMACOPHORE_SMARTS,
    PHARMACOPHORE_CATEGORIES,

    # Detailed functional group patterns
    FUNCTIONAL_GROUP_SMARTS,

    # Aromatic ring patterns
    AROMATIC_RING_SMARTS,

    # Protein-specific patterns
    RESIDUE_SMARTS,
    BACKBONE_SMARTS,

    # Rotatable bonds
    ROTATABLE_BOND_SMARTS,

    # Backward compatibility
    CHEMICAL_SMARTS,
)

# =============================================================================
# Interaction Constants
# =============================================================================
from .interactions import (
    # Interaction definitions
    INTERACTION_TYPES,
    INTERACTION_TYPE_IDX,
    NUM_INTERACTION_TYPES,
    IDEAL_DISTANCES,

    # Pharmacophore indices
    PHARMACOPHORE_IDX,
    NUM_PHARMACOPHORE_TYPES,

    # Compatibility
    INTERACTION_COMPATIBILITY,
    ANGLE_TYPE_ENCODING,
    STACKING_TYPES,

    # Distance cutoffs
    DEFAULT_DISTANCE_CUTOFF,
    POCKET_EXTRACTION_CUTOFF,
    CLOSE_CONTACT_CUTOFF,
)

# =============================================================================
# Physical Property Constants
# =============================================================================
from .physical_properties import (
    # Radii
    VDW_RADIUS,
    DEFAULT_VDW_RADIUS,
    COVALENT_RADIUS,
    DEFAULT_COVALENT_RADIUS,

    # Energies
    IONIZATION_ENERGY,
    DEFAULT_IONIZATION_ENERGY,

    # Other properties
    POLARIZABILITY,
    DEFAULT_POLARIZABILITY,
    VALENCE_ELECTRONS,
    DEFAULT_VALENCE_ELECTRONS,
    ATOMIC_MASS,

    # Bond lengths
    TYPICAL_BOND_LENGTHS,

    # Normalization
    NORM_CONSTANTS,
)

# =============================================================================
# Module-level exports
# =============================================================================
__all__ = [
    # Elements
    'ATOM_TYPES',
    'HEAVY_ELEMENT_TYPES',
    'NUM_HEAVY_ELEMENT_TYPES',
    'PROTEIN_ELEMENT_TYPES',
    'ATOM_NAME_TO_ELEMENT',
    'SIMPLIFIED_ELEMENT_TYPES',
    'NUM_SIMPLIFIED_ELEMENT_TYPES',
    'METAL_ELEMENTS',
    'PERIODIC_TABLE',
    'PERIODS',
    'GROUPS',
    'ELECTRONEGATIVITY',
    'DEFAULT_ELECTRONEGATIVITY',
    'HYBRIDIZATION_TYPES',
    'NUM_HYBRIDIZATION_TYPES',
    'BOND_TYPES',
    'BOND_STEREOS',
    'BOND_DIRS',
    'DEGREES',
    'HEAVY_DEGREES',
    'VALENCES',
    'TOTAL_HS',

    # Amino acids
    'AMINO_ACID_3TO1',
    'AMINO_ACID_1TO3',
    'AMINO_ACID_1_TO_INT',
    'AMINO_ACID_3_TO_INT',
    'AMINO_ACID_LETTERS',
    'RESIDUE_TYPES',
    'NUM_RESIDUE_TYPES',
    'MAX_ATOMS_PER_RESIDUE',
    'RESIDUE_TOKEN',
    'RESIDUE_ATOM_TOKEN',
    'UNK_TOKEN',
    'HISTIDINE_VARIANTS',
    'CYSTEINE_VARIANTS',
    'BACKBONE_ATOMS',
    'BACKBONE_ATOMS_WITH_CB',
    'STANDARD_ATOMS',
    'STANDARD_ATOMS_PTM',
    'RESIDUE_NAME_MAPPING',
    'PTM_RESIDUES',
    'NUCLEIC_ACID_RESIDUES',
    'METAL_RESIDUES',

    # SMARTS patterns
    'PHARMACOPHORE_SMARTS',
    'PHARMACOPHORE_CATEGORIES',
    'FUNCTIONAL_GROUP_SMARTS',
    'AROMATIC_RING_SMARTS',
    'RESIDUE_SMARTS',
    'BACKBONE_SMARTS',
    'ROTATABLE_BOND_SMARTS',
    'CHEMICAL_SMARTS',

    # Interactions
    'INTERACTION_TYPES',
    'INTERACTION_TYPE_IDX',
    'NUM_INTERACTION_TYPES',
    'IDEAL_DISTANCES',
    'PHARMACOPHORE_IDX',
    'NUM_PHARMACOPHORE_TYPES',
    'INTERACTION_COMPATIBILITY',
    'ANGLE_TYPE_ENCODING',
    'STACKING_TYPES',
    'DEFAULT_DISTANCE_CUTOFF',
    'POCKET_EXTRACTION_CUTOFF',
    'CLOSE_CONTACT_CUTOFF',

    # Physical properties
    'VDW_RADIUS',
    'DEFAULT_VDW_RADIUS',
    'COVALENT_RADIUS',
    'DEFAULT_COVALENT_RADIUS',
    'IONIZATION_ENERGY',
    'DEFAULT_IONIZATION_ENERGY',
    'POLARIZABILITY',
    'DEFAULT_POLARIZABILITY',
    'VALENCE_ELECTRONS',
    'DEFAULT_VALENCE_ELECTRONS',
    'ATOMIC_MASS',
    'TYPICAL_BOND_LENGTHS',
    'NORM_CONSTANTS',
]
