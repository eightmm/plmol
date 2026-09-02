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
    ELEMENT_SYMBOL_TO_ATOMIC_NUMBER,
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
    RESIDUE_TYPE_INDEX,
    OTHER_RESIDUE_INDEX,
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

    # Atom-level feature constants
    RESIDUE_MAX_SASA,
    FORMAL_CHARGE_MAP,
    HBOND_DONOR_ATOMS,
    HBOND_ACCEPTOR_ATOMS,
    HBOND_DONOR_ATOMS_BY_RESIDUE,
    HBOND_ACCEPTOR_ATOMS_BY_RESIDUE,
    BACKBONE_ATOM_SET,

    # Aromatic and ionizable atoms
    AROMATIC_RING_ATOMS,
    POS_IONIZABLE_ATOMS,
    HIS_POS_IONIZABLE_ATOMS,
    HIS_PARTIAL_CHARGE,
    NEG_IONIZABLE_ATOMS,

    # Residue physicochemical properties
    RESIDUE_PROPERTIES,
    NUM_RESIDUE_PROPERTIES,
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

    # Pi-stacking angle thresholds
    PI_STACK_PARALLEL_MAX,
    PI_STACK_PARALLEL_MIN_ANTI,
    PI_STACK_PERP_MIN,
    PI_STACK_PERP_MAX,
    CATION_PI_MAX_OFFSET_ANGLE,

    # Interaction strength parameters
    INTERACTION_STRENGTH_SIGMA,
    IDEAL_DISTANCE_FALLBACK,

    # Cross-contact density parameters
    CROSS_CONTACT_DENSITY_CUTOFF,
    CROSS_CONTACT_DENSITY_NORM,

    # Atom feature normalization
    FORMAL_CHARGE_OFFSET,
    FORMAL_CHARGE_SCALE,
    DEGREE_SCALE,
    NUM_HS_SCALE,

    # Metal coordination geometry angle thresholds
    SQUARE_PLANAR_LINEAR_ANGLE,
    TRIGONAL_BIPYRAMIDAL_LINEAR_ANGLE,

    # Metal coordination
    METAL_COORDINATION_CUTOFF,
    COMMON_COORDINATION_GEOMETRIES,
    METAL_PREFERRED_DONORS,
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

    # Surface/voxel feature constants
    ATOM_TYPE_MAP,
    ATOM_TYPE_LABELS,
    NUM_ATOM_TYPES,
    ATOMIC_MOLAR_REFRACTIVITY,
    KD_SCALE,
    CHARGED_RESIDUES,
)

# =============================================================================
# Nucleic Acid Constants
# =============================================================================
from .nucleic_acids import (
    # Nucleotide mappings
    NUCLEOTIDE_3TO1,

    # Tokens
    NUCLEOTIDE_TYPES,
    NUM_NUCLEOTIDE_TYPES,
    NUCLEOTIDE_TOKEN,

    # Backbone atoms
    DNA_BACKBONE_ATOMS,
    RNA_BACKBONE_ATOMS,
    NUCLEOTIDE_BACKBONE_SET,

    # Base atoms
    BASE_ATOMS,

    # Standard atoms per nucleotide
    STANDARD_NUCLEOTIDE_ATOMS,

    # SASA
    NUCLEOTIDE_MAX_SASA,

    # Base pairing
    WC_BASE_PAIRS,
    BASE_PAIR_PATTERNS,

    # Classification
    PURINES,
    PYRIMIDINES,
    IS_PURINE,
    IS_PYRIMIDINE,
    DNA_RESIDUES,
    RNA_RESIDUES,

    # Properties
    NUCLEOTIDE_PROPERTIES,
)

# =============================================================================
# Runtime/Default Constants
# =============================================================================
from .runtime import (
    IO_SUPPORTED_LIGAND_EXTENSIONS,
    PTM_HANDLING_MODES,
    DEFAULT_ATOM_GRAPH_DISTANCE_CUTOFF,
    DEFAULT_RESIDUE_GRAPH_DISTANCE_CUTOFF,
    DEFAULT_BACKBONE_KNN_NEIGHBORS,
    SURFACE_DEFAULT_CURVATURE_SCALES,
    SURFACE_DEFAULT_KNN_ATOMS,
    SURFACE_DEFAULT_POINTS_PER_ATOM,
    SURFACE_DEFAULT_PROBE_RADIUS,
    SURFACE_MIN_POINTS_PER_ATOM,
    SURFACE_MAX_POINTS_RATIO,
    VOXEL_DEFAULT_RESOLUTION,
    VOXEL_DEFAULT_BOX_SIZE,
    VOXEL_DEFAULT_PADDING,
    VOXEL_DEFAULT_SIGMA_SCALE,
    VOXEL_DEFAULT_CUTOFF_SIGMA,
    POCKET_MAX_ATOMS_PER_RESIDUE,
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
    'ELEMENT_SYMBOL_TO_ATOMIC_NUMBER',

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
    'RESIDUE_MAX_SASA',
    'FORMAL_CHARGE_MAP',
    'HBOND_DONOR_ATOMS',
    'HBOND_ACCEPTOR_ATOMS',
    'HBOND_DONOR_ATOMS_BY_RESIDUE',
    'HBOND_ACCEPTOR_ATOMS_BY_RESIDUE',
    'BACKBONE_ATOM_SET',
    'AROMATIC_RING_ATOMS',
    'POS_IONIZABLE_ATOMS',
    'HIS_POS_IONIZABLE_ATOMS',
    'HIS_PARTIAL_CHARGE',
    'NEG_IONIZABLE_ATOMS',
    'RESIDUE_PROPERTIES',
    'NUM_RESIDUE_PROPERTIES',

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
    'PI_STACK_PARALLEL_MAX',
    'PI_STACK_PARALLEL_MIN_ANTI',
    'PI_STACK_PERP_MIN',
    'PI_STACK_PERP_MAX',
    'CATION_PI_MAX_OFFSET_ANGLE',
    'INTERACTION_STRENGTH_SIGMA',
    'IDEAL_DISTANCE_FALLBACK',
    'CROSS_CONTACT_DENSITY_CUTOFF',
    'CROSS_CONTACT_DENSITY_NORM',
    'FORMAL_CHARGE_OFFSET',
    'FORMAL_CHARGE_SCALE',
    'DEGREE_SCALE',
    'NUM_HS_SCALE',
    'SQUARE_PLANAR_LINEAR_ANGLE',
    'TRIGONAL_BIPYRAMIDAL_LINEAR_ANGLE',

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
    'ATOM_TYPE_MAP',
    'ATOM_TYPE_LABELS',
    'NUM_ATOM_TYPES',
    'ATOMIC_MOLAR_REFRACTIVITY',
    'KD_SCALE',
    'CHARGED_RESIDUES',

    # Nucleic acids
    'NUCLEOTIDE_3TO1',
    'NUCLEOTIDE_TYPES',
    'NUM_NUCLEOTIDE_TYPES',
    'NUCLEOTIDE_TOKEN',
    'DNA_BACKBONE_ATOMS',
    'RNA_BACKBONE_ATOMS',
    'NUCLEOTIDE_BACKBONE_SET',
    'BASE_ATOMS',
    'STANDARD_NUCLEOTIDE_ATOMS',
    'NUCLEOTIDE_MAX_SASA',
    'WC_BASE_PAIRS',
    'BASE_PAIR_PATTERNS',
    'PURINES',
    'PYRIMIDINES',
    'IS_PURINE',
    'IS_PYRIMIDINE',
    'DNA_RESIDUES',
    'RNA_RESIDUES',
    'NUCLEOTIDE_PROPERTIES',

    # Runtime defaults
    'IO_SUPPORTED_LIGAND_EXTENSIONS',
    'PTM_HANDLING_MODES',
    'DEFAULT_ATOM_GRAPH_DISTANCE_CUTOFF',
    'DEFAULT_RESIDUE_GRAPH_DISTANCE_CUTOFF',
    'SURFACE_DEFAULT_CURVATURE_SCALES',
    'SURFACE_DEFAULT_KNN_ATOMS',
    'SURFACE_DEFAULT_POINTS_PER_ATOM',
    'SURFACE_DEFAULT_PROBE_RADIUS',
    'SURFACE_MIN_POINTS_PER_ATOM',
    'SURFACE_MAX_POINTS_RATIO',
    'VOXEL_DEFAULT_RESOLUTION',
    'VOXEL_DEFAULT_BOX_SIZE',
    'VOXEL_DEFAULT_PADDING',
    'VOXEL_DEFAULT_SIGMA_SCALE',
    'VOXEL_DEFAULT_CUTOFF_SIGMA',
    'DEFAULT_BACKBONE_KNN_NEIGHBORS',
    'POCKET_MAX_ATOMS_PER_RESIDUE',
]
