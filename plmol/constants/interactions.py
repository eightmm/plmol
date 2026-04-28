"""
Interaction Type Definitions and Constants.

Contains definitions for protein-ligand interaction types including
distance cutoffs, angle requirements, and feature indices.
"""

# =============================================================================
# Interaction Type Definitions
# =============================================================================

INTERACTION_TYPES = {
    'hydrogen_bond': {
        'description': 'Hydrogen bond (donor-acceptor pair)',
        'distance_cutoff': 3.5,   # Angstrom
        'angle_cutoff': 120.0,    # degrees (D-H...A angle minimum)
    },
    'salt_bridge': {
        'description': 'Ionic interaction between charged groups',
        'distance_cutoff': 4.0,
    },
    'pi_stacking': {
        'description': 'Aromatic ring stacking (face-to-face or edge-to-face)',
        'distance_cutoff': 5.5,
        'angle_parallel': 30.0,       # degrees from parallel for face-to-face
        'angle_perpendicular': 60.0,  # degrees for T-shaped
    },
    'cation_pi': {
        'description': 'Cation-aromatic interaction',
        'distance_cutoff': 6.0,
    },
    'hydrophobic': {
        'description': 'Hydrophobic/van der Waals contact',
        'distance_cutoff': 4.5,
    },
    'halogen_bond': {
        'description': 'Halogen bond (sigma-hole interaction)',
        'distance_cutoff': 3.5,
        'angle_cutoff': 140.0,    # C-X...A angle minimum
    },
    'metal_coordination': {
        'description': 'Metal ion coordination',
        'distance_cutoff': 2.8,
    },
}

# =============================================================================
# Interaction Type Indices for One-Hot Encoding
# =============================================================================

INTERACTION_TYPE_IDX = {
    'hydrogen_bond': 0,
    'salt_bridge': 1,
    'pi_stacking': 2,
    'cation_pi': 3,
    'hydrophobic': 4,
    'halogen_bond': 5,
    'metal_coordination': 6,
}
NUM_INTERACTION_TYPES = 7

# =============================================================================
# Ideal Distances for Interaction Quality Scoring
# =============================================================================

IDEAL_DISTANCES = {
    'hydrogen_bond': 2.8,
    'salt_bridge': 3.0,
    'pi_stacking': 3.8,
    'cation_pi': 4.5,
    'hydrophobic': 3.8,
    'halogen_bond': 3.0,
    'metal_coordination': 2.2,
}

# =============================================================================
# Pharmacophore Type Indices
# =============================================================================

PHARMACOPHORE_IDX = {
    'hbond_donor': 0,
    'hbond_acceptor': 1,
    'positive_charge': 2,
    'negative_charge': 3,
    'aromatic': 4,
    'hydrophobic': 5,
    'halogen_bond': 6,
    'metal_coord': 7,
    'other': 8,
}
NUM_PHARMACOPHORE_TYPES = 9

# =============================================================================
# Interaction Compatibility Matrix
# =============================================================================

# Which pharmacophore types can interact with which
# (protein_type, ligand_type) -> interaction_type
INTERACTION_COMPATIBILITY = {
    # Hydrogen bonds
    ('hbond_donor', 'hbond_acceptor'): 'hydrogen_bond',
    ('hbond_acceptor', 'hbond_donor'): 'hydrogen_bond',

    # Salt bridges
    ('positive_charge', 'negative_charge'): 'salt_bridge',
    ('negative_charge', 'positive_charge'): 'salt_bridge',

    # Pi-stacking
    ('aromatic', 'aromatic'): 'pi_stacking',

    # Cation-pi
    ('positive_charge', 'aromatic'): 'cation_pi',
    ('aromatic', 'positive_charge'): 'cation_pi',

    # Hydrophobic
    ('hydrophobic', 'hydrophobic'): 'hydrophobic',
    ('aromatic', 'hydrophobic'): 'hydrophobic',
    ('hydrophobic', 'aromatic'): 'hydrophobic',

    # Halogen bonds
    ('hbond_acceptor', 'halogen_bond'): 'halogen_bond',

    # Metal coordination
    ('metal_coord', 'metal_coord'): 'metal_coordination',
}

# =============================================================================
# Angle Type Encoding
# =============================================================================

# For edge feature encoding: what type of angle is stored
ANGLE_TYPE_ENCODING = {
    'none': 0.0,
    'ring': 0.33,      # Pi-stacking ring angle
    'dha': 0.67,       # D-H-A angle for H-bonds
    'cxa': 1.0,        # C-X-A angle for halogen bonds
}

# =============================================================================
# Stacking Types
# =============================================================================

STACKING_TYPES = {
    'parallel': 'face-to-face stacking (rings parallel)',
    'T-shaped': 'edge-to-face stacking (rings perpendicular)',
    'offset': 'offset parallel stacking',
}

# =============================================================================
# Distance Cutoff Defaults
# =============================================================================

DEFAULT_DISTANCE_CUTOFF = 4.5  # General PLI distance cutoff
POCKET_EXTRACTION_CUTOFF = 6.0  # Residue-wise pocket extraction
CLOSE_CONTACT_CUTOFF = 2.5      # Very close contacts (potential clashes)

# =============================================================================
# Pi-Stacking Angle Thresholds
# =============================================================================

# Parallel (face-to-face) stacking: angle < PI_STACK_PARALLEL_MAX or > PI_STACK_PARALLEL_MIN_ANTI
PI_STACK_PARALLEL_MAX = 30.0       # degrees — ring normals nearly parallel
PI_STACK_PARALLEL_MIN_ANTI = 150.0 # degrees — antiparallel normals also count as parallel
# Perpendicular (T-shaped) stacking: PI_STACK_PERP_MIN < angle < PI_STACK_PERP_MAX
PI_STACK_PERP_MIN = 60.0           # degrees
PI_STACK_PERP_MAX = 120.0          # degrees

# Cation-pi: cation must lie within this angle of the ring normal (projection check)
CATION_PI_MAX_OFFSET_ANGLE = 30.0  # degrees

# =============================================================================
# Interaction Strength (Gaussian decay) Parameters
# =============================================================================

INTERACTION_STRENGTH_SIGMA = 0.5   # Angstroms — width of Gaussian decay
IDEAL_DISTANCE_FALLBACK = 3.0      # Angstroms — default ideal distance if type unknown

# =============================================================================
# Cross-Contact Density Parameters (edge features)
# =============================================================================

CROSS_CONTACT_DENSITY_CUTOFF = 4.0  # Angstroms — radius for counting cross-contacts
CROSS_CONTACT_DENSITY_NORM = 10.0   # normalizer: contacts / NORM -> [0,1]

# =============================================================================
# Atom Feature Normalization
# =============================================================================

FORMAL_CHARGE_OFFSET = 2    # shift before dividing: (charge + offset) / scale
FORMAL_CHARGE_SCALE = 4.0   # divisor for formal charge normalization
DEGREE_SCALE = 4.0          # divisor for heavy-atom degree normalization
NUM_HS_SCALE = 4.0          # divisor for attached-hydrogen count normalization

# =============================================================================
# Metal Coordination Geometry Angle Thresholds
# =============================================================================

# Angles above these thresholds indicate near-linear (trans) bonds
SQUARE_PLANAR_LINEAR_ANGLE = 150.0       # degrees — used to detect square planar (4-coord)
TRIGONAL_BIPYRAMIDAL_LINEAR_ANGLE = 160.0  # degrees — used to detect trigonal bipyramidal (5-coord)

# =============================================================================
# Metal Coordination Constants
# =============================================================================

METAL_COORDINATION_CUTOFF = 2.8  # Angstroms

COMMON_COORDINATION_GEOMETRIES = [
    "linear",
    "trigonal_planar",
    "tetrahedral",
    "square_planar",
    "trigonal_bipyramidal",
    "square_pyramidal",
    "octahedral",
]

METAL_PREFERRED_DONORS = {
    'ZN': {'N', 'O', 'S'},
    'MG': {'O'},
    'CA': {'O'},
    'FE': {'N', 'O', 'S'},
    'CU': {'N', 'O', 'S'},
    'MN': {'N', 'O'},
    'CO': {'N', 'O', 'S'},
    'NI': {'N', 'O', 'S'},
    'NA': {'O'},
    'K':  {'O'},
}
