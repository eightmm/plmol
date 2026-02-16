"""
Element and Periodic Table Constants.

Contains element type mappings, periodic table data, and physical properties
for atoms commonly found in protein-ligand complexes.
"""

from rdkit import Chem

# =============================================================================
# Element Type Mappings
# =============================================================================

# Atom types for molecule featurization (ligands)
# Keep common bioorganic atoms first, then biorelevant metals, then UNK fallback.
ATOM_TYPES = [
    'H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Se',
    'Zn', 'Mg', 'Ca', 'Fe', 'Mn', 'Cu', 'Co', 'Ni', 'Na', 'K',
    'UNK',
]

# Heavy atom element types for PLI (no hydrogen)
HEAVY_ELEMENT_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
NUM_HEAVY_ELEMENT_TYPES = len(HEAVY_ELEMENT_TYPES)

# Protein element type mapping (includes metals)
PROTEIN_ELEMENT_TYPES = {
    'H': 0,    # Hydrogen (if kept)
    'C': 1,    # Carbon
    'N': 2,    # Nitrogen
    'O': 3,    # Oxygen
    'S': 4,    # Sulfur
    'P': 5,    # Phosphorus
    'SE': 6,   # Selenium
    # Metals
    'CA': 7,   # Calcium
    'MG': 8,   # Magnesium
    'ZN': 9,   # Zinc
    'FE': 10,  # Iron
    'MN': 11,  # Manganese
    'CU': 12,  # Copper
    'CO': 13,  # Cobalt
    'NI': 14,  # Nickel
    'NA': 15,  # Sodium
    'K': 16,   # Potassium
    'METAL': 17,  # Generic metal
    'UNK': 18,    # Unknown
}

# Simplified element types for hierarchical models (8 classes)
# Groups metals together, focuses on biologically relevant atoms
SIMPLIFIED_ELEMENT_TYPES = {
    'C': 0,      # Carbon
    'N': 1,      # Nitrogen
    'O': 2,      # Oxygen
    'S': 3,      # Sulfur
    'P': 4,      # Phosphorus
    'SE': 5,     # Selenium
    'METAL': 6,  # All metals (CA, MG, ZN, FE, MN, CU, CO, NI, NA, K, etc.)
    'UNK': 7,    # Unknown
}
NUM_SIMPLIFIED_ELEMENT_TYPES = len(SIMPLIFIED_ELEMENT_TYPES)

# Metal elements for detection
METAL_ELEMENTS = {'CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'NA', 'K'}

# Atom name to element mapping for standard amino acids
ATOM_NAME_TO_ELEMENT = {
    'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O', 'CB': 'C',
    'CG': 'C', 'CG1': 'C', 'CG2': 'C',
    'CD': 'C', 'CD1': 'C', 'CD2': 'C',
    'CE': 'C', 'CE1': 'C', 'CE2': 'C', 'CE3': 'C',
    'CZ': 'C', 'CZ2': 'C', 'CZ3': 'C',
    'CH2': 'C',
    'ND1': 'N', 'ND2': 'N', 'NE': 'N', 'NE1': 'N', 'NE2': 'N',
    'NH1': 'N', 'NH2': 'N', 'NZ': 'N',
    'OD1': 'O', 'OD2': 'O', 'OE1': 'O', 'OE2': 'O',
    'OG': 'O', 'OG1': 'O', 'OH': 'O',
    'SG': 'S', 'SD': 'S',
    'P': 'P',
    'SE': 'SE',
}

# =============================================================================
# Periodic Table Data
# =============================================================================

# Element symbol -> (period, group)
PERIODIC_TABLE = {
    'H': (0, 0), 'He': (0, 17),
    'Li': (1, 0), 'Be': (1, 1), 'B': (1, 12), 'C': (1, 13),
    'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17),
    'Na': (2, 0), 'Mg': (2, 1), 'Al': (2, 12), 'Si': (2, 13),
    'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17),
    'K': (3, 0), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3),
    'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7),
    'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11),
    'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14), 'Se': (3, 15),
    'Br': (3, 16), 'Kr': (3, 17),
    'Rb': (4, 0), 'Sr': (4, 1), 'Y': (4, 2), 'Zr': (4, 3),
    'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6), 'Ru': (4, 7),
    'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11),
    'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14), 'Te': (4, 15),
    'I': (4, 16), 'Xe': (4, 17)
}

PERIODS = list(range(5))
GROUPS = list(range(18))

# =============================================================================
# Electronegativity (Pauling scale)
# =============================================================================

# (period, group) -> electronegativity
ELECTRONEGATIVITY = {
    (0, 0): 2.20,  # H
    (1, 0): 0.98, (1, 1): 1.57, (1, 12): 2.04, (1, 13): 2.55,
    (1, 14): 3.04, (1, 15): 3.44, (1, 16): 3.98,
    (2, 0): 0.93, (2, 1): 1.31, (2, 12): 1.61, (2, 13): 1.90,
    (2, 14): 2.19, (2, 15): 2.58, (2, 16): 3.16,
    (3, 0): 0.82, (3, 1): 1.00, (3, 2): 1.36, (3, 3): 1.54,
    (3, 4): 1.63, (3, 5): 1.66, (3, 6): 1.55, (3, 7): 1.83,
    (3, 8): 1.88, (3, 9): 1.91, (3, 10): 1.90, (3, 11): 1.65,
    (3, 12): 1.81, (3, 13): 2.01, (3, 14): 2.18, (3, 15): 2.55,
    (3, 16): 2.96, (3, 17): 3.00,
    (4, 0): 0.82, (4, 1): 0.95, (4, 2): 1.22, (4, 3): 1.33,
    (4, 4): 1.60, (4, 5): 2.16, (4, 6): 1.90, (4, 7): 2.20,
    (4, 8): 2.28, (4, 9): 2.20, (4, 10): 1.93, (4, 11): 1.69,
    (4, 12): 1.78, (4, 13): 1.96, (4, 14): 2.05, (4, 15): 2.10,
    (4, 16): 2.66, (4, 17): 2.60
}

DEFAULT_ELECTRONEGATIVITY = 2.5

# =============================================================================
# RDKit Hybridization Types
# =============================================================================

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED
]
NUM_HYBRIDIZATION_TYPES = len(HYBRIDIZATION_TYPES)

# =============================================================================
# Bond Types
# =============================================================================

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]

BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOZ
]

BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.BEGINWEDGE,
    Chem.rdchem.BondDir.BEGINDASH,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.ENDUPRIGHT,
]

# =============================================================================
# Degree/Valence Ranges
# =============================================================================

DEGREES = list(range(7))
HEAVY_DEGREES = list(range(7))
VALENCES = list(range(8))
TOTAL_HS = list(range(5))

# =============================================================================
# Element Symbol -> Atomic Number
# =============================================================================

ELEMENT_SYMBOL_TO_ATOMIC_NUMBER = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "I": 53,
}
