"""
Physical Properties Constants.

Contains atomic physical properties including radii, ionization energies,
polarizabilities, bond lengths, and normalization constants.
"""

# =============================================================================
# Van der Waals Radii (Angstrom) - Bondi radii
# =============================================================================

VDW_RADIUS = {
    1: 1.20,    # H
    5: 1.92,    # B
    6: 1.70,    # C
    7: 1.55,    # N
    8: 1.52,    # O
    9: 1.47,    # F
    11: 2.27,   # Na
    12: 1.73,   # Mg
    14: 2.10,   # Si
    15: 1.80,   # P
    16: 1.80,   # S
    17: 1.75,   # Cl
    19: 2.75,   # K
    20: 2.31,   # Ca
    26: 2.04,   # Fe
    29: 1.40,   # Cu
    30: 1.39,   # Zn
    33: 1.85,   # As
    34: 1.90,   # Se
    35: 1.85,   # Br
    50: 2.17,   # Sn
    51: 2.06,   # Sb
    52: 2.06,   # Te
    53: 1.98,   # I
}

DEFAULT_VDW_RADIUS = 1.70

# =============================================================================
# Covalent Radii (Angstrom)
# =============================================================================

COVALENT_RADIUS = {
    1: 0.31,    # H
    5: 0.84,    # B
    6: 0.76,    # C
    7: 0.71,    # N
    8: 0.66,    # O
    9: 0.57,    # F
    11: 1.66,   # Na
    12: 1.41,   # Mg
    14: 1.11,   # Si
    15: 1.07,   # P
    16: 1.05,   # S
    17: 1.02,   # Cl
    19: 2.03,   # K
    20: 1.76,   # Ca
    26: 1.32,   # Fe
    29: 1.32,   # Cu
    30: 1.22,   # Zn
    33: 1.19,   # As
    34: 1.20,   # Se
    35: 1.20,   # Br
    50: 1.39,   # Sn
    51: 1.39,   # Sb
    52: 1.38,   # Te
    53: 1.39,   # I
}

DEFAULT_COVALENT_RADIUS = 0.76

# =============================================================================
# First Ionization Energy (eV)
# =============================================================================

IONIZATION_ENERGY = {
    1: 13.60,   # H
    5: 8.30,    # B
    6: 11.26,   # C
    7: 14.53,   # N
    8: 13.62,   # O
    9: 17.42,   # F
    11: 5.14,   # Na
    12: 7.65,   # Mg
    14: 8.15,   # Si
    15: 10.49,  # P
    16: 10.36,  # S
    17: 12.97,  # Cl
    19: 4.34,   # K
    20: 6.11,   # Ca
    26: 7.90,   # Fe
    29: 7.73,   # Cu
    30: 9.39,   # Zn
    33: 9.79,   # As
    34: 9.75,   # Se
    35: 11.81,  # Br
    50: 7.34,   # Sn
    51: 8.61,   # Sb
    52: 9.01,   # Te
    53: 10.45,  # I
}

DEFAULT_IONIZATION_ENERGY = 10.0

# =============================================================================
# Atomic Polarizability (Angstrom^3)
# =============================================================================

POLARIZABILITY = {
    1: 0.67,    # H
    5: 3.03,    # B
    6: 1.76,    # C
    7: 1.10,    # N
    8: 0.80,    # O
    9: 0.56,    # F
    11: 24.11,  # Na
    12: 10.60,  # Mg
    14: 5.38,   # Si
    15: 3.63,   # P
    16: 2.90,   # S
    17: 2.18,   # Cl
    19: 43.40,  # K
    20: 22.80,  # Ca
    26: 8.40,   # Fe
    29: 6.20,   # Cu
    30: 5.75,   # Zn
    33: 4.31,   # As
    34: 3.77,   # Se
    35: 3.05,   # Br
    50: 7.70,   # Sn
    51: 6.60,   # Sb
    52: 5.50,   # Te
    53: 5.35,   # I
}

DEFAULT_POLARIZABILITY = 1.76

# =============================================================================
# Valence Electrons (for lone pair calculation)
# =============================================================================

VALENCE_ELECTRONS = {
    1: 1,    # H
    5: 3,    # B
    6: 4,    # C
    7: 5,    # N
    8: 6,    # O
    9: 7,    # F
    11: 1,   # Na
    12: 2,   # Mg
    14: 4,   # Si
    15: 5,   # P
    16: 6,   # S
    17: 7,   # Cl
    19: 1,   # K
    20: 2,   # Ca
    26: 8,   # Fe
    29: 11,  # Cu
    30: 12,  # Zn
    33: 5,   # As
    34: 6,   # Se
    35: 7,   # Br
    50: 4,   # Sn
    51: 5,   # Sb
    52: 6,   # Te
    53: 7,   # I
}

DEFAULT_VALENCE_ELECTRONS = 4

# =============================================================================
# Typical Bond Lengths (Angstrom) - for normalization reference
# =============================================================================

# Bond type -> typical length range (min, max)
TYPICAL_BOND_LENGTHS = {
    'C-C': (1.20, 1.54),    # triple to single
    'C-N': (1.16, 1.47),
    'C-O': (1.13, 1.43),
    'C-S': (1.55, 1.82),
    'C-H': (1.06, 1.12),
    'N-H': (1.00, 1.04),
    'O-H': (0.94, 0.98),
    'N-N': (1.10, 1.45),
    'N-O': (1.15, 1.40),
    'O-O': (1.21, 1.48),
    'C-F': (1.27, 1.35),
    'C-Cl': (1.60, 1.79),
    'C-Br': (1.79, 1.97),
    'C-I': (1.99, 2.16),
    'default': (1.0, 2.5),
}

# =============================================================================
# Normalization Constants (for feature scaling)
# =============================================================================

NORM_CONSTANTS = {
    # Node feature normalization
    'atomic_mass': 200.0,
    'vdw_radius_min': 1.0,
    'vdw_radius_range': 2.0,
    'covalent_radius': 2.0,
    'ionization_energy_min': 4.0,
    'ionization_energy_range': 14.0,
    'polarizability_log_scale': 4.0,
    'lone_pairs': 3.0,
    'neighbor_en_sum': 16.0,
    'neighbor_en_diff': 3.2,
    'neighbor_mass_sum': 600.0,
    'neighbor_charge_shift': 4,
    'neighbor_charge_range': 8.0,
    'eccentricity': 20.0,
    'dist_to_special': 10.0,
    'logp_shift': 2.0,
    'logp_range': 4.0,
    'mr_max': 10.0,
    'tpsa_max': 30.0,
    'asa_max': 20.0,

    # Edge feature normalization
    'bond_length_min': 0.9,
    'bond_length_range': 1.6,
    'en_diff_max': 3.2,
    'mass_diff_max': 100.0,
    'mass_sum_max': 250.0,
    'charge_diff_max': 4.0,
    'path_length_max': 20.0,

    # PLI specific normalization
    'distance_cutoff': 4.5,
    'angle_max': 180.0,
    'degree_max': 4.0,
    'formal_charge_shift': 2,
    'formal_charge_range': 4.0,
}

# =============================================================================
# Atomic Mass (for reference)
# =============================================================================

ATOMIC_MASS = {
    1: 1.008,    # H
    6: 12.011,   # C
    7: 14.007,   # N
    8: 15.999,   # O
    9: 18.998,   # F
    15: 30.974,  # P
    16: 32.065,  # S
    17: 35.453,  # Cl
    35: 79.904,  # Br
    53: 126.90,  # I
}
