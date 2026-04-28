"""
Nucleic Acid Constants.

Contains nucleotide mappings, tokens, backbone atoms, base atoms, and
physicochemical properties for DNA and RNA featurization.
"""

# =============================================================================
# Nucleotide Mappings
# =============================================================================

# 3-letter (PDB residue name) to 1-letter code
NUCLEOTIDE_3TO1 = {
    'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C',  # DNA
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',        # RNA
}

# =============================================================================
# Nucleotide Token Mapping
# =============================================================================

NUCLEOTIDE_TYPES = ['DA', 'DT', 'DG', 'DC', 'A', 'U', 'G', 'C', 'UNK_NT']
NUM_NUCLEOTIDE_TYPES = len(NUCLEOTIDE_TYPES)
NUCLEOTIDE_TOKEN = {nt: i for i, nt in enumerate(NUCLEOTIDE_TYPES)}

# =============================================================================
# Backbone Atoms
# =============================================================================

# Standard phosphodiester backbone atoms (shared by DNA and RNA)
DNA_BACKBONE_ATOMS = [
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
]

# RNA has an additional 2'-OH group
RNA_BACKBONE_ATOMS = DNA_BACKBONE_ATOMS + ["O2'"]

# Union of all backbone atoms across DNA and RNA
NUCLEOTIDE_BACKBONE_SET = frozenset(
    atom for atoms in [DNA_BACKBONE_ATOMS, ["O2'"]] for atom in atoms
)

# =============================================================================
# Base Atoms per Nucleotide (ring + substituent heavy atoms, no backbone)
# =============================================================================

BASE_ATOMS: dict[str, list[str]] = {
    # Adenine (purine): 9-membered bicycle
    'DA': ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
    # Thymine (pyrimidine): has 5-methyl (C5M) instead of uracil C5-H
    'DT': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C5M', 'C6'],
    # Guanine (purine)
    'DG': ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
    # Cytosine (pyrimidine)
    'DC': ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
    # RNA adenine — same base as DA
    'A': ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
    # Uracil (pyrimidine): like thymine but no 5-methyl
    'U': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'],
    # RNA guanine — same base as DG
    'G': ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
    # RNA cytosine — same base as DC
    'C': ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
}

# =============================================================================
# Standard Atoms per Nucleotide (backbone + base)
# =============================================================================

def _standard_atoms(nt: str) -> list[str]:
    backbone = RNA_BACKBONE_ATOMS if nt in ('A', 'U', 'G', 'C') else DNA_BACKBONE_ATOMS
    return backbone + BASE_ATOMS[nt]


STANDARD_NUCLEOTIDE_ATOMS: dict[str, list[str]] = {
    nt: _standard_atoms(nt)
    for nt in BASE_ATOMS
}

# =============================================================================
# Max SASA per Nucleotide (Å², approximate values from literature)
# =============================================================================

NUCLEOTIDE_MAX_SASA: dict[str, float] = {
    'DA': 400.0,
    'DT': 380.0,
    'DG': 420.0,
    'DC': 350.0,
    'A':  400.0,
    'U':  340.0,
    'G':  420.0,
    'C':  350.0,
    'UNK_NT': 400.0,
}

# =============================================================================
# Watson-Crick Base Pairing
# =============================================================================

# Canonical Watson-Crick pairs (using 1-letter codes)
# BASE_PAIR_PATTERNS is an alias for WC_BASE_PAIRS
WC_BASE_PAIRS: dict[str, str] = {
    'A': 'T',  # DNA A-T
    'T': 'A',  # DNA T-A
    'G': 'C',  # G-C
    'C': 'G',  # C-G
    'U': 'A',  # RNA U-A
}

# Alias used by some downstream modules
BASE_PAIR_PATTERNS = WC_BASE_PAIRS

# =============================================================================
# Purine / Pyrimidine Classification
# =============================================================================

PURINES = frozenset({'DA', 'DG', 'A', 'G'})
PYRIMIDINES = frozenset({'DT', 'DC', 'U', 'C'})

# Per-nucleotide boolean classification dicts (True = is that class)
IS_PURINE: dict[str, bool] = {nt: (nt in PURINES) for nt in NUCLEOTIDE_TYPES}
IS_PYRIMIDINE: dict[str, bool] = {nt: (nt in PYRIMIDINES) for nt in NUCLEOTIDE_TYPES}

# =============================================================================
# DNA vs RNA Residue Sets
# =============================================================================

DNA_RESIDUES = frozenset({'DA', 'DT', 'DG', 'DC'})
RNA_RESIDUES = frozenset({'A', 'U', 'G', 'C'})

# =============================================================================
# Nucleotide Physicochemical Properties
# =============================================================================

# (molecular_weight_base [Da], n_hbond_donors, n_hbond_acceptors, is_purine)
# Molecular weights are for the nucleobase only (not nucleotide with sugar+phosphate).
NUCLEOTIDE_PROPERTIES: dict[str, tuple[float, int, int, float]] = {
    'DA': (313.2, 1, 3, 1.0),  # dAMP
    'DT': (304.2, 1, 4, 0.0),  # dTMP
    'DG': (329.2, 2, 4, 1.0),  # dGMP
    'DC': (289.2, 1, 3, 0.0),  # dCMP
    'A':  (329.2, 1, 3, 1.0),  # AMP
    'U':  (324.2, 1, 4, 0.0),  # UMP
    'G':  (345.2, 2, 4, 1.0),  # GMP
    'C':  (305.2, 1, 3, 0.0),  # CMP
    'UNK_NT': (320.0, 1, 3, 0.5),
}
