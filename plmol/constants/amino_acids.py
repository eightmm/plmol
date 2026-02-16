"""
Amino Acid and Residue Constants.

Contains amino acid mappings, residue tokens, and atom-level protein tokens
for featurization of protein structures.
"""

# =============================================================================
# Amino Acid Mappings
# =============================================================================

# Standard 20 amino acids: 3-letter to 1-letter
AMINO_ACID_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# Reverse mapping: 1-letter to 3-letter
AMINO_ACID_1TO3 = {v: k for k, v in AMINO_ACID_3TO1.items()}

# Integer encoding: 1-letter sorted alphabetically
AMINO_ACID_1_TO_INT = {k: i for i, k in enumerate(sorted(AMINO_ACID_1TO3.keys()))}

# Integer encoding: 3-letter based on 1-letter sorted order
AMINO_ACID_3_TO_INT = {AMINO_ACID_1TO3[k]: i for i, k in enumerate(sorted(AMINO_ACID_1TO3.keys()))}

# Add unknown residue
AMINO_ACID_1_TO_INT['X'] = 20
AMINO_ACID_3_TO_INT['UNK'] = 20
AMINO_ACID_3_TO_INT['XXX'] = 20

# List of standard 3-letter codes
AMINO_ACID_LETTERS = list(AMINO_ACID_3TO1.keys())

# =============================================================================
# Residue Token Mapping
# =============================================================================

# Residue type to integer token
RESIDUE_TYPES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'Other'
]
NUM_RESIDUE_TYPES = len(RESIDUE_TYPES)

# Maximum atoms per residue (for coordinate tensors)
# Standard amino acids have at most 14 heavy atoms (TRP), +1 for sidechain centroid = 15
MAX_ATOMS_PER_RESIDUE = 15

RESIDUE_TOKEN = {
    'ALA': 0,  'ARG': 1,  'ASN': 2,  'ASP': 3,  'CYS': 4,
    'GLN': 5,  'GLU': 6,  'GLY': 7,  'HIS': 8,  'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20, 'XXX': 20,
    'METAL': 21,
}

# =============================================================================
# Residue-Atom Token Mapping
# =============================================================================

# Combined (residue_name, atom_name) -> token mapping
RESIDUE_ATOM_TOKEN = {
    # ALA: N, CA, C, O, CB
    ('ALA', 'N'): 0, ('ALA', 'CA'): 1, ('ALA', 'C'): 2, ('ALA', 'O'): 3, ('ALA', 'CB'): 4,

    # ARG: N, CA, C, O, CB, CG, CD, NE, CZ, NH1, NH2
    ('ARG', 'N'): 5, ('ARG', 'CA'): 6, ('ARG', 'C'): 7, ('ARG', 'O'): 8, ('ARG', 'CB'): 9,
    ('ARG', 'CG'): 10, ('ARG', 'CD'): 11, ('ARG', 'NE'): 12, ('ARG', 'CZ'): 13,
    ('ARG', 'NH1'): 14, ('ARG', 'NH2'): 15,

    # ASN: N, CA, C, O, CB, CG, OD1, ND2
    ('ASN', 'N'): 16, ('ASN', 'CA'): 17, ('ASN', 'C'): 18, ('ASN', 'O'): 19,
    ('ASN', 'CB'): 20, ('ASN', 'CG'): 21, ('ASN', 'OD1'): 22, ('ASN', 'ND2'): 23,

    # ASP: N, CA, C, O, CB, CG, OD1, OD2
    ('ASP', 'N'): 24, ('ASP', 'CA'): 25, ('ASP', 'C'): 26, ('ASP', 'O'): 27,
    ('ASP', 'CB'): 28, ('ASP', 'CG'): 29, ('ASP', 'OD1'): 30, ('ASP', 'OD2'): 31,

    # CYS: N, CA, C, O, CB, SG
    ('CYS', 'N'): 32, ('CYS', 'CA'): 33, ('CYS', 'C'): 34, ('CYS', 'O'): 35,
    ('CYS', 'CB'): 36, ('CYS', 'SG'): 37,

    # GLN: N, CA, C, O, CB, CG, CD, OE1, NE2
    ('GLN', 'N'): 38, ('GLN', 'CA'): 39, ('GLN', 'C'): 40, ('GLN', 'O'): 41,
    ('GLN', 'CB'): 42, ('GLN', 'CG'): 43, ('GLN', 'CD'): 44, ('GLN', 'OE1'): 45,
    ('GLN', 'NE2'): 46,

    # GLU: N, CA, C, O, CB, CG, CD, OE1, OE2
    ('GLU', 'N'): 47, ('GLU', 'CA'): 48, ('GLU', 'C'): 49, ('GLU', 'O'): 50,
    ('GLU', 'CB'): 51, ('GLU', 'CG'): 52, ('GLU', 'CD'): 53, ('GLU', 'OE1'): 54,
    ('GLU', 'OE2'): 55,

    # GLY: N, CA, C, O
    ('GLY', 'N'): 56, ('GLY', 'CA'): 57, ('GLY', 'C'): 58, ('GLY', 'O'): 59,

    # HIS: N, CA, C, O, CB, CG, ND1, CD2, CE1, NE2
    ('HIS', 'N'): 60, ('HIS', 'CA'): 61, ('HIS', 'C'): 62, ('HIS', 'O'): 63,
    ('HIS', 'CB'): 64, ('HIS', 'CG'): 65, ('HIS', 'ND1'): 66, ('HIS', 'CD2'): 67,
    ('HIS', 'CE1'): 68, ('HIS', 'NE2'): 69,

    # ILE: N, CA, C, O, CB, CG1, CG2, CD1
    ('ILE', 'N'): 70, ('ILE', 'CA'): 71, ('ILE', 'C'): 72, ('ILE', 'O'): 73,
    ('ILE', 'CB'): 74, ('ILE', 'CG1'): 75, ('ILE', 'CG2'): 76, ('ILE', 'CD1'): 77,

    # LEU: N, CA, C, O, CB, CG, CD1, CD2
    ('LEU', 'N'): 78, ('LEU', 'CA'): 79, ('LEU', 'C'): 80, ('LEU', 'O'): 81,
    ('LEU', 'CB'): 82, ('LEU', 'CG'): 83, ('LEU', 'CD1'): 84, ('LEU', 'CD2'): 85,

    # LYS: N, CA, C, O, CB, CG, CD, CE, NZ
    ('LYS', 'N'): 86, ('LYS', 'CA'): 87, ('LYS', 'C'): 88, ('LYS', 'O'): 89,
    ('LYS', 'CB'): 90, ('LYS', 'CG'): 91, ('LYS', 'CD'): 92, ('LYS', 'CE'): 93,
    ('LYS', 'NZ'): 94,

    # MET: N, CA, C, O, CB, CG, SD, CE
    ('MET', 'N'): 95, ('MET', 'CA'): 96, ('MET', 'C'): 97, ('MET', 'O'): 98,
    ('MET', 'CB'): 99, ('MET', 'CG'): 100, ('MET', 'SD'): 101, ('MET', 'CE'): 102,

    # PHE: N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ
    ('PHE', 'N'): 103, ('PHE', 'CA'): 104, ('PHE', 'C'): 105, ('PHE', 'O'): 106,
    ('PHE', 'CB'): 107, ('PHE', 'CG'): 108, ('PHE', 'CD1'): 109, ('PHE', 'CD2'): 110,
    ('PHE', 'CE1'): 111, ('PHE', 'CE2'): 112, ('PHE', 'CZ'): 113,

    # PRO: N, CA, C, O, CB, CG, CD
    ('PRO', 'N'): 114, ('PRO', 'CA'): 115, ('PRO', 'C'): 116, ('PRO', 'O'): 117,
    ('PRO', 'CB'): 118, ('PRO', 'CG'): 119, ('PRO', 'CD'): 120,

    # SER: N, CA, C, O, CB, OG
    ('SER', 'N'): 121, ('SER', 'CA'): 122, ('SER', 'C'): 123, ('SER', 'O'): 124,
    ('SER', 'CB'): 125, ('SER', 'OG'): 126,

    # THR: N, CA, C, O, CB, OG1, CG2
    ('THR', 'N'): 127, ('THR', 'CA'): 128, ('THR', 'C'): 129, ('THR', 'O'): 130,
    ('THR', 'CB'): 131, ('THR', 'OG1'): 132, ('THR', 'CG2'): 133,

    # TRP: N, CA, C, O, CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2
    ('TRP', 'N'): 134, ('TRP', 'CA'): 135, ('TRP', 'C'): 136, ('TRP', 'O'): 137,
    ('TRP', 'CB'): 138, ('TRP', 'CG'): 139, ('TRP', 'CD1'): 140, ('TRP', 'CD2'): 141,
    ('TRP', 'NE1'): 142, ('TRP', 'CE2'): 143, ('TRP', 'CE3'): 144, ('TRP', 'CZ2'): 145,
    ('TRP', 'CZ3'): 146, ('TRP', 'CH2'): 147,

    # TYR: N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ, OH
    ('TYR', 'N'): 148, ('TYR', 'CA'): 149, ('TYR', 'C'): 150, ('TYR', 'O'): 151,
    ('TYR', 'CB'): 152, ('TYR', 'CG'): 153, ('TYR', 'CD1'): 154, ('TYR', 'CD2'): 155,
    ('TYR', 'CE1'): 156, ('TYR', 'CE2'): 157, ('TYR', 'CZ'): 158, ('TYR', 'OH'): 159,

    # VAL: N, CA, C, O, CB, CG1, CG2
    ('VAL', 'N'): 160, ('VAL', 'CA'): 161, ('VAL', 'C'): 162, ('VAL', 'O'): 163,
    ('VAL', 'CB'): 164, ('VAL', 'CG1'): 165, ('VAL', 'CG2'): 166,

    # UNK: N, CA, C, O, CB (unknown residue, backbone + CB only)
    ('UNK', 'N'): 167, ('UNK', 'CA'): 168, ('UNK', 'C'): 169, ('UNK', 'O'): 170, ('UNK', 'CB'): 171,
    ('XXX', 'N'): 167, ('XXX', 'CA'): 168, ('XXX', 'C'): 169, ('XXX', 'O'): 170, ('XXX', 'CB'): 171,

    # Metal ions (biologically important metals with distinct roles)
    ('METAL', 'CA'): 175,   # Calcium - signaling, structural
    ('METAL', 'MG'): 176,   # Magnesium - enzymatic cofactor, ATP binding
    ('METAL', 'ZN'): 177,   # Zinc - structural (zinc fingers), catalytic
    ('METAL', 'FE'): 178,   # Iron - electron transfer, oxygen binding
    ('METAL', 'MN'): 179,   # Manganese - photosynthesis, oxidoreductases
    ('METAL', 'CU'): 180,   # Copper - electron transfer, oxidases
    ('METAL', 'CO'): 181,   # Cobalt - vitamin B12, some enzymes
    ('METAL', 'NI'): 182,   # Nickel - urease, hydrogenases
    ('METAL', 'NA'): 183,   # Sodium - ion channels, osmotic balance
    ('METAL', 'K'): 184,    # Potassium - ion channels, protein stability
    ('METAL', 'METAL'): 185,  # Generic/unspecified metal

    # Special tokens
    ('UNK', 'UNK'): 186,
}

# Unknown token ID
UNK_TOKEN = 186

# =============================================================================
# Histidine Variants
# =============================================================================

HISTIDINE_VARIANTS = ['HIS', 'HID', 'HIE', 'HIP']

# =============================================================================
# Cysteine Variants
# =============================================================================

CYSTEINE_VARIANTS = ['CYS', 'CYX', 'CYM']

# =============================================================================
# Backbone Atom Names
# =============================================================================

BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
BACKBONE_ATOMS_WITH_CB = ['N', 'CA', 'C', 'O', 'CB']

# =============================================================================
# Standard Atoms per Residue (for standardization)
# =============================================================================

STANDARD_ATOMS = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': ['N', 'CA', 'C', 'O'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'UNK': ['N', 'CA', 'C', 'O', 'CB'],  # Unknown residue, backbone + CB only
}

# PTM standard atoms (for ptm_handling='preserve' mode)
STANDARD_ATOMS_PTM = {
    'SEP': ['N', 'CA', 'C', 'O', 'CB', 'OG', 'P', 'O1P', 'O2P', 'O3P'],  # Phosphoserine
    'TPO': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'P', 'O1P', 'O2P', 'O3P'],  # Phosphothreonine
    'PTR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'P', 'O1P', 'O2P', 'O3P'],  # Phosphotyrosine
    'MSE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SE', 'CE'],  # Selenomethionine
    'HYP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OD1'],  # Hydroxyproline
    'MLY': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CM1', 'CM2'],  # Dimethyllysine
    'M3L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CM1', 'CM2', 'CM3'],  # Trimethyllysine
    'ALY': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'C1', 'O1', 'C2'],  # Acetyllysine
    'CSO': ['N', 'CA', 'C', 'O', 'CB', 'SG', 'OD1'],  # Cysteine sulfenic acid
    'CSS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],  # S-mercaptocysteine
    'CME': ['N', 'CA', 'C', 'O', 'CB', 'SG', 'CE'],  # S-methylcysteine
    'OCS': ['N', 'CA', 'C', 'O', 'CB', 'SG', 'O1', 'O2'],  # Cysteinesulfonic acid
    'MEN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', 'CN'],  # N-methylasparagine
    'FME': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'CN', 'O1'],  # N-formylmethionine
}

# =============================================================================
# Residue Name Normalization Mapping
# =============================================================================

# Comprehensive mapping of non-standard residue names to standard amino acids
# Includes protonation states, PTMs, and modified residues
RESIDUE_NAME_MAPPING = {
    # -----------------------------------------------------------------
    # Histidine protonation states (heavy atoms identical)
    # -----------------------------------------------------------------
    'HID': 'HIS',  # δ-protonated histidine (neutral)
    'HIE': 'HIS',  # ε-protonated histidine (neutral)
    'HIP': 'HIS',  # doubly protonated histidine (positive)
    'HSD': 'HIS',  # alternative δ-protonated (CHARMM naming)
    'HSE': 'HIS',  # alternative ε-protonated (CHARMM naming)
    'HSP': 'HIS',  # alternative doubly protonated (CHARMM naming)
    'HIN': 'HIS',  # alternative neutral histidine

    # -----------------------------------------------------------------
    # Cysteine protonation/bonding states (heavy atoms identical)
    # -----------------------------------------------------------------
    'CYX': 'CYS',  # disulfide-bonded cysteine (deprotonated thiol)
    'CYM': 'CYS',  # deprotonated cysteine (thiolate anion)
    'CYN': 'CYS',  # alternative deprotonated cysteine

    # -----------------------------------------------------------------
    # Aspartic acid protonation states (heavy atoms identical)
    # -----------------------------------------------------------------
    'ASH': 'ASP',  # protonated aspartic acid (neutral COOH)
    'ASPP': 'ASP',  # alternative protonated form
    # NOTE: ASN (Asparagine) is different amino acid, NOT mapped to ASP

    # -----------------------------------------------------------------
    # Glutamic acid protonation states (heavy atoms identical)
    # -----------------------------------------------------------------
    'GLH': 'GLU',  # protonated glutamic acid (neutral COOH)
    'GLUP': 'GLU',  # alternative protonated form
    'GLUH': 'GLU',  # alternative protonated form

    # -----------------------------------------------------------------
    # Lysine protonation states (heavy atoms identical)
    # -----------------------------------------------------------------
    'LYN': 'LYS',  # deprotonated lysine (neutral amine)
    'LYSN': 'LYS',  # alternative deprotonated lysine

    # -----------------------------------------------------------------
    # Arginine protonation states (heavy atoms identical)
    # -----------------------------------------------------------------
    'ARN': 'ARG',  # deprotonated arginine (neutral, rare)

    # -----------------------------------------------------------------
    # Tyrosine protonation states (heavy atoms identical)
    # -----------------------------------------------------------------
    'TYM': 'TYR',  # deprotonated tyrosine (tyrosinate anion)
    'TYN': 'TYR',  # alternative deprotonated tyrosine

    # -----------------------------------------------------------------
    # Other protonation variants
    # -----------------------------------------------------------------
    'SER-': 'SER',  # deprotonated serine
    'THR-': 'THR',  # deprotonated threonine
    'TRP-': 'TRP',  # deprotonated tryptophan

    # -----------------------------------------------------------------
    # Modified amino acids (commonly found in X-ray structures and PTMs)
    # -----------------------------------------------------------------
    'MSE': 'MET',  # Selenomethionine (Se replaces S, common in X-ray)
    'SEP': 'SER',  # Phosphoserine
    'TPO': 'THR',  # Phosphothreonine
    'PTR': 'TYR',  # Phosphotyrosine
    'HYP': 'PRO',  # Hydroxyproline (common in collagen)
    'MLY': 'LYS',  # N-dimethyllysine
    'M3L': 'LYS',  # N-trimethyllysine
    'ALY': 'LYS',  # N-acetyllysine
    'CSO': 'CYS',  # S-hydroxycysteine (oxidized cysteine)
    'CSS': 'CYS',  # S-mercaptocysteine (disulfide-bonded)
    'CME': 'CYS',  # S-methylcysteine
    'OCS': 'CYS',  # Cysteinesulfonic acid (oxidized)
    'MEN': 'ASN',  # N-methylasparagine
    'FME': 'MET',  # N-formylmethionine (translation initiation)

    # -----------------------------------------------------------------
    # N-terminal and C-terminal variants (keep as is for HETATM)
    # -----------------------------------------------------------------
    'ACE': 'ACE',  # acetylated N-terminus
    'NME': 'NME',  # N-methylated C-terminus
    'NH2': 'NH2',  # amidated C-terminus
}

# List of PTM residue codes for special handling
PTM_RESIDUES = {
    'SEP', 'TPO', 'PTR',  # Phosphorylation
    'MSE',  # Selenomethionine
    'HYP',  # Hydroxyproline
    'MLY', 'M3L', 'ALY',  # Methylation/Acetylation
    'CSO', 'CSS', 'CME', 'OCS',  # Cysteine modifications
    'MEN', 'FME',  # Other modifications
}

# Nucleic acid residues to exclude
NUCLEIC_ACID_RESIDUES = {
    # DNA
    'DA', 'DT', 'DG', 'DC', 'DI', 'DU',
    # RNA
    'A', 'U', 'G', 'C', 'I',
    # Modified nucleotides
    'ADE', 'THY', 'GUA', 'CYT', 'URA',
    '1MA', '2MG', '4SU', '5MC', '5MU', 'PSU', 'H2U', 'M2G', 'OMC', 'OMG',
}

# =============================================================================
# Metal Ion Residue Names
# =============================================================================

# Common metal ion residue names in PDB files
# These should be excluded from amino acid sequence/residue counts
METAL_RESIDUES = {
    'ZN',   # Zinc - structural (zinc fingers), catalytic centers
    'CA',   # Calcium - signaling, structural stabilization
    'MG',   # Magnesium - ATP binding, enzymatic cofactor
    'MN',   # Manganese - photosynthesis, oxidoreductases
    'FE',   # Iron - electron transfer, oxygen binding (heme)
    'CU',   # Copper - electron transfer, oxidases
    'NI',   # Nickel - urease, hydrogenases
    'CO',   # Cobalt - vitamin B12, some enzymes
    'NA',   # Sodium - ion channels, osmotic balance
    'K',    # Potassium - ion channels, protein stability
}

# =============================================================================
# Residue Maximum Accessible Surface Area (Å²)
# =============================================================================
# Empirical max ASA from Tien et al. 2013 (Gly-X-Gly tripeptides)
RESIDUE_MAX_SASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
    'UNK': 200.0, 'XXX': 200.0,
}

# =============================================================================
# Formal Charges at Physiological pH (per-atom)
# =============================================================================
# Partial charges for charged residue sidechain atoms
FORMAL_CHARGE_MAP = {
    # ASP: carboxylate (-1 total, split across OD1/OD2)
    ('ASP', 'OD1'): -0.5, ('ASP', 'OD2'): -0.5,
    # GLU: carboxylate (-1 total, split across OE1/OE2)
    ('GLU', 'OE1'): -0.5, ('GLU', 'OE2'): -0.5,
    # LYS: ammonium (+1)
    ('LYS', 'NZ'): 1.0,
    # ARG: guanidinium (+1 total, distributed)
    ('ARG', 'NH1'): 0.5, ('ARG', 'NH2'): 0.5,
}

# =============================================================================
# H-Bond Donor Atoms (heavy atoms bonded to H that can donate)
# =============================================================================
HBOND_DONOR_ATOMS = {
    # Backbone N is donor for all residues except PRO
    # (handled programmatically: if atom_name == 'N' and res_name != 'PRO')
    # Sidechain donors:
    ('ARG', 'NE'), ('ARG', 'NH1'), ('ARG', 'NH2'),
    ('ASN', 'ND2'),
    ('GLN', 'NE2'),
    ('HIS', 'ND1'), ('HIS', 'NE2'),
    ('LYS', 'NZ'),
    ('SER', 'OG'),
    ('THR', 'OG1'),
    ('TRP', 'NE1'),
    ('TYR', 'OH'),
    ('CYS', 'SG'),
}

# =============================================================================
# H-Bond Acceptor Atoms (atoms with lone pairs that can accept H-bond)
# =============================================================================
HBOND_ACCEPTOR_ATOMS = {
    # Backbone O is acceptor for all residues
    # (handled programmatically: if atom_name == 'O')
    # Sidechain acceptors:
    ('ASN', 'OD1'),
    ('ASP', 'OD1'), ('ASP', 'OD2'),
    ('GLN', 'OE1'),
    ('GLU', 'OE1'), ('GLU', 'OE2'),
    ('HIS', 'ND1'), ('HIS', 'NE2'),
    ('MET', 'SD'),
    ('SER', 'OG'),
    ('THR', 'OG1'),
    ('TYR', 'OH'),
}

# Set of backbone atom names
BACKBONE_ATOM_SET = frozenset({'N', 'CA', 'C', 'O'})

# =============================================================================
# Residue Physicochemical Properties
# =============================================================================
# Per-residue properties for residue-level graph features.
# Sources: Kyte-Doolittle hydrophobicity, Zamyatnin molecular volumes,
# physiological pH charges, Karplus flexibility (B-factor proxy),
# Zimmerman polarity index.
# All values pre-normalized to approximately [0, 1] or [-1, 1].

RESIDUE_PROPERTIES = {
    #              hydrophobicity  volume   charge  flexibility  polarity
    #              (KD, [-1,1])    (norm)   (-1~1)  (0~1)        (0~1)
    'ALA': ( 0.40,  0.27,  0.0, 0.36, 0.00),
    'ARG': (-1.00,  0.72,  1.0, 0.53, 1.00),
    'ASN': (-0.78,  0.46,  0.0, 0.46, 0.69),
    'ASP': (-0.78,  0.40, -1.0, 0.51, 1.00),
    'CYS': ( 0.56,  0.38,  0.0, 0.33, 0.26),
    'GLN': (-0.78,  0.56,  0.0, 0.49, 0.69),
    'GLU': (-0.78,  0.51, -1.0, 0.50, 1.00),
    'GLY': (-0.09,  0.16,  0.0, 0.54, 0.00),
    'HIS': (-0.72,  0.56,  0.5, 0.32, 0.65),
    'ILE': ( 1.00,  0.56,  0.0, 0.46, 0.00),
    'LEU': ( 0.84,  0.56,  0.0, 0.51, 0.00),
    'LYS': (-0.87,  0.62,  1.0, 0.47, 1.00),
    'MET': ( 0.42,  0.56,  0.0, 0.49, 0.04),
    'PHE': ( 0.62,  0.67,  0.0, 0.36, 0.03),
    'PRO': (-0.36,  0.40,  0.0, 0.51, 0.38),
    'SER': (-0.18,  0.29,  0.0, 0.51, 0.41),
    'THR': (-0.16,  0.38,  0.0, 0.44, 0.40),
    'TRP': (-0.24,  0.82,  0.0, 0.31, 0.41),
    'TYR': (-0.29,  0.71,  0.0, 0.42, 0.53),
    'VAL': ( 0.93,  0.47,  0.0, 0.39, 0.00),
    'UNK': ( 0.00,  0.50,  0.0, 0.45, 0.50),
    'Other':( 0.00,  0.50,  0.0, 0.45, 0.50),
}
NUM_RESIDUE_PROPERTIES = 5
