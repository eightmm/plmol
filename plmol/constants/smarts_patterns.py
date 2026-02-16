"""
SMARTS Patterns for Chemical Feature Detection.

Contains chemically accurate SMARTS patterns for identifying pharmacophoric
features in molecules. These patterns are designed to correctly identify
functional groups relevant to protein-ligand interactions.
"""

# =============================================================================
# Primary Pharmacophore Patterns (Chemically Accurate)
# =============================================================================

# These patterns are comprehensive and chemically validated
PHARMACOPHORE_SMARTS = {
    # H-bond acceptors: atoms with available lone pairs
    # Includes: sp3 O/S with H, sp2 O/S, anionic O/S, sp3 N (not amide),
    #           aromatic N without H, aromatic O/S (not next to aromatic N)
    'h_acceptor': (
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),"
        "$([O,S;H0;v2]),"
        "$([O,S;-]),"
        "$([N;v3;!$(N-*=[O,N,P,S])]),"
        "n&H0&+0,"
        "$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
    ),

    # H-bond donors: atoms with H that can donate
    # Includes: N with H (sp3 or cationic sp3), O/S with H, aromatic N with H
    'h_donor': "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",

    # Hydrophobic: non-polar atoms
    # Includes: C (not carbonyl/nitrile), aromatic c, divalent S, halogens
    'hydrophobic': "[C,c,S&H0&v2,F,Cl,Br,I&!$(C=[O,N,P,S])&!$(C#N);!$(C=O)]",

    # Positive ionizable: atoms that can be protonated
    # Includes: cationic N, primary amine (not amide), secondary amine (not amide),
    #           tertiary amine (not amide)
    'positive': (
        "[#7;+,"
        "$([N;H2&+0][$([C,a]);!$([C,a](=O))]),"
        "$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),"
        "$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]"
    ),

    # Negative ionizable: carboxylic acids, phosphates, sulfonates, etc.
    'negative': "[$([C,S](=[O,S,P])-[O;H1,-1])]",

    # Aromatic: any aromatic atom
    'aromatic': "[a]",

    # Halogen bond donor: halogens attached to carbon (sigma-hole)
    'halogen': "[Cl,Br,I;$([Cl,Br,I]-[C,c])]",

    # Metal coordinating atoms: lone-pair donors that can coordinate metal ions
    # Includes: O/S with lone pairs, anionic O/S, sp3 N (not amide),
    #           aromatic N without H (imidazole, pyridine)
    'metal_coord': (
        "[$([O;H0,H1;v2]),"
        "$([O;-]),"
        "$([S;H0;v2]),"
        "$([N;v3;!$(N-*=O)]),"
        "n&H0&+0]"
    ),
}

# =============================================================================
# Extended Patterns for Detailed Analysis
# =============================================================================

# More specific patterns when detailed functional group info is needed
FUNCTIONAL_GROUP_SMARTS = {
    # Carboxylic acid and carboxylate
    'carboxylic_acid': '[CX3](=O)[OX2H1]',
    'carboxylate': '[CX3](=O)[OX1-]',

    # Amines by class
    'primary_amine': '[NX3H2;!$(NC=O)]',
    'secondary_amine': '[NX3H1;!$(NC=O)]',
    'tertiary_amine': '[NX3H0;!$(NC=O)]',

    # Amide
    'amide': '[NX3][CX3](=[OX1])',

    # Hydroxyl groups
    'alcohol': '[OX2H][CX4]',
    'phenol': '[OX2H]c',

    # Carbonyl
    'aldehyde': '[CX3H1](=O)',
    'ketone': '[CX3](=O)([C])[C]',

    # Nitrogen heterocycles
    'pyridine_n': '[nX2]c',
    'imidazole': 'n1c[nH]cc1',

    # Sulfur groups
    'thiol': '[SX2H]',
    'thioether': '[SX2]([C])[C]',
    'sulfoxide': '[SX3](=O)',
    'sulfone': '[SX4](=O)(=O)',

    # Phosphate
    'phosphate': '[PX4](=O)([O])([O])[O]',

    # Guanidinium (Arg-like)
    'guanidinium': '[NX3H2]C(=[NX2H])([NX3H2])',

    # Nitro
    'nitro': '[NX3+](=O)[O-]',
}

# =============================================================================
# Aromatic Ring Patterns
# =============================================================================

AROMATIC_RING_SMARTS = {
    'benzene': 'c1ccccc1',
    'pyridine': 'n1ccccc1',
    'pyrimidine': 'n1ccncc1',
    'pyrrole': '[nH]1cccc1',
    'imidazole': 'n1c[nH]cc1',
    'indole': 'c1ccc2[nH]ccc2c1',
    'furan': 'o1cccc1',
    'thiophene': 's1cccc1',
}

# =============================================================================
# Protein-Specific Patterns
# =============================================================================

RESIDUE_SMARTS = {
    # Charged sidechains
    'lys_amine': '[NX3H2,NX4H3+][CX4]',       # Lys epsilon-NH2/NH3+
    'arg_guanidinium': '[NX3H2]C(=[NX2H,NX3H2+])([NX3H2])',  # Arg guanidinium
    'asp_glu_carboxyl': '[CX3](=O)[OX1-,OX2H1]',  # Asp/Glu carboxyl

    # Aromatic sidechains
    'phe_ring': 'c1ccc(C)cc1',
    'tyr_ring': 'c1cc(O)ccc1',
    'trp_indole': 'c1ccc2[nH]ccc2c1',
    'his_imidazole': 'n1c[nH]cc1',

    # Polar sidechains
    'ser_thr_oh': '[OX2H][CX4]',
    'cys_sh': '[SX2H][CX4]',
    'asn_gln_amide': '[NX3H2][CX3](=O)',
}

# =============================================================================
# Backbone Pattern
# =============================================================================

BACKBONE_SMARTS = {
    'peptide_bond': '[NX3][CX4][CX3](=[OX1])',
    'backbone_nh': '[NX3H1][CX4][CX3](=[OX1])',
    'backbone_co': '[CX3](=[OX1])[NX3]',
}

# =============================================================================
# Rotatable Bond Pattern
# =============================================================================

ROTATABLE_BOND_SMARTS = "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"

# =============================================================================
# Pharmacophore Categories (maps category to pattern key)
# =============================================================================

PHARMACOPHORE_CATEGORIES = {
    'hbond_donor': 'h_donor',
    'hbond_acceptor': 'h_acceptor',
    'positive_charge': 'positive',
    'negative_charge': 'negative',
    'aromatic': 'aromatic',
    'hydrophobic': 'hydrophobic',
    'halogen_bond': 'halogen',
    'metal_coord': 'metal_coord',
}

# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# These are kept for backward compatibility with existing code
# They map to the new PHARMACOPHORE_SMARTS
CHEMICAL_SMARTS = {
    'h_acceptor': PHARMACOPHORE_SMARTS['h_acceptor'],
    'h_donor': PHARMACOPHORE_SMARTS['h_donor'],
    'hydrophobic': PHARMACOPHORE_SMARTS['hydrophobic'],
    'positive': PHARMACOPHORE_SMARTS['positive'],
    'negative': PHARMACOPHORE_SMARTS['negative'],
}
