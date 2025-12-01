"""
plfeature Package

A comprehensive Python package for extracting features from molecular and protein structures
for machine learning applications and protein-ligand modeling.
"""

# Import molecule featurizer components
from .molecule_featurizer import MoleculeFeaturizer, MoleculeGraphFeaturizer

# Import protein featurizer components
from .protein_featurizer import (
    ProteinFeaturizer,
    PDBStandardizer,
    ResidueFeaturizer,
    HierarchicalFeaturizer,
    HierarchicalProteinData,
    standardize_pdb,
    extract_hierarchical_features,
)

# Import interaction featurizer components
from .interaction_featurizer import PLInteractionFeaturizer

# Import ESM featurizer (lazy import to avoid dependency issues)
def get_esm_featurizer():
    """Get ESMFeaturizer class (lazy import)."""
    from .protein_featurizer.esm_featurizer import ESMFeaturizer
    return ESMFeaturizer

def get_dual_esm_featurizer():
    """Get DualESMFeaturizer class (lazy import)."""
    from .protein_featurizer.esm_featurizer import DualESMFeaturizer
    return DualESMFeaturizer

# Import constants module
from . import constants

__version__ = "0.1.0"
__author__ = "Jaemin Sim"

__all__ = [
    # Molecule features
    "MoleculeFeaturizer",
    "MoleculeGraphFeaturizer",
    # Protein features
    "ProteinFeaturizer",
    "PDBStandardizer",
    "ResidueFeaturizer",
    "HierarchicalFeaturizer",
    "HierarchicalProteinData",
    "standardize_pdb",
    "extract_hierarchical_features",
    # ESM features
    "get_esm_featurizer",
    "get_dual_esm_featurizer",
    # Interaction features
    "PLInteractionFeaturizer",
    # Constants
    "constants",
]

# Convenience functions for quick access
def extract_molecule_features(mol_or_smiles, add_hs=False, canonicalize=True):
    """
    Convenience function to extract molecule features.

    Args:
        mol_or_smiles: RDKit mol object or SMILES string
        add_hs: Whether to add hydrogens (default: False, heavy atoms only)
        canonicalize: Whether to canonicalize atom order (default: True)

    Returns:
        Dictionary containing molecule features
    """
    featurizer = MoleculeFeaturizer(
        mol_or_smiles, hydrogen=add_hs, canonicalize=canonicalize
    )
    return featurizer.get_feature()

def extract_protein_features(pdb_file, standardize=True, save_to=None):
    """
    Convenience function to extract protein features.

    Args:
        pdb_file: Path to PDB file
        standardize: Whether to standardize PDB first (default: True)
        save_to: Optional path to save features

    Returns:
        Dictionary containing protein features
    """
    featurizer = ProteinFeaturizer(standardize=standardize)
    return featurizer.extract(pdb_file, save_to=save_to)
