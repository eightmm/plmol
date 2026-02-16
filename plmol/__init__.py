"""
plmol Package - Aligned & Flattened Structure
"""

from .protein.core import Protein
from .ligand.core import Ligand
from .protein.protein_featurizer import ProteinFeaturizer
from .protein.pdb_standardizer import PDBStandardizer
from .protein.residue_featurizer import ResidueFeaturizer
from .protein.atom_featurizer import AtomFeaturizer
from .protein.hierarchical_featurizer import HierarchicalFeaturizer, HierarchicalProteinData
from .protein.esm_featurizer import ESMFeaturizer
from .ligand.base import MoleculeFeaturizer
from .ligand.graph import MoleculeGraphFeaturizer
from .ligand.featurizer import LigandFeaturizer
from .interaction.pli_featurizer import PLInteractionFeaturizer
from .complex import Complex
from .errors import PlmolError, InputError, DependencyError, FeatureError
from .specs import FEATURE_SPECS, FeatureSpec

from . import constants

__version__ = "0.2.1"

__all__ = [
    "Protein", "ProteinFeaturizer", "PDBStandardizer", "ResidueFeaturizer", "AtomFeaturizer",
    "HierarchicalFeaturizer", "HierarchicalProteinData", "ESMFeaturizer",
    "Ligand", "MoleculeFeaturizer", "MoleculeGraphFeaturizer", "LigandFeaturizer",
    "PLInteractionFeaturizer", "Complex",
    "PlmolError", "InputError", "DependencyError", "FeatureError",
    "FeatureSpec", "FEATURE_SPECS",
    "constants"
]
