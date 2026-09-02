"""plmol - Unified protein-ligand feature extraction toolkit."""

# --- Protein ---
from .protein.core import Protein
from .protein.featurizer import ProteinFeaturizer
from .protein.pdb_standardizer import PDBStandardizer
from .protein.residue_featurizer import ResidueFeaturizer
from .protein.atom_featurizer import AtomFeaturizer
from .protein.hierarchical_featurizer import HierarchicalFeaturizer, HierarchicalProteinData
from .protein.esm_featurizer import ESMFeaturizer

# --- Ligand ---
from .ligand.core import Ligand
from .ligand.descriptors import MoleculeFeaturizer
from .ligand.graph import MoleculeGraphFeaturizer
from .ligand.line_graph import build_bond_graph
from .ligand.featurizer import LigandFeaturizer
from .ligand.fragment import fragment_by_brics, fragment_molecule, fragment_on_rotatable_bonds

# --- Nucleic Acid ---
from .nucleic_acid.core import NucleicAcid
from .nucleic_acid.featurizer import NucleicFeaturizer

# --- Parsers ---
from .parsers import StructureParser, MMCIFParser

# --- Interaction ---
from .interaction.pli_featurizer import PLInteractionFeaturizer

# --- Complex ---
from .complex import Complex, MolecularComplex

# --- Infrastructure ---
from .errors import PlmolError, InputError, DependencyError, FeatureError
from .specs import FEATURE_SPECS, FeatureSpec
from . import constants

__version__ = "0.2.1"

__all__ = [
    "Protein", "ProteinFeaturizer", "PDBStandardizer", "ResidueFeaturizer", "AtomFeaturizer",
    "HierarchicalFeaturizer", "HierarchicalProteinData", "ESMFeaturizer",
    "Ligand", "MoleculeFeaturizer", "MoleculeGraphFeaturizer", "LigandFeaturizer",
    "fragment_by_brics", "fragment_molecule", "fragment_on_rotatable_bonds",
    "build_bond_graph",
    "NucleicAcid", "NucleicFeaturizer",
    "StructureParser", "MMCIFParser",
    "PLInteractionFeaturizer", "Complex", "MolecularComplex",
    "PlmolError", "InputError", "DependencyError", "FeatureError",
    "FeatureSpec", "FEATURE_SPECS",
    "constants",
]
