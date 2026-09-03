"""plmol - Unified protein-ligand feature extraction toolkit."""

# --- Protein ---
from .protein.core import Protein
from .protein.featurizer import ProteinFeaturizer
from .protein.pdb_standardizer import PDBStandardizer
from .protein.residue_featurizer import ResidueFeaturizer
from .protein.atom_featurizer import AtomFeaturizer
from .protein.hierarchical_featurizer import HierarchicalFeaturizer, HierarchicalProteinData
from .protein.plm import (
    PLMSpec,
    PLM_REGISTRY,
    ProteinLanguageModel,
    clear_plm_cache,
    embed_sequence,
    embed_sequences,
    list_protein_language_models,
    load_plm,
    plm_dim,
    register_plm,
)
from .protein.esm_featurizer import ESMFeaturizer

# --- Ligand ---
from .ligand.core import Ligand
from .ligand.descriptors import MoleculeFeaturizer
from .ligand.graph import MoleculeGraphFeaturizer
from .base import BaseMolecule, TempFileOwner
from .arrays import to_numpy, to_torch
from .spatial import (
    SPATIAL_BACKENDS,
    NeighbourIndex,
    get_spatial_backend,
    knn,
    pairs_within,
    resolve_spatial_backend,
    set_spatial_backend,
)
from .sasa import (
    SASA_BACKENDS,
    get_sasa_backend,
    resolve_sasa_backend,
    set_sasa_backend,
    shrake_rupley,
)
from .graph_view import FEATURE_DIMS, as_graph, collate, feature_dims
from .ligand.fragment_graph import build_fragment_graph
from .ligand.graph_edge_features import (
    BOND_FEATURE_DIM,
    BOND_VIEW_CHANNELS,
    BOND_VIEW_DROPPED_CHANNELS,
    bond_view_channels,
    PAIR_FEATURE_DIM,
)
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

__version__ = "0.3.0"

__all__ = [
    "Protein", "ProteinFeaturizer", "PDBStandardizer", "ResidueFeaturizer", "AtomFeaturizer",
    "HierarchicalFeaturizer", "HierarchicalProteinData", "ESMFeaturizer",
    "PLMSpec", "PLM_REGISTRY", "ProteinLanguageModel", "register_plm",
    "list_protein_language_models", "plm_dim", "load_plm", "clear_plm_cache",
    "embed_sequence", "embed_sequences",
    "Ligand", "MoleculeFeaturizer", "MoleculeGraphFeaturizer", "LigandFeaturizer",
    "fragment_by_brics", "fragment_molecule", "fragment_on_rotatable_bonds",
    "build_bond_graph", "build_fragment_graph",
    "BOND_FEATURE_DIM", "PAIR_FEATURE_DIM",
    "BOND_VIEW_CHANNELS", "BOND_VIEW_DROPPED_CHANNELS", "bond_view_channels",
    "bond_view_channels",
    "as_graph", "collate", "feature_dims", "FEATURE_DIMS",
    "SASA_BACKENDS", "set_sasa_backend", "get_sasa_backend",
    "resolve_sasa_backend", "shrake_rupley",
    "to_torch", "to_numpy",
    "SPATIAL_BACKENDS", "set_spatial_backend", "get_spatial_backend",
    "resolve_spatial_backend", "knn", "NeighbourIndex", "pairs_within",
    "NucleicAcid", "NucleicFeaturizer",
    "StructureParser", "MMCIFParser",
    "PLInteractionFeaturizer", "Complex", "MolecularComplex",
    "PlmolError", "InputError", "DependencyError", "FeatureError",
    "FeatureSpec", "FEATURE_SPECS",
    "BaseMolecule", "TempFileOwner",
    "constants",
]
