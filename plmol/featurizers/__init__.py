"""Public featurizer layer (lazy exports to avoid import cycles)."""

from importlib import import_module
from typing import Any

__all__ = [
    "LigandFeaturizer",
    "MoleculeFeaturizer",
    "MoleculeGraphFeaturizer",
    "ProteinFeaturizer",
    "PDBStandardizer",
    "ResidueFeaturizer",
    "AtomFeaturizer",
    "HierarchicalFeaturizer",
    "HierarchicalProteinData",
    "ESMFeaturizer",
    "PLInteractionFeaturizer",
]


_EXPORT_MAP = {
    "LigandFeaturizer": "plmol.ligand.featurizer",
    "MoleculeFeaturizer": "plmol.ligand.base",
    "MoleculeGraphFeaturizer": "plmol.ligand.graph",
    "ProteinFeaturizer": "plmol.protein.protein_featurizer",
    "PDBStandardizer": "plmol.protein.pdb_standardizer",
    "ResidueFeaturizer": "plmol.protein.residue_featurizer",
    "AtomFeaturizer": "plmol.protein.atom_featurizer",
    "HierarchicalFeaturizer": "plmol.protein.hierarchical_featurizer",
    "HierarchicalProteinData": "plmol.protein.hierarchical_featurizer",
    "ESMFeaturizer": "plmol.protein.esm_featurizer",
    "PLInteractionFeaturizer": "plmol.interaction.pli_featurizer",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
