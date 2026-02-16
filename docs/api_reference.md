# plmol API Reference

> **Version**: 0.2.1
> Protein-Ligand Molecular Feature Extraction Toolkit

## Detailed References

- [Protein API](protein.md) — Initialization, graph (residue/atom), backbone, surface, voxel, sequence, pocket, ESM embeddings, geometry functions
- [Ligand API](ligand.md) — Initialization, graph, fingerprint, fragment, surface, voxel, low-level featurizers
- [Complex API](complex.md) — Initialization, combined featurization, interaction features, contact edges, pocket extraction

## Quick Start

```python
from plmol import Protein, Ligand, Complex

# Protein
protein = Protein.from_pdb("protein.pdb")
result = protein.featurize(mode="all")

# Ligand
ligand = Ligand.from_smiles("CCO")
result = ligand.featurize(mode=["graph", "fingerprint"])

# Complex
cx = Complex.from_files("protein.pdb", "ligand.sdf")
result = cx.featurize(requests="all")
```

## Directory Structure

```
plmol/
├── __init__.py                     # Top-level exports (Protein, Ligand, Complex, ...)
├── base.py                         # BaseMolecule abstract class
├── cache.py                        # LRU caching utility
├── complex.py                      # Complex class
├── errors.py                       # PlmolError, InputError, DependencyError, FeatureError
├── specs.py                        # FeatureSpec, LIGAND_SPEC, PROTEIN_SPEC, INTERACTION_SPEC
├── utils.py                        # kNN mask utilities
├── constants/
│   ├── amino_acids.py              # Amino acid mappings & tokens
│   ├── elements.py                 # Element types & periodic table
│   ├── interactions.py             # Interaction types & ideal distances
│   ├── physical_properties.py      # VdW radius, mass, etc.
│   ├── runtime.py                  # Default parameters (cutoffs, grid density)
│   └── smarts_patterns.py          # Pharmacophore & rotatable bond SMARTS patterns
├── protein/
│   ├── core.py                     # Protein class
│   ├── protein_featurizer.py       # ProteinFeaturizer (parse + cache)
│   ├── residue_featurizer.py       # ResidueFeaturizer (residue features)
│   ├── atom_featurizer.py          # AtomFeaturizer (atom features)
│   ├── geometry.py                 # Stateless geometric functions
│   ├── backbone_featurizer.py      # Backbone features for inverse folding
│   ├── hierarchical_featurizer.py  # HierarchicalFeaturizer + HierarchicalProteinData
│   ├── esm_featurizer.py          # ESM3/ESMC embedding extraction
│   ├── pdb_standardizer.py        # PDB standardization
│   └── utils.py                    # PDBParser and utilities
├── ligand/
│   ├── core.py                     # Ligand class
│   ├── descriptors.py              # MoleculeFeaturizer (descriptors + fingerprints)
│   ├── featurizer.py               # LigandFeaturizer
│   ├── fragment.py                 # Rotatable-bond fragmentation
│   └── graph.py                    # MoleculeGraphFeaturizer
├── interaction/
│   ├── pli_featurizer.py           # PLInteractionFeaturizer
│   └── pocket_extractor.py         # Pocket extraction
├── surface/
│   └── __init__.py                 # Surface building (mesh / point cloud)
├── voxel/
│   └── __init__.py                 # Voxel building (3D grid)
├── io/
│   └── loaders.py                  # load_protein_input, load_ligand_input
└── cli/                            # Command-line interface
```
