# Feature Types Overview

## Molecule Features

### Available Features
- **40 Molecular Descriptors**: Physicochemical, topological, and structural properties
- **9 Fingerprint Types**: Morgan, MACCS, RDKit, Atom Pair, Torsion, Layered, Pattern, Pharmacophore
- **Graph Representations**: 122D atom features, 44D bond features with 3D coordinates

### API
```python
from plfeature import MoleculeFeaturizer

featurizer = MoleculeFeaturizer()
features = featurizer.get_feature("CCO")  # Descriptors and fingerprints
node, edge = featurizer.get_graph("CCO")  # Graph representation
```

## Protein Features

### Atom-Level
- **175 Token Types**: Unique residue-atom combinations
- **Atomic SASA**: Solvent accessible surface area per atom
- **3D Coordinates**: Precise atomic positions

### Residue-Level
- **Sequence Features**: Residue types and one-hot encoding
- **Geometric Features**: Dihedrals, curvature, torsion, distances
- **SASA Features**: 10-component solvent accessibility analysis
- **Contact Maps**: Customizable distance thresholds (4.5-12.0 Å)
- **Graph Representations**: Node and edge features for protein networks

### API
```python
from plfeature import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")
atom_features = featurizer.get_atom_features_with_sasa()
node, edge = featurizer.get_features()
contacts = featurizer.get_contact_map(cutoff=8.0)
```

## Documentation

- [Molecular Descriptors & Fingerprints](molecule_feature.md)
- [Molecule Graph Representations](molecule_graph.md)
- [Molecular Descriptors Reference](molecule_descriptors.md)
- [Protein Residue Features](protein_residue_feature.md)
- [Protein Atom Features](protein_atom_feature.md)