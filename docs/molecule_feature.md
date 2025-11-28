# Molecule Feature Extraction

## Overview

The `MoleculeFeaturizer` extracts molecular-level features including:
- **40 normalized descriptors**: Physicochemical, topological, structural properties
- **9 fingerprint types**: For similarity search and machine learning

## Quick Start

```python
from plfeature import MoleculeFeaturizer

# Initialize with SMILES
featurizer = MoleculeFeaturizer("CCO")  # ethanol

# Get all features
features = featurizer.get_feature()
descriptors = features['descriptor']  # [40]
morgan_fp = features['morgan']        # [2048]

# Or use functional API
featurizer = MoleculeFeaturizer()
features = featurizer.get_feature("CCO")
```

## Input Formats

```python
from rdkit import Chem

# From SMILES
featurizer = MoleculeFeaturizer("CCO")
featurizer = MoleculeFeaturizer("CCO", hydrogen=False)  # without hydrogens

# From RDKit mol
mol = Chem.MolFromSmiles("CCO")
featurizer = MoleculeFeaturizer(mol, hydrogen=True)

# From SDF file
suppl = Chem.SDMolSupplier('molecules.sdf')
for mol in suppl:
    featurizer = MoleculeFeaturizer(mol)
    features = featurizer.get_feature()
```

## Molecular Descriptors (40 dimensions)

All descriptors are normalized to approximately [0, 1] range.

### Physicochemical Properties (12)

| Descriptor | Description | Normalization |
|------------|-------------|---------------|
| `mw` | Molecular weight | /1000.0 |
| `logp` | Octanol-water partition coefficient | (+5)/10.0 |
| `tpsa` | Topological polar surface area | /200.0 |
| `n_rotatable_bonds` | Number of rotatable bonds | /20.0 |
| `flexibility` | Ratio of rotatable bonds to total bonds | 0-1 |
| `hbd` | Hydrogen bond donors | /10.0 |
| `hba` | Hydrogen bond acceptors | /15.0 |
| `n_atoms` | Total number of atoms | /100.0 |
| `n_bonds` | Total number of bonds | /120.0 |
| `n_rings` | Number of rings | /10.0 |
| `n_aromatic_rings` | Number of aromatic rings | /8.0 |
| `heteroatom_ratio` | Ratio of heteroatoms to total atoms | 0-1 |

### Topological Indices (9)

| Descriptor | Description | Normalization |
|------------|-------------|---------------|
| `balaban_j` | Balaban's J index (topological) | /5.0 |
| `bertz_ct` | Bertz complexity index | /2000.0 |
| `chi0` | Connectivity index 0 | /50.0 |
| `chi1` | Connectivity index 1 | /30.0 |
| `chi0n` | Normalized connectivity index | /50.0 |
| `hall_kier_alpha` | Hall-Kier alpha value | /5.0 |
| `kappa1` | Kappa shape index 1 | /50.0 |
| `kappa2` | Kappa shape index 2 | /20.0 |
| `kappa3` | Kappa shape index 3 | /10.0 |

### Electronic & Surface Properties (4)

| Descriptor | Description | Normalization |
|------------|-------------|---------------|
| `mol_mr` | Molar refractivity | /200.0 |
| `labute_asa` | LabuteASA (accessible surface area) | /500.0 |
| `num_radical_electrons` | Number of radical electrons | /5.0 |
| `num_valence_electrons` | Number of valence electrons | /500.0 |

### Ring & Structural Features (10)

| Descriptor | Description | Normalization |
|------------|-------------|---------------|
| `num_saturated_rings` | Number of saturated rings | /10.0 |
| `num_aliphatic_rings` | Number of aliphatic rings | /10.0 |
| `num_saturated_heterocycles` | Saturated heterocycles | /8.0 |
| `num_aliphatic_heterocycles` | Aliphatic heterocycles | /8.0 |
| `num_aromatic_heterocycles` | Aromatic heterocycles | /8.0 |
| `num_heteroatoms` | Total heteroatoms | /30.0 |
| `formal_charge` | Sum of formal charges | (+5)/10.0 |
| `n_ring_systems` | Number of ring systems | /8.0 |
| `max_ring_size` | Maximum ring size | /12.0 |
| `avg_ring_size` | Average ring size | /8.0 |

### Drug-likeness Properties (5)

| Descriptor | Description | Range |
|------------|-------------|-------|
| `lipinski_violations` | Lipinski rule of 5 violations | 0-1 |
| `passes_lipinski` | Passes Lipinski rule | 0 or 1 |
| `qed` | Quantitative Estimate of Drug-likeness | 0-1 |
| `num_heavy_atoms` | Number of heavy atoms | /50.0 |
| `frac_csp3` | Fraction of sp3 carbons | 0-1 |

## Molecular Fingerprints

### Available Fingerprints

```python
features = featurizer.get_feature()

# Returns dictionary with all fingerprints:
{
    'descriptor': torch.Tensor,           # [40]
    'maccs': torch.Tensor,                # [167]
    'morgan': torch.Tensor,               # [2048]
    'morgan_count': torch.Tensor,         # [2048]
    'feature_morgan': torch.Tensor,       # [2048]
    'rdkit': torch.Tensor,                # [2048]
    'atom_pair': torch.Tensor,            # [2048]
    'topological_torsion': torch.Tensor,  # [2048]
    'pharmacophore2d': torch.Tensor,      # [1024]
}
```

### Fingerprint Types

| Fingerprint | Dimensions | Description |
|-------------|------------|-------------|
| `maccs` | 167 | MACCS structural keys |
| `morgan` | 2048 | Morgan circular fingerprint (radius=2, with chirality) |
| `morgan_count` | 2048 | Morgan count fingerprint |
| `feature_morgan` | 2048 | Feature-based Morgan fingerprint |
| `rdkit` | 2048 | RDKit path-based fingerprint |
| `atom_pair` | 2048 | Atom pair fingerprint |
| `topological_torsion` | 2048 | Topological torsion fingerprint |
| `pharmacophore2d` | 1024 | 2D pharmacophore fingerprint |

## Usage Examples

### Descriptor Access

```python
featurizer = MoleculeFeaturizer("CCO")
features = featurizer.get_feature()
descriptors = features['descriptor']  # [40]

# Access specific descriptors by index
mw = descriptors[0]          # Molecular weight
logp = descriptors[1]        # LogP
tpsa = descriptors[2]        # TPSA
qed = descriptors[37]        # QED score
```

### Fingerprint Similarity

```python
from torch.nn.functional import cosine_similarity

mol1 = MoleculeFeaturizer("CCO")
mol2 = MoleculeFeaturizer("CCCO")

fp1 = mol1.get_feature()['morgan']
fp2 = mol2.get_feature()['morgan']

similarity = cosine_similarity(fp1.unsqueeze(0), fp2.unsqueeze(0))
print(f"Tanimoto-like similarity: {similarity.item():.3f}")
```

### Batch Processing

```python
smiles_list = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O"]

all_descriptors = []
all_fingerprints = []

for smiles in smiles_list:
    featurizer = MoleculeFeaturizer(smiles)
    features = featurizer.get_feature()
    all_descriptors.append(features['descriptor'])
    all_fingerprints.append(features['morgan'])

# Stack into tensors
descriptors_batch = torch.stack(all_descriptors)  # [4, 40]
fingerprints_batch = torch.stack(all_fingerprints)  # [4, 2048]
```

### Custom SMARTS Features

```python
# Define custom SMARTS patterns
custom_smarts = {
    'carboxyl': '[CX3](=O)[OX2H1]',
    'amine': '[NX3;H2,H1;!$(NC=O)]',
    'hydroxyl': '[OX2H]',
}

featurizer = MoleculeFeaturizer("CC(=O)O", custom_smarts=custom_smarts)
node, edge, adj = featurizer.get_graph(include_custom_smarts=True)

# Custom SMARTS features are appended to node features
# node['node_feats'] shape: [n_atoms, 157 + n_custom_patterns]
```

## Normalization Strategy

Descriptors are normalized to approximately [0, 1] range using:

1. **Division normalization**: Most descriptors divided by typical maximum values
2. **Range shifting**: LogP and formal charge shifted to positive range before division
3. **Direct ratios**: Some descriptors (flexibility, heteroatom_ratio, frac_csp3) naturally in [0, 1]

## API Reference

### Constructor

```python
MoleculeFeaturizer(
    mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
    hydrogen: bool = True,
    custom_smarts: Optional[Dict[str, str]] = None,
)
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_feature(mol_or_smiles)` | All descriptors and fingerprints | `Dict` |
| `get_graph(mol_or_smiles)` | Graph representation | `(node, edge, adj)` |
| `get_descriptors()` | Only descriptors | `Tensor [40]` |
| `get_morgan_fingerprint()` | Morgan fingerprint | `Tensor [2048]` |
| `get_all_features()` | All features with metadata | `Dict` |

## See Also

- [Molecule Graph Features](molecule_graph.md) - Graph representations for GNN
