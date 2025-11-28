# Protein Residue-Level Features

## Overview

Residue-level feature extraction from protein structures, providing structural features suitable for graph neural networks.

## ResidueFeaturizer Class

### Initialization

```python
from plfeature.protein_featurizer import ResidueFeaturizer

# From PDB file
featurizer = ResidueFeaturizer("protein.pdb")

# From pre-parsed PDBParser (more efficient when PDBParser already available)
from plfeature.protein_featurizer.pdb_utils import PDBParser
parser = PDBParser("protein.pdb")
featurizer = ResidueFeaturizer.from_parser(parser, "protein.pdb")
```

## Main Method: `get_features()`

Returns node and edge features for graph neural networks.

```python
node, edge = featurizer.get_features(distance_cutoff=8.0)  # Default: 8.0 Å
```

### Node Features (First Return Value)

```python
node, edge = featurizer.get_features()

# Node dictionary contains:
node['coord']                     # [n_residues, 2, 3] CA and SC coordinates
node['node_scalar_features']      # [n_residues, n_scalar] Scalar features
node['node_vector_features']      # [n_residues, n_vector, 3] Vector features
```

**Node Scalar Features Include:**
- Residue type encoding (one-hot)
- Backbone dihedral angles (phi, psi, omega)
- SASA values (10 components)
- Secondary structure indicators
- Terminal flags

**Node Vector Features Include:**
- CA-SC direction vectors
- Local frame orientations
- Backbone normal vectors

### Edge Features (Second Return Value)

```python
node, edge = featurizer.get_features()

# Edge dictionary contains:
edge['edges']                     # (2, n_edges) Source and destination indices
edge['edge_scalar_features']      # [n_edges, n_scalar] Scalar edge features
edge['edge_vector_features']      # [n_edges, n_vector, 3] Vector edge features
```

**Edge Scalar Features Include:**
- CA-CA distance
- SC-SC distance
- CA-SC and SC-CA distances
- Sequential distance
- Angle features

**Edge Vector Features Include:**
- CA-CA direction vectors
- SC-SC direction vectors
- Cross-direction vectors

## Alternative Constructors

### from_parser() - Recommended for Pipelines

When working with multiple featurizers on the same PDB:

```python
from plfeature.protein_featurizer import ResidueFeaturizer, AtomFeaturizer
from plfeature.protein_featurizer.pdb_utils import PDBParser

# Parse PDB once
parser = PDBParser("protein.pdb")

# ResidueFeaturizer from parser (avoids re-parsing)
res_featurizer = ResidueFeaturizer.from_parser(parser, "protein.pdb")

# Same parser with AtomFeaturizer
atom_featurizer = AtomFeaturizer()
token, coord = atom_featurizer.get_protein_atom_features_from_parser(parser)

# Get residue features
residues = res_featurizer.get_residues()
node, edge = res_featurizer.get_features()
```

## Individual Feature Methods

### 1. Sequence Features

```python
seq_features = featurizer.get_sequence_features()
# Returns:
{
    'residue_types': torch.Tensor,     # [n_residues] Integer encoding (0-20)
    'residue_one_hot': torch.Tensor,   # [n_residues, 21] One-hot vectors
    'num_residues': int                 # Total number of residues
}
```

### 2. Geometric Features

```python
geo_features = featurizer.get_geometric_features()
# Returns:
{
    'dihedrals': torch.Tensor,         # Backbone and sidechain angles
    'has_chi_angles': torch.Tensor,    # Boolean flags for chi angles
    'backbone_curvature': torch.Tensor,# Local curvature
    'backbone_torsion': torch.Tensor,  # Local torsion
    'self_distances': torch.Tensor,    # Intra-residue distances
    'self_vectors': torch.Tensor,      # Direction vectors
    'coordinates': torch.Tensor        # [n_residues, 15, 3] All atom coords
}
```

**Dihedral Angles:**
- Phi (φ): C(-1) - N - CA - C
- Psi (ψ): N - CA - C - N(+1)
- Omega (ω): CA - C - N(+1) - CA(+1)
- Chi angles (χ1-χ4): Sidechain rotamers

### 3. SASA Features

```python
sasa = featurizer.get_sasa_features()  # [n_residues, 10]
```

**10 SASA Components:**
1. Total SASA
2. Polar SASA
3. Apolar SASA
4. Backbone SASA
5. Sidechain SASA
6. Relative total SASA
7. Relative polar SASA
8. Relative apolar SASA
9. Relative backbone SASA
10. Relative sidechain SASA

### 4. Contact Map

```python
contacts = featurizer.get_contact_map(cutoff=8.0)
# Returns:
{
    'adjacency_matrix': torch.Tensor,  # [n_res, n_res] Binary contacts
    'distance_matrix': torch.Tensor,   # [n_res, n_res] Distances in Å
    'edges': tuple,                    # (src_indices, dst_indices)
    'edge_distances': torch.Tensor,    # Edge distances
    'interaction_vectors': torch.Tensor # Direction vectors
}
```

**Common Cutoffs:**
- 4.5 Å: H-bonds, salt bridges
- 8.0 Å: Standard interactions
- 12.0 Å: Long-range interactions

### 5. Relative Position Encoding

```python
rel_pos = featurizer.get_relative_position(cutoff=32)
# Returns one-hot encoded relative positions
```

### 6. Terminal Flags

```python
terminals = featurizer.get_terminal_flags()
# Returns:
{
    'n_terminal': torch.Tensor,  # Binary flags for N-terminus
    'c_terminal': torch.Tensor   # Binary flags for C-terminus
}
```

### 7. All Features Combined

```python
all_features = featurizer.get_all_features()
# Returns:
{
    'node': dict,  # All node features
    'edge': dict,  # All edge features
    'metadata': {
        'num_residues': int,
        'distance_cutoff': float
    }
}
```

## Usage Examples

### PyTorch Geometric Integration

```python
from torch_geometric.data import Data

node, edge = featurizer.get_features()

data = Data(
    x=node['node_scalar_features'],
    edge_index=edge['edges'],
    edge_attr=edge['edge_scalar_features'],
    pos=node['coord'].reshape(-1, 6)  # Flatten CA and SC coords
)
```

### Efficient Multi-Feature Extraction

```python
# Parse PDB once
featurizer = ResidueFeaturizer("protein.pdb")

# All subsequent calls use cached structure
seq = featurizer.get_sequence_features()
geo = featurizer.get_geometric_features()
sasa = featurizer.get_sasa_features()
contacts = featurizer.get_contact_map(8.0)
```

### Efficient Pipeline with Shared Parser

```python
from plfeature.protein_featurizer import ResidueFeaturizer, AtomFeaturizer
from plfeature.protein_featurizer.pdb_utils import PDBParser
from plfeature.protein_featurizer.esm_featurizer import DualESMFeaturizer

# Single PDBParser for entire pipeline
parser = PDBParser("protein.pdb")

# Residue features
res_featurizer = ResidueFeaturizer.from_parser(parser, "protein.pdb")
node, edge = res_featurizer.get_features()

# Atom features
atom_featurizer = AtomFeaturizer()
token, coord = atom_featurizer.get_protein_atom_features_from_parser(parser)

# ESM embeddings (uses parser's sequence extraction)
esm_featurizer = DualESMFeaturizer()
esm_result = esm_featurizer.extract_from_parser(parser)
```

### PDB Standardization Options

```python
# With standardization (default)
featurizer = ResidueFeaturizer("protein.pdb")  # standardize=True by default

# From parser (parser handles preprocessing)
parser = PDBParser("protein.pdb")
featurizer = ResidueFeaturizer.from_parser(parser, "protein.pdb")
```

## Method Aliases

For clarity, all methods have descriptive aliases:
- `get_features()` → `get_residue_features()`, `get_residue_level_features()`
- `get_sequence_features()` → `get_residue_sequence()`, `get_residue_types()`
- `get_geometric_features()` → `get_residue_geometry()`, `get_residue_dihedrals()`
- `get_sasa_features()` → `get_residue_sasa()`, `get_residue_level_sasa()`
- `get_contact_map()` → `get_residue_contacts()`, `get_residue_contact_map()`

## API Reference

### Constructor Methods

| Method | Description |
|--------|-------------|
| `__init__(pdb_file)` | Create from PDB file path |
| `from_parser(parser, pdb_file)` | Create from pre-parsed PDBParser |

### Feature Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_features(cutoff)` | Node and edge graph features | `(node_dict, edge_dict)` |
| `get_residues()` | List of residue indices | `List[Tuple]` |
| `get_sequence_features()` | Sequence encoding | `Dict` |
| `get_geometric_features()` | Geometric features | `Dict` |
| `get_sasa_features()` | SASA features | `Tensor [n_res, 10]` |
| `get_contact_map(cutoff)` | Contact map | `Dict` |
| `get_terminal_flags()` | N/C terminal flags | `Dict` |
| `get_all_features()` | All features combined | `Dict` |

## See Also

- [Protein Atom Features](protein_atom_feature.md) - Atom-level features
- [Hierarchical Featurizer](hierarchical_featurizer.md) - Combined atom/residue/ESM features
