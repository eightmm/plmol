# Hierarchical Protein Featurizer

Hierarchical feature extraction for atom-residue attention models with ESM embeddings.

## Overview

The `HierarchicalFeaturizer` extracts multi-level protein features designed for hierarchical attention mechanisms:
- **Atom-level**: Integer token indices for efficient embedding lookup
- **Residue-level**: Scalar and vector features from local geometry
- **ESM embeddings**: Pre-trained language model embeddings (ESMC + ESM3)

## Quick Start

```python
from plfeature.protein_featurizer import HierarchicalFeaturizer

# Initialize (loads ESM models ~2-3s)
featurizer = HierarchicalFeaturizer()

# Extract features
data = featurizer.featurize("protein.pdb")

# Access features (integer indices for nn.Embedding)
print(data.atom_tokens.shape)        # [N_atom] - indices 0-186
print(data.residue_features.shape)   # [N_res, 76]
print(data.esmc_embeddings.shape)    # [N_res, 1152]

# Convert to one-hot in model if needed:
# atom_tokens_onehot = F.one_hot(data.atom_tokens, num_classes=187)
```

## Architecture

```
PDB File
    │
    ▼
┌─────────────┐
│  PDBParser  │  ← Single source of truth (parsed once)
└─────────────┘
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    ▼                  ▼                  ▼                  ▼
AtomFeaturizer   ResidueFeaturizer   ESMFeaturizer      FreeSASA
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
atom_tokens      residue_features    esm_embeddings    atom_sasa
atom_coords      residue_vectors     bos/eos tokens
atom_elements
```

## Feature Dimensions

### Atom-level Features (Integer Indices)

| Feature | Shape | Description |
|---------|-------|-------------|
| `atom_tokens` | [N_atom] | Atom token indices (0-186, 187 classes) |
| `atom_coords` | [N_atom, 3] | 3D coordinates (raw) |
| `atom_sasa` | [N_atom] | Solvent accessible surface area (normalized /100) |
| `atom_elements` | [N_atom] | Element type indices (0-7, 8 classes) |
| `atom_residue_types` | [N_atom] | Residue type indices (0-21, 22 classes) |

**Note**: Categorical features are stored as integer indices for memory efficiency.
Use `torch.nn.Embedding` or `F.one_hot()` in your model as needed.

### Residue-level Features

| Feature | Shape | Description |
|---------|-------|-------------|
| `residue_features` | [N_res, 76] | Scalar features (see breakdown below) |
| `residue_vector_features` | [N_res, 31, 3] | Vector features for SE(3) equivariance |
| `residue_ca_coords` | [N_res, 3] | CA atom coordinates |
| `residue_sc_coords` | [N_res, 3] | Sidechain centroid coordinates |

**Scalar Features (76-dim) Breakdown:**
- Residue type one-hot: 21
- Terminal flags (N/C): 2
- Self distances: 10
- Dihedral features (phi/psi/omega): 20
- Chi angle flags: 5
- SASA features: 10
- Forward/reverse distances: 8

**Vector Features (31x3) Breakdown:**
- Self vectors: [20, 3]
- Reference frame vectors: [8, 3]
- Local coordinate frames: [3, 3]

### ESM Embeddings (6 Tensors)

| Feature | Shape | Description |
|---------|-------|-------------|
| `esmc_embeddings` | [N_res, 1152] | ESMC per-residue embeddings |
| `esmc_bos` | [1152] | ESMC BOS (beginning of sequence) token |
| `esmc_eos` | [1152] | ESMC EOS (end of sequence) token |
| `esm3_embeddings` | [N_res, 1536] | ESM3 per-residue embeddings |
| `esm3_bos` | [1536] | ESM3 BOS token |
| `esm3_eos` | [1536] | ESM3 EOS token |

**ESM Per-Chain Extraction**: Each chain is processed separately through ESM models,
preserving proper BOS/EOS token handling. Embeddings are then concatenated.

### Atom-Residue Mapping

| Feature | Shape | Description |
|---------|-------|-------------|
| `atom_to_residue` | [N_atom] | Residue index for each atom |
| `residue_atom_indices` | [N_res, max_atoms] | Atom indices per residue |
| `residue_atom_mask` | [N_res, max_atoms] | Valid atom mask |
| `num_atoms_per_residue` | [N_res] | Atom count per residue |

## Usage Examples

### Basic Feature Extraction

```python
from plfeature.protein_featurizer import HierarchicalFeaturizer
import torch.nn.functional as F

featurizer = HierarchicalFeaturizer()
data = featurizer.featurize("protein.pdb")

# Atom features (integer indices)
atom_tokens = data.atom_tokens           # [N_atom] indices
atom_coords = data.atom_coords           # [N_atom, 3]

# Convert to one-hot if needed
atom_tokens_onehot = F.one_hot(atom_tokens, num_classes=187)  # [N_atom, 187]

# Or use nn.Embedding in your model
embedding = torch.nn.Embedding(187, 64)
atom_embeddings = embedding(atom_tokens)  # [N_atom, 64]

# Residue features
res_feats = data.residue_features         # [N_res, 76]
res_vectors = data.residue_vector_features  # [N_res, 31, 3]

# ESM embeddings
esmc = data.esmc_embeddings               # [N_res, 1152]
esm3 = data.esm3_embeddings               # [N_res, 1536]

# Special tokens for sequence-level representation
esmc_seq = torch.cat([data.esmc_bos.unsqueeze(0),
                      data.esmc_embeddings,
                      data.esmc_eos.unsqueeze(0)])  # [N_res+2, 1152]
```

### Pocket Extraction

```python
from rdkit import Chem

# Load ligand
ligand = Chem.MolFromMolFile("ligand.sdf")

# Extract pocket features (6.0 Å cutoff)
pocket_data = featurizer.featurize_pocket("protein.pdb", ligand, cutoff=6.0)
```

### Residue Subset Selection

```python
# Select specific residues (e.g., binding site)
binding_residues = [10, 11, 12, 45, 46, 47, 100, 101]
subset = data.select_residues(binding_residues)

# Subset maintains all features
print(subset.atom_tokens.shape)        # Atoms from selected residues only
print(subset.residue_features.shape)   # [8, 76]
print(subset.esmc_embeddings.shape)    # [8, 1152]
print(subset.esmc_bos.shape)           # [1152] - original BOS preserved
```

### Move to GPU

```python
# Move all tensors to GPU
data_gpu = data.to(torch.device('cuda'))
```

### Batch Processing

```python
from plfeature.protein_featurizer import HierarchicalFeaturizer
import torch

featurizer = HierarchicalFeaturizer()

# Process multiple PDB files
pdb_files = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
all_data = [featurizer.featurize(pdb) for pdb in pdb_files]

# Access feature dimensions
for data in all_data:
    dims = data.get_feature_dims()
    print(f"Atoms: {dims['num_atoms']}, Residues: {dims['num_residues']}")
    print(f"  Atom classes: {dims['num_atom_classes']}")      # 187
    print(f"  Element classes: {dims['num_element_classes']}") # 8
    print(f"  Residue classes: {dims['num_residue_classes']}") # 22
```

## Model Configuration

```python
# Custom ESM models
featurizer = HierarchicalFeaturizer(
    esmc_model="esmc_600m",    # or "esmc_300m"
    esm3_model="esm3-open",
    esm_device="cuda",         # or "cpu"
)
```

**Available ESM Models:**

| Model | Embedding Dim | Parameters |
|-------|---------------|------------|
| esmc_300m | 960 | 300M |
| esmc_600m | 1152 | 600M |
| esm3-open | 1536 | Open weights |

## Integration with PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, pdb_files):
        self.featurizer = HierarchicalFeaturizer()
        self.pdb_files = pdb_files

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        data = self.featurizer.featurize(self.pdb_files[idx])
        return {
            'atom_tokens': data.atom_tokens,           # [N_atom] int
            'atom_elements': data.atom_elements,       # [N_atom] int
            'atom_coords': data.atom_coords,           # [N_atom, 3]
            'residue_features': data.residue_features, # [N_res, 76]
            'esmc_embeddings': data.esmc_embeddings,   # [N_res, 1152]
            'esm3_embeddings': data.esm3_embeddings,   # [N_res, 1536]
            'atom_to_residue': data.atom_to_residue,   # [N_atom]
        }


class HierarchicalProteinEncoder(nn.Module):
    """Example model using integer indices with nn.Embedding."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        # Embedding layers for categorical features
        self.atom_token_emb = nn.Embedding(187, hidden_dim)
        self.element_emb = nn.Embedding(8, hidden_dim // 4)
        self.residue_type_emb = nn.Embedding(22, hidden_dim // 4)

        # Projection layers
        self.coord_proj = nn.Linear(3, hidden_dim)
        self.residue_proj = nn.Linear(76, hidden_dim)

    def forward(self, data):
        # Atom-level: use embeddings
        atom_emb = self.atom_token_emb(data['atom_tokens'])
        elem_emb = self.element_emb(data['atom_elements'])
        coord_emb = self.coord_proj(data['atom_coords'])

        atom_features = atom_emb + coord_emb

        # Residue-level
        res_features = self.residue_proj(data['residue_features'])

        return atom_features, res_features
```

## Performance

| Operation | Time |
|-----------|------|
| Model initialization | ~2-3s |
| Feature extraction (per protein) | ~0.9-1.0s |
| Throughput | ~1.2 proteins/sec |

**Memory estimate per protein (~400 residues):**
- Atom features: ~1.5 MB (reduced from ~2.5 MB with one-hot)
- Residue features: ~0.5 MB
- ESM embeddings: ~2.0 MB
- Total: ~4 MB per protein

## Internal Architecture

The `HierarchicalFeaturizer` uses a **single PDBParser** instance to parse the PDB file once,
then shares the parsed data with all sub-featurizers:

```python
# Internal flow (simplified)
def featurize(self, pdb_path):
    # Step 1: Parse PDB once
    pdb_parser = PDBParser(pdb_path)

    # Step 2: All featurizers use pre-parsed data
    atom_tokens, atom_coords = self._atom_featurizer.get_protein_atom_features_from_parser(pdb_parser)
    residue_featurizer = ResidueFeaturizer.from_parser(pdb_parser, pdb_path)
    esm_result = self._esm_featurizer.extract_from_parser(pdb_parser)

    # Step 3: FreeSASA still uses file path (required by library)
    atom_sasa, _ = self._atom_featurizer.get_atom_sasa(pdb_path)
```

## Dependencies

- `torch`: PyTorch for tensor operations
- `freesasa`: SASA calculation
- `esm`: ESM model library (`pip install esm`)

## See Also

- [Protein Atom Features](protein_atom_feature.md) - Atom token definitions
- [Protein Residue Features](protein_residue_feature.md) - Residue feature details
