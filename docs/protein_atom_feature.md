# Protein Atom-Level Features

## Overview

Atom-level feature extraction from protein structures with 187 unique token types and atomic SASA calculation.

## Atom Tokenization System

### Token Mapping

The atom featurizer encodes each atom based on its residue type and atom name, creating 187 unique tokens.

**Token Range:**
- 0-185: Standard residue-atom combinations
- 186: Unknown/non-standard atoms (UNK_TOKEN)

**Example Tokens:**
```python
# Some example mappings
('ALA', 'N'): 0     # Alanine backbone nitrogen
('ALA', 'CA'): 1    # Alanine alpha carbon
('ALA', 'C'): 2     # Alanine backbone carbon
('ALA', 'O'): 3     # Alanine backbone oxygen
('ALA', 'CB'): 4    # Alanine beta carbon
('CYS', 'SG'): 21   # Cysteine sulfur
('TRP', 'CZ2'): 163 # Tryptophan ring carbon
('XXX', 'N'): 180   # Unknown residue backbone N
('XXX', 'CA'): 181  # Unknown residue alpha carbon
```

## Feature Extraction Methods

### AtomFeaturizer Class

The main class for atom-level protein feature extraction.

```python
from plfeature.protein_featurizer import AtomFeaturizer

featurizer = AtomFeaturizer()

# From PDB file
token, coord = featurizer.get_protein_atom_features("protein.pdb")

# From pre-parsed PDBParser (more efficient when PDBParser already available)
from plfeature.protein_featurizer.pdb_utils import PDBParser
parser = PDBParser("protein.pdb")
token, coord = featurizer.get_protein_atom_features_from_parser(parser)
```

### Method Reference

| Method | Description | Returns |
|--------|-------------|---------|
| `get_protein_atom_features(pdb_file)` | Extract tokens and coords from PDB file | `(token, coord)` tensors |
| `get_protein_atom_features_from_parser(parser)` | Extract from pre-parsed PDBParser | `(token, coord)` tensors |
| `get_atom_sasa(pdb_file)` | Calculate SASA using FreeSASA | `(sasa, atom_info)` |
| `get_all_atom_features(pdb_file)` | Get all features including SASA | Dictionary |
| `get_residue_aggregated_features(pdb_file)` | Aggregate atom features to residues | Dictionary |

### 1. Basic Token and Coordinates

```python
from plfeature.protein_featurizer import AtomFeaturizer

featurizer = AtomFeaturizer()
token, coord = featurizer.get_protein_atom_features("protein.pdb")

# Returns:
# token: torch.Tensor [n_atoms] - Atom type tokens (0-186)
# coord: torch.Tensor [n_atoms, 3] - 3D coordinates
```

### 2. From Pre-parsed PDBParser (Recommended for pipelines)

When multiple featurizers need the same PDB data, use a shared PDBParser:

```python
from plfeature.protein_featurizer import AtomFeaturizer
from plfeature.protein_featurizer.pdb_utils import PDBParser

# Parse once
parser = PDBParser("protein.pdb")

# Use with AtomFeaturizer
featurizer = AtomFeaturizer()
token, coord = featurizer.get_protein_atom_features_from_parser(parser)

# Same parser can be used with other featurizers
from plfeature.protein_featurizer import ResidueFeaturizer
res_featurizer = ResidueFeaturizer.from_parser(parser, "protein.pdb")
```

### 3. All Atom Features with SASA

```python
features = featurizer.get_all_atom_features("protein.pdb")

# Returns dictionary:
{
    'token': torch.Tensor,         # [n_atoms] Atom type tokens
    'coord': torch.Tensor,         # [n_atoms, 3] 3D coordinates
    'sasa': torch.Tensor,          # [n_atoms] SASA per atom (Å²)
    'residue_token': torch.Tensor, # [n_atoms] Residue type (0-21)
    'atom_element': torch.Tensor,  # [n_atoms] Element type (0-7)
    'radius': torch.Tensor,        # [n_atoms] Atomic radii
    'metadata': {
        'n_atoms': int,
        'residue_names': list,
        'residue_numbers': tensor,
        'atom_names': list,
        'chain_labels': list
    }
}
```

### 4. Standalone Function Usage

```python
from plfeature.protein_featurizer import get_protein_atom_features

# Basic features
token, coord = get_protein_atom_features("protein.pdb")

# With SASA
from plfeature.protein_featurizer import get_atom_features_with_sasa
features = get_atom_features_with_sasa("protein.pdb")
```

## SASA Calculation

Uses FreeSASA library for solvent accessible surface area calculation.

**Parameters:**
- Algorithm: Lee & Richards
- Probe radius: 1.4 Å (water molecule)
- Resolution: Default FreeSASA parameters

**SASA Values:**
```python
atom_sasa, atom_info = featurizer.get_atom_sasa("protein.pdb")

# atom_sasa: torch.Tensor [n_atoms] - SASA in Å² per atom
# atom_info: Dictionary with atom metadata

# Statistics
total_sasa = atom_sasa.sum()
buried_atoms = (atom_sasa < 0.01).sum()
exposed_atoms = (atom_sasa > 20.0).sum()
```

## Token Distribution

### Residue-Specific Tokens

Each amino acid has specific atom tokens:

```python
# Glycine (smallest): 4 heavy atoms
GLY_tokens = [('GLY', atom) for atom in ['N', 'CA', 'C', 'O']]

# Tryptophan (largest): 14 heavy atoms
TRP_tokens = [('TRP', atom) for atom in [
    'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2',
    'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'
]]

# Unknown residues (XXX): backbone + common atoms
XXX_tokens = [('XXX', atom) for atom in ['N', 'CA', 'C', 'O', 'CB', 'P', 'S', 'SE']]
```

### Token Categories

- **Backbone atoms** (N, CA, C, O): Present in all residues
- **Beta carbon** (CB): Present in all except glycine
- **Sidechain atoms**: Residue-specific
- **Unknown token** (186): For non-standard residue-atom combinations

## Usage Examples

### Basic Atom-Level Analysis

```python
from plfeature.protein_featurizer import AtomFeaturizer
import torch

featurizer = AtomFeaturizer()
token, coord = featurizer.get_protein_atom_features("protein.pdb")

print(f"Number of atoms: {len(token)}")
print(f"Unique atom types: {torch.unique(token).shape[0]}")
print(f"Token range: {token.min()} - {token.max()}")
```

### SASA-Based Analysis

```python
features = featurizer.get_all_atom_features("protein.pdb")

sasa = features['sasa']
tokens = features['token']
residue_tokens = features['residue_token']

# Find exposed atoms
exposed_mask = sasa > 20.0  # Å² threshold
exposed_tokens = tokens[exposed_mask]

print(f"Exposed atoms: {exposed_mask.sum()}/{len(tokens)}")
print(f"Total SASA: {sasa.sum():.2f} Å²")

# Per-residue SASA
for res_type in torch.unique(residue_tokens):
    mask = residue_tokens == res_type
    res_sasa = sasa[mask].sum()
    print(f"Residue type {res_type}: {res_sasa:.2f} Å²")
```

### Efficient Pipeline with Shared Parser

```python
from plfeature.protein_featurizer import AtomFeaturizer, ResidueFeaturizer
from plfeature.protein_featurizer.pdb_utils import PDBParser

# Parse PDB once
parser = PDBParser("protein.pdb")

# Use with multiple featurizers
atom_featurizer = AtomFeaturizer()
token, coord = atom_featurizer.get_protein_atom_features_from_parser(parser)

res_featurizer = ResidueFeaturizer.from_parser(parser, "protein.pdb")
residues = res_featurizer.get_residues()

# Get sequences for ESM
sequences = parser.get_sequence_by_chain()
```

### Integration with Deep Learning

```python
import torch
import torch.nn as nn

class AtomLevelModel(nn.Module):
    def __init__(self, n_tokens=187, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, hidden_dim)
        self.sasa_proj = nn.Linear(1, hidden_dim)
        self.coord_proj = nn.Linear(3, hidden_dim)

    def forward(self, tokens, coords, sasa):
        token_emb = self.embedding(tokens)
        coord_emb = self.coord_proj(coords)
        sasa_emb = self.sasa_proj(sasa.unsqueeze(-1))

        # Combine features
        atom_features = token_emb + coord_emb + sasa_emb
        return atom_features

# Use with extracted features
featurizer = AtomFeaturizer()
features = featurizer.get_all_atom_features("protein.pdb")

model = AtomLevelModel()
atom_features = model(
    features['token'],
    features['coord'],
    features['sasa']
)
```

### Batch Processing

```python
import glob
from plfeature.protein_featurizer import AtomFeaturizer

featurizer = AtomFeaturizer()
pdb_files = glob.glob("pdbs/*.pdb")

all_features = []
for pdb_file in pdb_files:
    features = featurizer.get_all_atom_features(pdb_file)
    all_features.append(features)

# Statistics
total_atoms = sum(f['token'].shape[0] for f in all_features)
avg_sasa = sum(f['sasa'].mean() for f in all_features) / len(all_features)

print(f"Total atoms: {total_atoms}")
print(f"Average SASA per atom: {avg_sasa:.2f} Å²")
```

## Atom Selection and Filtering

### By Element Type

```python
features = featurizer.get_all_atom_features("protein.pdb")
elements = features['atom_element']

# Element type indices: C=0, N=1, O=2, S=3, P=4, Se=5, Metal=6, UNK=7
carbon_mask = elements == 0
carbon_tokens = features['token'][carbon_mask]
carbon_coords = features['coord'][carbon_mask]
```

### By Residue Type

```python
# Select atoms from aromatic residues
aromatic_residues = [5, 19, 9]  # PHE, TYR, TRP
res_tokens = features['residue_token']
aromatic_mask = torch.isin(res_tokens, torch.tensor(aromatic_residues))

aromatic_atoms = features['token'][aromatic_mask]
```

### By SASA Exposure

```python
# Categorize by exposure
sasa = features['sasa']

buried = sasa < 0.01       # Completely buried
partially_exposed = (sasa >= 0.01) & (sasa < 20.0)
exposed = sasa >= 20.0      # Highly exposed

print(f"Buried: {buried.sum()}")
print(f"Partially exposed: {partially_exposed.sum()}")
print(f"Exposed: {exposed.sum()}")
```

## Token Reference Table

Complete mapping available in the source code:

```python
# Access the full token dictionary
from plfeature.protein_featurizer.atom_featurizer import AtomFeaturizer

featurizer = AtomFeaturizer()
token_dict = featurizer.res_atm_token

# Print all mappings
for (res, atom), token in sorted(token_dict.items(), key=lambda x: x[1]):
    print(f"{res:3s} {atom:4s} -> {token:3d}")
```

## Constants

```python
from plfeature.constants import (
    RESIDUE_ATOM_TOKEN,  # Dict[(res, atom)] -> int
    UNK_TOKEN,           # 186 - unknown token
    NUM_ATOM_TOKENS,     # 187 - total token classes
)
```
