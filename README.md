# plmol

Unified protein-ligand feature extraction for ML. Converts PDB files, SMILES, and SDF into tensors ready for GNNs, transformers, and 3D models.

## Installation

```bash
pip install -e .

# With surface mesh support
pip install -e ".[surface]"

# With dev tools (pytest)
pip install -e ".[dev]"
```

**Requirements**: Python >= 3.9, PyTorch, RDKit, NumPy, SciPy, FreeSASA, Pandas

## Quick Start

```python
from plmol import Protein, Ligand, Complex

# Protein
protein = Protein.from_pdb("protein.pdb")
result = protein.featurize(mode="all")
# result.keys() -> ['sequence', 'graph', 'surface', 'backbone']

# Ligand
ligand = Ligand.from_smiles("CCO")
result = ligand.featurize(mode=["graph", "fingerprint"])
# result.keys() -> ['graph', 'fingerprint']

# Complex (protein + ligand + interactions)
cx = Complex.from_files("protein.pdb", "ligand.sdf")
result = cx.featurize(requests="all")
# result.keys() -> ['ligand', 'protein', 'interaction']
```

## Feature Overview

| Component | Modes | Key Outputs |
|-----------|-------|-------------|
| **Protein** | `graph` (residue/atom), `backbone`, `surface`, `voxel`, `sequence` | Residue graph (81-dim nodes, 39-dim edges), atom graph (187 tokens), SE(3)-invariant backbone, MaSIF surface mesh |
| **Ligand** | `graph`, `fingerprint`, `surface`, `voxel`, `smiles` | Dense adjacency (N, N, 37), node features (N, 98), 62-dim descriptors, ECFP4/6, MACCS, ErG |
| **Interaction** | pharmacophore, contact | Bipartite edges (E, 74), 7 interaction types (H-bond, hydrophobic, pi-stacking, salt bridge, ...) |

All graph modes support `distance_cutoff` and `knn_cutoff` (union strategy) for flexible edge construction.

## Batch Processing

```bash
plmol-batch-protein-featurize --input_dir pdbs/ --output_dir features/
plmol-batch-ligand-featurize --input_dir sdfs/ --output_dir features/
```

## Documentation

Detailed API reference with feature dimensions, index ranges, and parameters:

- [Protein API](docs/protein.md) — graph (residue/atom), backbone, surface, voxel, sequence, ESM embeddings
- [Ligand API](docs/ligand.md) — graph, fingerprint, surface, voxel
- [Complex API](docs/complex.md) — interaction detection, contact edges, pocket extraction

## License

MIT
