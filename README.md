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

## Protein Features

### Residue Graph

```python
result = protein.featurize(mode="graph", graph_kwargs={"level": "residue", "distance_cutoff": 8.0})
graph = result["graph"]
```

| Output | Shape | Description |
|--------|-------|-------------|
| `node_features` | tuple of 8 | Scalar features (81-dim): residue type, terminal flags, geometry, dihedrals, chi angles, SASA, neighbor distances, physicochemical properties |
| `node_vector_features` | tuple of 3 | Vector features (31 x 3): intra-residue vectors, neighbor directions, local frames |
| `edge_index` | `(2, E)` | Sparse edges (CA/SC distance < cutoff) |
| `edge_features` | tuple of 2 | Scalar (39-dim): distances + sequence separation |
| `edge_vector_features` | tuple of 1 | Vector (8 x 3): directional features |
| `coords` | `(L, 2, 3)` | CA + sidechain centroid |

### Atom Graph

```python
result = protein.featurize(mode="graph", graph_kwargs={"level": "atom", "distance_cutoff": 4.0})
graph = result["graph"]
```

| Output | Description |
|--------|-------------|
| `atom_tokens` | Residue-atom pair token (187 vocab, for `nn.Embedding`) |
| `residue_token` | Residue type (22 vocab) |
| `atom_element` | Element type (19 vocab) |
| Scalar features (11-dim) | SASA, relative SASA, B-factor, B-factor z-score, is_backbone, formal_charge, H-bond donor/acceptor, secondary structure |
| Edge features (6-dim) | Distance, same_residue, sequence_separation, unit_vector |

### Backbone (Inverse Folding)

For ProteinMPNN / ESM-IF / GVP / PiFold style models.

```python
result = protein.featurize(mode="backbone", backbone_kwargs={"k_neighbors": 30})
backbone = result["backbone"]
```

| Output | Shape | Description |
|--------|-------|-------------|
| `backbone_coords` | `(L, 4, 3)` | N, CA, C, O positions |
| `cb_coords` | `(L, 3)` | Virtual CB (ProteinMPNN geometry) |
| `dihedrals` | `(L, 3)` | phi, psi, omega (radians) |
| `dihedrals_sincos` | `(L, 6)` | sin/cos encoding of dihedrals |
| `orientation_frames` | `(L, 3, 3)` | N-CA-C local frames |
| `edge_index` | `(2, E)` | kNN graph over CA atoms |
| `edge_rbf` | `(E, 16)` | Gaussian RBF distance encoding |
| `edge_local_pos` | `(E, 3)` | SE(3)-invariant neighbor position |
| `edge_rel_orient` | `(E, 3, 3)` | Relative rotation between frames |

### Surface & Sequence

```python
# Surface mesh (MaSIF-style features)
result = protein.featurize(mode="surface")
# -> points, faces, normals, hydropathy, charge, curvature, shape_index

# Amino acid sequence
result = protein.featurize(mode="sequence")
# -> str or Dict[str, str] for multi-chain
```

### Pocket Featurization

```python
pocket = protein.featurize_pocket(ligand="ligand.sdf", cutoff=6.0, mode="graph")
```

## Ligand Features

### Graph

```python
result = ligand.featurize(mode="graph")
graph = result["graph"]
```

| Output | Shape | Description |
|--------|-------|-------------|
| `node_features` | `(N, 98)` | Per-atom features: atom type, charge, hybridization, stereochemistry, physical properties, topological indices, SMARTS patterns, neighborhood, Crippen, TPSA, Labute ASA |
| `adjacency` | `(N, N, 37)` | Dense adjacency: bond features [0:27] + pair features [27:37] |
| `bond_mask` | `(N, N)` | True where chemical bond exists |
| `distance_matrix` | `(N, N)` | 3D Euclidean distances |
| `coords` | `(N, 3)` | 3D coordinates |

Sparse conversion: `LigandFeaturizer.adjacency_to_bond_edges(adjacency) -> (edge_index, edge_features)`

### Fingerprints & Descriptors

```python
result = ligand.featurize(mode="fingerprint")
fp = result["fingerprint"]
```

| Key | Dim | Description |
|-----|-----|-------------|
| `descriptors` | 62 | Normalized molecular descriptors (drug-likeness, ADMET, 3D shape, ...) |
| `ecfp4` | 2048 | Morgan radius=2, chirality-aware |
| `ecfp6` | 2048 | Morgan radius=3, chirality-aware |
| `maccs` | 167 | MACCS structural keys |
| `rdkit` | 2048 | RDKit path-based fingerprint |
| `atom_pair` | 2048 | Atom pair fingerprint |
| `erg` | 315 | ErG pharmacophore fingerprint |

Additional FPs available: `ecfp4_count`, `ecfp4_feature`, `ecfp6_feature`, `pharmacophore2d`, `avalon`, `peoe_vsa`, `slogp_vsa`, `smr_vsa`, `mqn`

## Interaction Features

```python
cx = Complex.from_files("protein.pdb", "ligand.sdf")
result = cx.featurize(requests="interaction")
interaction = result["interaction"]
```

| Output | Shape | Description |
|--------|-------|-------------|
| `edges` | `(2, E)` | Protein-ligand heavy atom pairs |
| `edge_features` | `(E, 74)` | Interaction type (7), geometry (4), element types (20), hybridization (12), charges (2), aromatic (2), ring/degree (4), residue type (21), backbone (1), strength (1) |
| `contact_edges` | `(2, E_c)` | All atom pairs within cutoff (optional, via `include_contacts=True`) |

**Interaction types**: hydrogen bond, salt bridge, pi-stacking, cation-pi, hydrophobic, halogen bond, metal coordination

## ESM Embeddings

```python
from plmol import ESMFeaturizer

esm = ESMFeaturizer(model_type="esmc", model_name="esmc_600m", device="cuda")
embeddings = esm.extract("MKFLIL...")
# embeddings["embeddings"]: (L, 1152)
```

| Model | Embedding Dim |
|-------|--------------|
| `esmc_300m` | 960 |
| `esmc_600m` | 1152 |
| `esm3-open` | 1536 |

## Batch Processing

```bash
plmol-batch-protein-featurize --input_dir pdbs/ --output_dir features/
plmol-batch-ligand-featurize --input_dir sdfs/ --output_dir features/
```

## Project Structure

```
plmol/
├── protein/           # Protein featurizers (residue, atom, backbone, surface, ESM)
├── ligand/            # Ligand featurizers (graph, descriptors, fingerprints)
├── interaction/       # PLI featurizer + pocket extraction
├── constants/         # Domain constants (amino acids, elements, SMARTS, interactions)
├── featurizers/       # Shared surface/voxel builders
├── io/                # File loaders
└── cli/               # Batch processing scripts
```

## Documentation

See [docs/api_reference.md](docs/api_reference.md) for full feature dimensions, index ranges, and API details.

## License

MIT
