# plmol

Unified bio-molecule feature extraction for ML. Convert PDB, mmCIF, SMILES, SDF, and sequence strings into tensors ready for GNNs, transformers, and 3D models. Supports proteins, ligands, nucleic acids (DNA/RNA), metal coordination, and arbitrary molecular complexes.

## Installation

```bash
pip install plmol

# With mmCIF support (for mmCIF structure parsing via gemmi)
pip install 'plmol[mmcif]'

# For development
pip install -e ".[dev]"
```

**Requirements**: Python >= 3.9, PyTorch, RDKit, NumPy, SciPy, FreeSASA, Pandas

## Quick Start

### Protein

```python
from plmol import Protein

# From PDB file
protein = Protein.from_pdb("protein.pdb")
result = protein.featurize(mode="all")
# result.keys() -> ['sequence', 'graph', 'surface', 'backbone', 'voxel']

# From sequence string
protein = Protein.from_sequence("MVHLLSPLEEQ")
sequence = protein.sequence
```

### Ligand

```python
from plmol import Ligand

# From SMILES
ligand = Ligand.from_smiles("CCO")
result = ligand.featurize(mode=["graph", "fingerprint"])
# result.keys() -> ['graph', 'fingerprint']

# From SDF file
ligand = Ligand.from_sdf("molecule.sdf")
descriptors = ligand.featurize(mode="descriptor")["descriptor"]["descriptors"]
```

### Nucleic Acid (DNA/RNA)

```python
from plmol import NucleicAcid

# From PDB file (auto-detects DNA/RNA)
dna = NucleicAcid.from_pdb("dna.pdb")
result = dna.featurize(mode="all")
# result.keys() -> ['sequence', 'graph', 'backbone', 'atom_graph']
```

### Arbitrary Molecular Complexes

```python
from plmol import MolecularComplex

# Multi-molecule workflows
cx = MolecularComplex.from_inputs(
    protein="protein.pdb",
    ligand="ligand.sdf",
    nucleic_acid=NucleicAcid.from_pdb("dna.pdb")
)
result = cx.featurize(requests="all")
```

### Protein-Ligand Complex with Interactions

```python
from plmol import Complex

# Traditional protein-ligand binding
cx = Complex.from_files("protein.pdb", "ligand.sdf")
result = cx.featurize(requests="all")
# result.keys() -> ['ligand', 'protein', 'interaction']
```

## Feature Overview

| Component | Input | Modes | Key Outputs |
|-----------|-------|-------|-------------|
| **Protein** | PDB, mmCIF, sequence | `graph` (residue/atom), `atom_graph`, `backbone`, `surface`, `voxel`, `sequence` | Residue graph (12-dim SASA with burial_index), atom graph (187 tokens with burial_index/is_polar_sasa), SE(3)-invariant backbone, dMaSIF point cloud |
| **Ligand** | SMILES, SDF, RDKit Mol | `graph`, `bond_graph`, `fragment_graph`, `fingerprint`, `descriptor`, `fragment`, `surface`, `voxel`, `morgan`, `smiles` | Dense adjacency (N, N, 37), node features (N, 98), bond-wise and fragment-level graphs, 62-dim descriptors, ECFP4/6, MACCS, ErG, rotatable-bond fragments, 16-channel voxel grids |
| **Nucleic Acid** | PDB, mmCIF, sequence | `sequence`, `graph`, `backbone`, `atom_graph` | Nucleotide graph, sugar-phosphate backbone coordinates, atom graph, auto DNA/RNA detection |
| **Interaction** | Protein + Ligand | `graph` | Bipartite edges (E, 79), pharmacophore interactions, optional contact edges, metal coordination |

All graph modes support `distance_cutoff` and `knn_cutoff` (union strategy) for flexible edge construction.

## One Graph Shape for Models

The graph views disagree on how they express edges — the ligand graph is a dense adjacency, the protein atom graph is token ids plus loose per-edge arrays, the rest use `edge_index`. `as_graph` maps any of them onto one shape, `collate` batches them the way PyTorch Geometric does, and `feature_dims` reports widths so models do not hardcode them.

```python
from plmol import Ligand, as_graph, collate, feature_dims

batch = collate([Ligand.from_smiles(s).featurize(mode="graph")["graph"] for s in smiles])
# batch: node_features, edge_index, edge_features, coords, batch, ptr

dims = feature_dims("ligand", "graph")   # {"node_features": 98, "edge_features": 37}
model = MyGNN(dims["node_features"], dims["edge_features"])
```

These are additive; `featurize` still returns exactly what it did.

## Architecture Overview

**plmol** is built on a modular hierarchy:

- **BaseMolecule** (abstract) — sequence, graph, coords, surface
  - **Protein** — PDB/mmCIF parsing, residue/atom graphs, surface, voxel, backbone
  - **Ligand** — SMILES/SDF parsing, molecule graphs, fingerprints, fragments
  - **NucleicAcid** — DNA/RNA parsing, nucleotide graphs, backbone and atom graphs

- **Complex** — protein-ligand binding with interaction detection
- **MolecularComplex** — arbitrary N-molecule workflows with unified featurization API

- **Parsers** — abstraction for PDB, mmCIF (via gemmi), and format-agnostic structure loading
- **Featurizers** — specialized orchestrators (ProteinFeaturizer, LigandFeaturizer, NucleicFeaturizer, PLInteractionFeaturizer)

Features are computed lazily and cached. All APIs follow the same `.featurize(mode=...)` pattern.

## Batch Processing

```bash
plmol-batch-protein-featurize --input-dir pdbs/ --output-dir features/
plmol-batch-ligand-featurize --input-dir sdfs/ --output-dir features/

# Common options
plmol-batch-protein-featurize --input-dir pdbs/ --output-dir features/ --all-pdb --device auto --resume
plmol-batch-ligand-featurize --input-dir ligands/ --output-dir features/ --extensions sdf,mol2 --graph-only
```

Underscore option names such as `--input_dir` are still accepted for compatibility.

## Documentation

Detailed API reference with feature dimensions, index ranges, and parameters:

- [Protein API](docs/protein.md) — graph (residue/atom), backbone, surface, voxel, sequence, ESM embeddings
- [Ligand API](docs/ligand.md) — graph, bond graph, fragment graph, fingerprint, fragment, surface, voxel
- [Nucleic Acid API](docs/nucleic_acid.md) — graph, sequence, backbone, atom graph
- [Complex API](docs/complex.md) — interaction detection, contact edges, pocket extraction
- [Graph View API](docs/graph_view.md) — one graph shape across views, batching, feature dimensions

## Citation

If you use plmol in your research, please cite:

```bibtex
@software{plmol2024,
  title={plmol: Unified Bio-Molecule Feature Extraction Toolkit},
  author={Sim, Jaemin},
  year={2024},
  url={https://github.com/eightmm/plmol}
}
```

## License

MIT
