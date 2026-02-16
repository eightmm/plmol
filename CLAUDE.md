# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is plmol?

A unified protein-ligand feature extraction toolkit for ML. Converts PDB files and SMILES into tensors (graphs, fingerprints, surfaces, voxels) ready for GNNs, transformers, and 3D CNNs.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_complex.py::test_complex_ligand_features_from_sequence_protein -v

# Batch featurization CLI
plmol-batch-protein-featurize --input_dir data/ --output_dir outputs/
plmol-batch-ligand-featurize --input_dir data/ --output_dir outputs/
```

## Architecture

### Multi-View Molecule Hierarchy

```
BaseMolecule (base.py) — abstract: sequence, graph, coords, surface
├── Protein (protein/core.py) — from_pdb(), from_sequence(), featurize()
└── Ligand  (ligand/core.py)  — from_smiles(), from_sdf(), featurize()
```

`Complex` (complex.py) wraps both with a unified `.featurize(requests=["ligand","protein","interaction"])` API. Uses `LRUCache` and `FeatureSpec` contracts (specs.py) for validation.

### Protein Pipeline

`Protein.featurize(mode=...)` delegates to `ProteinFeaturizer` (protein/protein_featurizer.py), which is the central orchestrator. It parses PDB once, caches results, and dispatches to specialized featurizers:

| Mode | Featurizer | Output |
|------|-----------|--------|
| `graph` (level=residue) | `ResidueFeaturizer` → `get_features()` | Scalar/vector node+edge tuples |
| `graph` (level=atom) | `AtomFeaturizer` → `get_atom_graph()` | Token-based node dict + edge dict |
| `backbone` | `backbone_featurizer` → `compute_backbone_features()` | SE(3)-invariant kNN graph |
| `surface` | `featurizers/surface.py` → `build_protein_surface()` | Mesh or point cloud with MaSIF features |
| `voxel` | `featurizers/voxel.py` → `build_protein_voxel()` | 16-channel 3D grid |
| `sequence` | Direct from parser | Amino acid string |

Key internal flow: PDB file → `PDBStandardizer` → `PDBParser` (utils.py, cached) → featurizers.

### Ligand Pipeline

`Ligand.featurize(mode=...)` delegates to `LigandFeaturizer` which wraps two core classes:

- **`MoleculeFeaturizer`** (ligand/base.py): Descriptors (62-dim) and fingerprints (ECFP4/6, MACCS, RDKit, ERG, + optional VSA/MQN)
- **`MoleculeGraphFeaturizer`** (ligand/graph.py): Dense adjacency `(N, N, 37)` with node features `(N, 78)`, coords, distance matrix

The graph uses **dense adjacency** (not sparse edge_index). Channels [0:27] = bond features, [27:37] = 3D pair features.

### Interaction Pipeline

`PLInteractionFeaturizer` (interaction/pli_featurizer.py) detects protein-ligand interactions (H-bonds, hydrophobic, pi-stacking, etc.) and builds a bipartite interaction graph. `extract_pocket()` selects binding-site residues by distance cutoff.

### Constants

All domain constants are centralized in `constants/` and re-exported from `constants/__init__.py`. Submodules: `amino_acids` (tokens, residue-atom mappings), `elements` (element types), `smarts_patterns`, `interactions`, `physical_properties`, `runtime` (defaults).

### Error Handling

Custom hierarchy in `errors.py`: `PlmolError` → `InputError` (bad user input), `DependencyError` (missing optional dep), `FeatureError` (runtime extraction failure). Follow fail-fast principle — no silent fallbacks in core featurization paths.

## Key Conventions

- **Heavy atoms only**: Protein/ligand graphs exclude hydrogens. H-bond info is encoded via lookup tables and interaction features.
- **Lazy init + caching**: `ProteinFeaturizer` and `PDBParser` cache parsed data. Features are computed on first access.
- **Token-based atom graph**: Protein atom-level graph uses integer token IDs (187 classes from `RESIDUE_ATOM_TOKEN`) designed for `nn.Embedding`, not one-hot.
- **Residue graph uses tuples**: `get_features()` returns `(node_dict, edge_dict)` where values contain tuples of tensors (scalar_features, vector_features).
- **PDB standardization**: Enabled by default. Normalizes residue names (HIS variants → HIS, modified residues → standard), removes waters/metals/ligands.
- **PDB parsing is centralized**: All modules use `PDBParser` and `parse_pdb_line()` from `protein/utils.py`. The `ParsedAtom` dataclass is the single source of truth for atom data.
- **Feature dimension docs**: Detailed dimension breakdowns with index ranges are in `docs/api_reference.md`.

## Dependencies

Core: `torch`, `rdkit`, `numpy`, `pandas`, `scipy`, `freesasa`
Optional: `open3d`, `scikit-image`, `trimesh` (for surface mode)
