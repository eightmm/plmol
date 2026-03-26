# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is plmol?

A unified protein-ligand feature extraction toolkit for ML. Converts PDB files and SMILES into tensors (graphs, fingerprints, fragments, surfaces, voxels) ready for GNNs, transformers, and 3D CNNs.

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

`Complex` (complex.py) wraps both with a unified `.featurize(requests=["ligand","protein","interaction"])` API. Uses `LRUCache` and `FeatureSpec` contracts (specs.py) for validation. Cache is invalidated on ligand mutation (e.g., SMILES reassignment).

### Protein Pipeline

`Protein.featurize(mode=...)` delegates to `ProteinFeaturizer` (protein/protein_featurizer.py), which is the central orchestrator. It parses PDB once, caches results, and dispatches to specialized featurizers:

| Mode | Featurizer | Output |
|------|-----------|--------|
| `graph` (level=residue) | `ResidueFeaturizer` → `get_features()` | Scalar/vector node+edge tuples |
| `graph` (level=atom) | `AtomFeaturizer` → `get_atom_graph()` | Token-based node dict + edge dict |
| `backbone` | `backbone_featurizer` → `compute_backbone_features()` | SE(3)-invariant kNN graph |
| `surface` | `surface/` → `build_protein_surface()` | dMaSIF point cloud with PCA curvature features |
| `voxel` | `voxel/` → `build_protein_voxel()` | 16-channel 3D grid |
| `sequence` | Direct from parser | Amino acid string |

Key internal flow: PDB file → `PDBStandardizer` → `PDBParser` (utils.py, cached) → featurizers.

**SASA-based features**: Residue SASA is normalized by per-residue `RESIDUE_MAX_SASA` (Tien et al. 2013), not arbitrary constants. Atom-level features use FreeSASA to compute `burial_index` (1 - relative SASA) and `is_polar_sasa` (polar vs apolar classification). These replace the former b_factor-based features throughout the codebase (residue graph, atom graph, surface, voxel).

### Ligand Pipeline

`Ligand.featurize(mode=...)` delegates to `LigandFeaturizer` which wraps specialized classes:

- **`MoleculeFeaturizer`** (ligand/descriptors.py): Descriptors (62-dim) and fingerprints via `FingerprintGenerator` (ligand/fingerprint_generator.py): ECFP4/6, MACCS, RDKit, ERG, + optional VSA/MQN
- **`MoleculeGraphFeaturizer`** (ligand/graph.py): Orchestrator that composes `AtomFeatureMixin` (ligand/graph_atom_features.py) and `EdgeFeatureMixin` (ligand/graph_edge_features.py). Dense adjacency `(N, N, 37)` with node features `(N, 98)`, coords, distance matrix
- **`fragment_on_rotatable_bonds`** (ligand/fragment.py): Cuts molecule at rotatable bonds → fragment SMILES, atom-to-fragment mapping, fragment adjacency matrix

The graph uses **dense adjacency** (not sparse edge_index). Channels [0:27] = bond features, [27:37] = 3D pair features.

### Interaction Pipeline

`PLInteractionFeaturizer` (interaction/pli_featurizer.py) is the orchestrator that delegates to:

- **`pli_detectors.py`**: Detects protein-ligand interactions (H-bonds, hydrophobic, pi-stacking, salt bridges, metal coordination, etc.)
- **`pli_encoding.py`**: Builds edge features and interaction graphs from detected interactions
- **`pocket_extractor.py`**: Extracts binding-site residues by distance cutoff, preserves metal HETATM records for metal coordination detection

Bipartite interaction graph edges have 79-dim features (7 interaction type one-hot, geometric features, element/hybridization encoding, residue context, cross-contact density, endpoint min distance, relative pocket distance).

### Shared Utilities

- **`rdkit_utils.py`**: Centralized RDKit molecule preparation (`prepare_mol`, `canonicalize_mol`, `ensure_3d_conformer`, `has_3d`, `get_positions`). Used by ligand core, graph, descriptors, and interaction modules.

### Constants

All domain constants are centralized in `constants/` and re-exported from `constants/__init__.py`. Submodules: `amino_acids` (tokens, residue-atom mappings, `RESIDUE_MAX_SASA`), `elements` (element types), `smarts_patterns`, `interactions`, `physical_properties`, `runtime` (defaults).

### Error Handling

Custom hierarchy in `errors.py`: `PlmolError` → `InputError` (bad user input), `DependencyError` (missing optional dep), `FeatureError` (runtime extraction failure). Follow fail-fast principle — no silent fallbacks in core featurization paths.

## Key Conventions

- **Heavy atoms only**: Protein/ligand graphs exclude hydrogens. H-bond info is encoded via lookup tables and interaction features.
- **Lazy init + caching**: `ProteinFeaturizer` and `PDBParser` cache parsed data. Features are computed on first access.
- **Token-based atom graph**: Protein atom-level graph uses integer token IDs (187 classes from `RESIDUE_ATOM_TOKEN`) designed for `nn.Embedding`, not one-hot. Atom features include `burial_index` and `is_polar_sasa` (FreeSASA-based).
- **Residue graph uses tuples**: `get_features()` returns `(node_dict, edge_dict)` where values contain tuples of tensors (scalar_features, vector_features). Residue SASA features are 12-dim (includes burial_index, polar_apolar_ratio).
- **PDB standardization**: Enabled by default. Normalizes residue names (HIS variants → HIS, modified residues → standard), removes waters/metals/ligands.
- **PDB parsing is centralized**: All modules use `PDBParser` and `parse_pdb_line()` from `protein/utils.py`. The `ParsedAtom` dataclass is the single source of truth for atom data.
- **Feature dimension docs**: Detailed dimension breakdowns with index ranges are in `docs/protein.md`, `docs/ligand.md`, `docs/complex.md`.
- **Bidirectional atom_to_X mappings**: Hierarchical groupings provide both forward (`atom_to_X`: `(A,)` int, atom→group) and reverse (`X_atom_indices`: `List[List[int]]`, group→atoms). Protein atom graph has `atom_to_residue`/`residue_atom_indices`; ligand graph has `atom_to_fragment`/`fragment_atom_indices` plus `molecule_features` (62-dim) for a 3-level atom↔fragment↔molecule hierarchy.
- **RDKit utilities are centralized**: All molecule preparation, conformer generation, and 3D coordinate checks go through `rdkit_utils.py`. Ligand's `smiles` setter rebuilds `_rdmol`; `generate_conformer` preserves explicit-H count; `has_3d` checks `Is3D()`.

## Dependencies

Core: `torch`, `rdkit`, `numpy`, `pandas`, `scipy`, `freesasa`
