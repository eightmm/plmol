# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is plmol?

A unified multi-molecule feature extraction toolkit for ML. Converts PDB/mmCIF files, SMILES, and RNA/DNA sequences into tensors (graphs, fingerprints, fragments, surfaces, voxels) ready for GNNs, transformers, and 3D CNNs. Supports proteins, ligands, nucleic acids, and arbitrary multi-molecule complexes.

## Commands

```bash
# Install (editable, with dev deps)
uv sync

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_complex.py::test_complex_ligand_features_from_sequence_protein -v

# Batch featurization CLI
plmol-batch-protein-featurize --input_dir data/ --output_dir outputs/
```

## Architecture

### Multi-View Molecule Hierarchy

```
BaseMolecule (base.py) — abstract: sequence, graph, coords, surface
├── Protein (protein/core.py) — from_pdb(), from_structure(), featurize()
├── Ligand  (ligand/core.py)  — from_smiles(), from_sdf(), featurize()
└── NucleicAcid (nucleic_acid/core.py) — from_pdb(), from_sequence(), featurize()
```

`MolecularComplex` (complex.py) wraps arbitrary molecule combinations with a unified `.featurize(requests=["protein","ligand","nucleic_acid","interaction"])` API. Uses `LRUCache` and `FeatureSpec` contracts (specs.py) for validation. Cache is invalidated on molecule mutation (e.g., SMILES/sequence reassignment).

### Protein Pipeline

`Protein.featurize(mode=...)` delegates to `ProteinFeaturizer` (protein/featurizer.py), which is the central orchestrator. It parses PDB/mmCIF once, caches results, and dispatches to specialized featurizers:

| Mode | Featurizer | Output |
|------|-----------|--------|
| `graph` (level=residue) | `ResidueFeaturizer` → `get_features()` | Scalar/vector node+edge tuples |
| `graph` (level=atom) | `AtomFeaturizer` → `get_atom_graph()` | Token-based node dict + edge dict |
| `backbone` | `backbone_featurizer` → `compute_backbone_features()` | SE(3)-invariant kNN graph |
| `surface` | `surface/orchestrator.py` → `build_protein_surface()` | dMaSIF point cloud with PCA curvature features |
| `voxel` | `voxel/` → `build_protein_voxel()` | 16-channel 3D grid |
| `sequence` | Direct from parser | Amino acid string |

Key internal flow: PDB/mmCIF file → `PDBStandardizer` → `StructureParser` (parsers/, cached) → featurizers. All parsers (PDBParser, MMCIFParser) conform to `StructureParser` ABC.

**SASA-based features**: Residue SASA is normalized by per-residue `RESIDUE_MAX_SASA` (Tien et al. 2013), not arbitrary constants. Atom-level features use FreeSASA to compute `burial_index` (1 - relative SASA) and `is_polar_sasa` (polar vs apolar classification). These replace the former b_factor-based features throughout the codebase (residue graph, atom graph, surface, voxel).

### Parsers

`StructureParser` (parsers/base.py) is the unified interface for all structure file parsers. Implementations:

- **`PDBParser`** (parsers/pdb_parser.py): Parses PDB files, extracts protein atoms (no waters/hydrogens), nucleic acids, metals.
- **`MMCIFParser`** (parsers/mmcif_parser.py): Parses mmCIF files with equivalent functionality.

All parsers return `ParsedAtom` dataclass (single source of truth for atom data) and expose properties: `protein_atoms`, `all_atoms`, `file_path`, `get_sequence()`, `get_sequence_by_chain()`, `get_atom_coords()`.

**NOTE**: `protein/utils.py` is now a re-export shim for backward compatibility — all parser logic moved to `parsers/` package.

### Ligand Pipeline

`Ligand.featurize(mode=...)` delegates to `LigandFeaturizer` which wraps specialized classes:

- **`MoleculeFeaturizer`** (ligand/descriptors.py): Descriptors (62-dim) and fingerprints via `FingerprintGenerator` (ligand/fingerprint_generator.py): ECFP4/6, MACCS, RDKit, ERG, + optional VSA/MQN
- **`MoleculeGraphFeaturizer`** (ligand/graph.py): Orchestrator that composes `AtomFeatureMixin` (ligand/graph_atom_features.py) and `EdgeFeatureMixin` (ligand/graph_edge_features.py). Dense adjacency `(N, N, 37)` with node features `(N, 98)`, coords, distance matrix
- **`fragment_on_rotatable_bonds`** (ligand/fragment.py): Cuts molecule at rotatable bonds → fragment SMILES, atom-to-fragment mapping, fragment adjacency matrix

The graph uses **dense adjacency** (not sparse edge_index). Channels [0:27] = bond features, [27:37] = 3D pair features.

### Nucleic Acid Pipeline

`NucleicAcid.featurize(mode=...)` delegates to `NucleicFeaturizer` (nucleic_acid/featurizer.py), which extracts DNA/RNA features:

| Mode | Output |
|------|--------|
| `sequence` | One-letter sequence (A, G, C, T/U) with GC content, purine/pyrimidine info |
| `graph` (level=residue) | Residue-level graph with 7 backbone torsion angles (alpha, beta, gamma, delta, epsilon, zeta, chi) and base properties |
| `backbone` | Sugar-phosphate geometry features |
| `atom_graph` | Atom-level graph with token-based features (token IDs from `NUCLEOTIDE_TOKEN`) |

Nucleotide SASA is normalized by `NUCLEOTIDE_MAX_SASA`. Automatically detects DNA vs RNA from residue names (DA/DG/DC/DT vs A/G/C/U).

### Interaction Pipeline

`PLInteractionFeaturizer` (interaction/pli_featurizer.py) is the orchestrator that delegates to:

- **`pli_detectors.py`**: Detects protein-ligand interactions (H-bonds, hydrophobic, pi-stacking, salt bridges, etc.)
- **`pli_encoding.py`**: Builds edge features and interaction graphs from detected interactions
- **`pocket_extractor.py`**: Extracts binding-site residues by distance cutoff
- **`metal_coordination.py`**: Detects and classifies `MetalSite` instances (coordination geometry, donor atoms, distances)

Bipartite interaction graph edges have 79-dim features (7 interaction type one-hot, geometric features, element/hybridization encoding, residue context, cross-contact density, endpoint min distance, relative pocket distance).

### Surface Module

The surface module is split into 7 focused files (no monolithic `features.py`):

- **`orchestrator.py`**: Entry point `build_protein_surface()` that delegates to geometry, chemistry, and type-feature extraction
- **`point_cloud.py`**: dMaSIF point cloud generation with PCA-computed curvature and geometric features
- **`geometry.py`**: SE(3)-invariant geometric feature calculations (curvature, smoothness, shape index)
- **`chemical.py`**: Chemical properties (hydrophobicity, polarity, charge) mapped to surface points
- **`type_features.py`**: Atom/residue type information for surface points
- **`mapping.py`**: Point-cloud-to-protein mapping utilities
- **`_protein_adapter.py`**: Internal adapter for Protein objects to surface featurizers

### Shared Utilities

- **`rdkit_utils.py`**: Centralized RDKit molecule preparation (`prepare_mol`, `canonicalize_mol`, `ensure_3d_conformer`, `has_3d`, `get_positions`). Used by ligand core, graph, descriptors, and interaction modules.

### Constants

All domain constants are centralized in `constants/` and re-exported from `constants/__init__.py`. Submodules: `amino_acids` (tokens, residue-atom mappings, `RESIDUE_MAX_SASA`), `nucleic_acids` (nucleotide tokens, DNA/RNA residues, `NUCLEOTIDE_MAX_SASA`), `elements` (element types), `smarts_patterns`, `interactions`, `physical_properties`, `runtime` (defaults).

### Error Handling

Custom hierarchy in `errors.py`: `PlmolError` → `InputError` (bad user input), `DependencyError` (missing optional dep), `FeatureError` (runtime extraction failure). Follow fail-fast principle — no silent fallbacks in core featurization paths.

## Key Conventions

- **Heavy atoms only**: Protein/ligand/nucleic-acid graphs exclude hydrogens. H-bond info is encoded via interaction features.
- **Lazy init + caching**: `ProteinFeaturizer`, `NucleicFeaturizer`, and `StructureParser` cache parsed data. Features are computed on first access.
- **Token-based atom graphs**: Protein/nucleic-acid atom-level graphs use integer token IDs (187 for protein, ~14 for nucleotides, designed for `nn.Embedding`). Atom features include `burial_index` and `is_polar_sasa` (FreeSASA-based).
- **Residue/nucleotide graphs use tuples**: `get_features()` returns `(node_dict, edge_dict)` where values contain tuples of tensors (scalar_features, vector_features).
- **PDB standardization**: Enabled by default. Normalizes residue names (HIS variants → HIS, modified residues → standard), removes waters/hydrogens, preserves metals/nucleic acids.
- **Parser interface is unified**: All modules use `StructureParser` ABC (parsers/base.py). Implementations (PDBParser, MMCIFParser) return `ParsedAtom` dataclass as single source of truth.
- **Feature dimension docs**: Detailed dimension breakdowns with index ranges are in `docs/protein.md`, `docs/ligand.md`, `docs/complex.md`, `docs/nucleic_acid.md`.
- **Bidirectional atom_to_X mappings**: Hierarchical groupings provide both forward (`atom_to_X`: `(A,)` int, atom→group) and reverse (`X_atom_indices`: `List[List[int]]`, group→atoms). Protein has `atom_to_residue`/`residue_atom_indices`; ligand has `atom_to_fragment`/`fragment_atom_indices` plus `molecule_features` (62-dim) for 3-level hierarchy.
- **RDKit utilities are centralized**: All molecule preparation, conformer generation, and 3D coordinate checks go through `rdkit_utils.py`. Ligand's `smiles` setter rebuilds `_rdmol`; `generate_conformer` preserves explicit-H count; `has_3d` checks `Is3D()`.
- **Error handling is consistent**: Custom hierarchy (`InputError`, `DependencyError`, `FeatureError`) used throughout. No silent fallbacks in core featurization paths.

## Dependencies

Core: `torch`, `rdkit`, `numpy`, `pandas`, `scipy`, `freesasa`
Optional: `biopython` (for mmCIF parsing)
