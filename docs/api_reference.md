# plmol API Reference

> **Version**: 0.2.1
> Protein-Ligand Molecular Feature Extraction Toolkit

## Detailed References

- [Protein API](protein.md) — Initialization, graph (residue/atom), backbone, surface, voxel, sequence, pocket, ESM embeddings, geometry functions
- [Ligand API](ligand.md) — Initialization, graph, fingerprint, fragment, surface, voxel, low-level featurizers
- [Complex API](complex.md) — Initialization, combined featurization, interaction features, contact edges, pocket extraction

## Quick Start

```python
from plmol import Protein, Ligand, Complex

# Protein
protein = Protein.from_pdb("protein.pdb")
result = protein.featurize(mode="all")

# Ligand
ligand = Ligand.from_smiles("CCO")
result = ligand.featurize(mode=["graph", "fingerprint"])

# Complex
cx = Complex.from_files("protein.pdb", "ligand.sdf")
result = cx.featurize(requests="all")
```

## Directory Structure

```
plmol/
├── __init__.py                     # Top-level exports (Protein, Ligand, Complex, ...)
├── base.py                         # BaseMolecule abstract class
├── cache.py                        # LRU caching utility
├── complex.py                      # Complex, MolecularComplex classes
├── errors.py                       # PlmolError, InputError, DependencyError, FeatureError
├── specs.py                        # FeatureSpec, FEATURE_SPECS (PROTEIN/LIGAND/INTERACTION/NUCLEIC_ACID)
├── arrays.py                       # numpy spellings of the torch ops, and to_torch
├── cavity.py                       # LIGSITE cavity detection, no ligand needed
├── utils.py                        # SASA/burial helpers and kNN mask utilities
├── sasa.py                         # Shrake-Rupley SASA; freesasa or native backend
├── spatial.py                      # Neighbour search; sphere occlusion, knn, NeighbourIndex
├── constants/
│   ├── __init__.py                 # Re-exports all constants (amino_acids, nucleic_acids, elements, etc.)
│   ├── amino_acids.py              # Amino acid mappings & tokens (RESIDUE_MAX_SASA)
│   ├── nucleic_acids.py            # Nucleotide mappings, tokens, base atoms, SASA, Watson-Crick pairs
│   ├── elements.py                 # Element types & periodic table
│   ├── interactions.py             # Interaction types, metal coordination constants
│   ├── physical_properties.py      # VdW radius, mass, etc.
│   └── runtime.py                  # Default parameters (cutoffs, grid density)
├── protein/
│   ├── core.py                     # Protein class
│   ├── featurizer.py               # ProteinFeaturizer (parse + cache)
│   ├── residue_featurizer.py       # ResidueFeaturizer (residue features)
│   ├── atom_featurizer.py          # AtomFeaturizer (atom features)
│   ├── pdb_standardizer.py         # PDB standardization
│   ├── backbone_featurizer.py      # Backbone features for inverse folding
│   ├── hierarchical_featurizer.py  # HierarchicalFeaturizer + HierarchicalProteinData
│   ├── esm_featurizer.py           # ESM3/ESMC embedding extraction
│   └── utils.py                    # Utilities (PDB parsing helpers)
├── graph_view.py               # as_graph / collate / feature_dims
├── parsers/
│   ├── __init__.py                 # StructureParser, MMCIFParser, PDBParser, ParsedAtom, ParsedResidue
│   ├── base.py                     # StructureParser (ABC)
│   ├── mmcif_parser.py             # MMCIFParser — parse mmCIF/pdbx format
│   └── pdb_parser.py               # PDBParser — parse PDB format, ParsedAtom/ParsedResidue dataclasses
├── rdkit_utils.py                  # RDKit helper utilities
├── ligand/
│   ├── core.py                     # Ligand class
│   ├── descriptors.py              # MoleculeFeaturizer (descriptors + fingerprints)
│   ├── featurizer.py               # LigandFeaturizer
│   ├── fingerprint_generator.py    # FingerprintGenerator (ECFP, MACCS, RDKit, ERG, etc.)
│   ├── fragment.py                 # Rotatable-bond fragmentation
│   ├── graph.py                    # MoleculeGraphFeaturizer
│   ├── graph_atom_features.py      # AtomFeatureMixin (per-atom node features)
│   └── graph_edge_features.py      # EdgeFeatureMixin (bond/3D pair edge features)
├── nucleic_acid/
│   ├── core.py                     # NucleicAcid class
│   └── featurizer.py               # NucleicFeaturizer
├── interaction/
│   ├── pli_featurizer.py           # PLInteractionFeaturizer
│   ├── pli_detectors.py            # InteractionDetector (H-bond, hydrophobic, pi-stacking, etc.)
│   ├── pli_encoding.py             # InteractionGraphBuilder (bipartite graph encoding)
│   ├── pocket_extractor.py         # Pocket extraction
│   └── metal_coordination.py        # MetalSite, detect_metal_sites, classify_coordination_geometry, encode_metal_features
├── surface/
│   ├── __init__.py                 # Surface public API and builders
│   ├── point_cloud.py              # Surface point generation
│   ├── geometry.py                 # Curvature, normals, and geometric descriptors
│   ├── chemical.py                 # Charge and chemical descriptors
│   ├── mapping.py                  # Atom-to-surface feature mapping
│   ├── type_features.py            # Ligand/protein type descriptors
│   └── orchestrator.py             # Protein and ligand surface feature composition
├── voxel/
│   ├── __init__.py                 # Voxel building (3D grid)
│   └── features.py                 # Voxel feature computation (16-channel grid)
└── io/
    └── loaders.py                  # load_protein_input, load_ligand_input
```

## Core Classes

### Molecules

- **`Protein(pdb_path=None, pdb_id=None, standardize=True, keep_hydrogens=False)`** — Protein structure & featurization. Methods: `from_pdb()`, `from_mmcif()`, `from_structure()`, `from_sequence()`, `featurize(mode, ...)`.

- **`Ligand(rdmol=None)`** — Small molecule. Methods: `from_smiles()`, `from_sdf()`, `featurize(mode, ...)`.

- **`NucleicAcid(pdb_path=None, chain_id=None, na_type=None)`** — DNA/RNA structure & featurization. Methods: `from_pdb()`, `from_mmcif()`, `from_structure()`, `from_sequence()`, `featurize(mode, ...)`.

- **`Complex` | `MolecularComplex`** — Unified protein-ligand(-nucleic acid) container. Methods: `from_files()`, `from_inputs()`, `from_mmcif()`, `featurize(requests, ...)`, with unified caching and cache invalidation.

### Featurizers

- **`ProteinFeaturizer(pdb_file, standardize=True, keep_hydrogens=False)`** — Central orchestrator for protein featurization.

- **`LigandFeaturizer(mol_or_smiles=None, add_hs=False, canonicalize=True, custom_smarts=None)`** — Orchestrator for ligand features.

- **`NucleicFeaturizer(pdb_path, chain_id=None)`** — Orchestrator for nucleic acid features.

- **`PLInteractionFeaturizer(protein_mol, ligand_mol, distance_cutoff=4.5, knn_cutoff=None)`** — Detects and encodes protein-ligand interactions.

- **`ResidueFeaturizer`** — Per-residue features (node & edge).

- **`AtomFeaturizer`** — Per-atom features with burial index (SASA-based) and polar/apolar classification.

- **`MoleculeFeaturizer`** — Ligand descriptors (62-dim) + fingerprints (ECFP, MACCS, RDKit, ERG).

- **`MoleculeGraphFeaturizer`** — Ligand graph with dense adjacency `(N, N, 37)` + atom features `(N, 98)` + atom-to-fragment mappings.

- **`HierarchicalFeaturizer`** — Multi-level protein features (residue + atom + interaction layers). Returns `HierarchicalProteinData`.

- **`ESMFeaturizer`** — ESM3/ESMC embeddings from sequence. Returns an `(L, D)` array.

### Parsing

- **`StructureParser`** — Abstract base for structure file parsers.

- **`PDBParser`** — Parses PDB format. Returns `List[ParsedAtom]`, `List[ParsedResidue]`. Cached via `PDBStandardizer`.

- **`MMCIFParser`** — Parses mmCIF/pdbx format. Same interface as PDBParser.

- **`ParsedAtom`** — Dataclass: `{atom_name, res_name, res_num, chain_id, element, coords, occupancy, b_factor, is_hetatm, is_metal}`.

- **`ParsedResidue`** — Dataclass: `{res_name, res_num, chain_id, atom_indices, atoms}`.

### Interaction & Metal Coordination

- **`MetalSite(metal_element, metal_coords, coordinating_atoms, coordination_number, geometry)`** — Detected metal coordination site.

- **`detect_metal_sites(atom_coords, atom_metadata, metal_indices, distance_cutoff)`** — Returns `List[MetalSite]`. `distance_cutoff` defaults to `METAL_COORDINATION_CUTOFF`.

- **`classify_coordination_geometry(coordinating_atoms, metal_coords)`** — Classifies geometry (linear, trigonal_planar, tetrahedral, square_planar, trigonal_bipyramidal, octahedral).

- **`encode_metal_features(metal_sites, n_residues)`** — Encodes detected metal coordination sites into metal type, coordination number, geometry, and distance-stat arrays.

### Arrays

plmol's featurizers return numpy arrays. `plmol/arrays.py` holds the operations
whose obvious numpy spelling differs from what torch did, and the converters.

- `to_torch(value, device=None)` — every array in a nested result as a torch
  tensor. Raises `DependencyError` without torch.
- `to_numpy(value)` — the other direction.
- `normalize(vectors, axis, eps)` — unit vectors, dividing by `max(norm, eps)`.
- `pairwise_distances(left, right)` — direct subtraction, not the
  `|a|^2 + |b|^2 - 2ab` expansion.
- `one_hot(indices, num_classes)`, `pad_last(array, before, after)`.
- `FLOAT` and `INT` — the float32 and int64 widths every feature uses.

### Cavities

Enclosed spaces found from the structure alone. See
[Cavities](protein.md#cavities).

- `detect_cavities(coords, radii, ...)` — every cavity, largest first.
- `Cavity` — centre, volume, buriedness, grid points, lining atoms and residues.
- `element_vdw_radii(elements)` — the radii detection expects.
- `Protein.featurize(mode="cavity")`, `Protein.featurize_cavity(index)`.

### Nucleic acid pairing

- `find_base_pairs(residues)` — Watson-Crick pairs by hydrogen bond geometry.
- `BasePair` — the two indices, the kind, the bond lengths, the plane angle.
  See [Watson-Crick Pairing](nucleic_acid.md#watson-crick-pairing).

### Backends

Both cover an optional dependency: the library runs without either, on its own
numpy implementation. See [Spatial Backends](protein.md#spatial-backends) and
[SASA Backends](protein.md#sasa-backends).

- `set_spatial_backend(name)`, `get_spatial_backend()`, `resolve_spatial_backend()` — `"auto"` | `"scipy"` | `"native"`.
- `knn(data, queries, k)` — the *k* nearest points, nearest first.
- `NeighbourIndex(points)` — the same query against a fixed point set, built once.
- `set_sasa_backend(name)`, `get_sasa_backend()`, `resolve_sasa_backend()` — `"auto"` | `"freesasa"` | `"native"`.
- `shrake_rupley(coords, radii)` — per-atom SASA in square Angstrom.

### Error Hierarchy

- **`PlmolError`** — Base exception.
  - **`InputError`** — Invalid user input (bad SMILES, missing atoms, etc.).
  - **`DependencyError`** — Missing optional dependency.
  - **`FeatureError`** — Extraction failure at runtime.

### Specifications

- **`FeatureSpec`** — Contract for mode normalization and stable response keys. Fields: `name`, `allowed_modes`, `default_modes`, `output_keys`.

- **`FEATURE_SPECS`** — Registry. Access via `FEATURE_SPECS[spec_name]`.
  - `PROTEIN_SPEC` — Protein graph, backbone, surface, voxel, sequence modes.
  - `LIGAND_SPEC` — Ligand graph, fingerprint, descriptor, fragment modes.
  - `INTERACTION_SPEC` — Complex-level interaction graph.
  - `NUCLEIC_ACID_SPEC` — Nucleic acid graph and sequence modes.

## Constants

All constants live in `plmol.constants` and are re-exported from `plmol.constants.__init__`.

### Amino Acids
- `AMINO_ACID_TYPES` — List of 20 standard + UNK (21 total).
- `AMINO_ACID_TOKEN` — `{residue_name: int_token}` for `nn.Embedding`.
- `RESIDUE_ATOM_TOKEN` — `{(res, atom): int_token}` (187 classes).
- `RESIDUE_MAX_SASA` — Per-residue max SASA (Tien et al. 2013).
- `STANDARD_AMINO_ACIDS_ATOMS` — Standard atom lists per residue.

### Nucleic Acids
- `NUCLEOTIDE_TYPES` — ['DA', 'DT', 'DG', 'DC', 'A', 'U', 'G', 'C', 'UNK_NT'] (9 total).
- `NUCLEOTIDE_TOKEN` — `{nt_name: int_token}` for `nn.Embedding`.
- `NUCLEOTIDE_3TO1` — 3-letter to 1-letter code (DNA + RNA).
- `DNA_BACKBONE_ATOMS`, `RNA_BACKBONE_ATOMS` — Backbone atom lists.
- `BASE_ATOMS` — Per-nucleotide base atoms (ring + substituents, no backbone).
- `STANDARD_NUCLEOTIDE_ATOMS` — Full atom lists (backbone + base).
- `NUCLEOTIDE_MAX_SASA` — Per-nucleotide max SASA.
- `WC_BASE_PAIRS` | `BASE_PAIR_PATTERNS` — Watson-Crick pair rules.
- `DNA_RESIDUES`, `RNA_RESIDUES` — Classification sets.
- `IS_PURINE`, `IS_PYRIMIDINE` — Per-nucleotide boolean dicts.
- `NUCLEOTIDE_PROPERTIES` — `(mol_weight, n_hbond_donors, n_hbond_acceptors, is_purine)` per nucleotide.

### Interactions
- `INTERACTION_TYPES` — List of detectable interactions (H-bond, hydrophobic, pi-stacking, salt bridge, metal coordination, etc.).
- `INTERACTION_TOKEN` — Token mapping for edge encoding.
- `METAL_COORDINATION_CUTOFF` — Default metal-ligand distance cutoff (Angstroms).
- `COMMON_COORDINATION_GEOMETRIES` — Expected geometries (tetrahedral, octahedral, square_planar, etc.).
- `METAL_PREFERRED_DONORS` — Element-preferred donor lists.

### Physical Properties
- `VDW_RADII` — Van der Waals radii by element.
- `ELEMENT_MASS` — Atomic mass by element.
- `ELEMENT_TYPES` — All PDB elements.

### Runtime
- Default cutoffs & thresholds (pocket distance, SASA burial, grid density, etc.).
- Cavity detection: `CAVITY_GRID_RESOLUTION`, `CAVITY_PSP_THRESHOLD`,
  `CAVITY_SCAN_LENGTH`, `CAVITY_MIN_POINTS`, `CAVITY_PROBE_RADIUS`,
  `CAVITY_PADDING`, `CAVITY_LINING_MARGIN`.
