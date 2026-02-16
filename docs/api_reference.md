# plmol API Reference

> **Version**: 0.2.1
> Protein-Ligand Molecular Feature Extraction Toolkit

---

## Table of Contents

- [Quick Start](#quick-start)
- [Protein](#protein)
  - [Initialization](#protein-initialization)
  - [Featurization Modes](#protein-featurization-modes)
  - [Graph Mode](#protein-graph-mode)
  - [Backbone Mode](#protein-backbone-mode)
  - [Surface Mode](#protein-surface-mode)
  - [Sequence Mode](#protein-sequence-mode)
  - [Pocket Featurization](#pocket-featurization)
- [Ligand](#ligand)
  - [Initialization](#ligand-initialization)
  - [Featurization Modes](#ligand-featurization-modes)
  - [Graph Mode](#ligand-graph-mode)
  - [Fingerprint Mode](#ligand-fingerprint-mode)
  - [Surface Mode](#ligand-surface-mode)
- [Complex](#complex)
  - [Initialization](#complex-initialization)
  - [Combined Featurization](#combined-featurization)
  - [Interaction Features](#interaction-features)
- [Geometry Functions](#geometry-functions)
- [Backbone Featurizer](#backbone-featurizer)
- [Low-Level Featurizers](#low-level-featurizers)
- [Constants](#constants)
- [Scripts & Examples](#scripts--examples)

---

## Quick Start

```python
from plmol import Protein, Ligand, Complex

# Protein
protein = Protein.from_pdb("protein.pdb")
result = protein.featurize(mode="all")
# result.keys() -> dict_keys(['sequence', 'graph', 'surface', 'backbone'])

# Ligand
ligand = Ligand.from_smiles("CCO")
result = ligand.featurize(mode=["graph", "fingerprint"])
# result.keys() -> dict_keys(['graph', 'fingerprint'])

# Complex (protein + ligand + interaction)
cx = Complex.from_files("protein.pdb", "ligand.sdf")
result = cx.featurize(requests="all")
# result.keys() -> dict_keys(['ligand', 'protein', 'interaction'])
```

---

## Protein

### Protein Initialization

```python
from plmol import Protein

# From PDB file (recommended)
protein = Protein.from_pdb("protein.pdb", standardize=True, keep_hydrogens=False)

# From sequence (ESM/Foldseek style - no structure)
protein = Protein.from_sequence("MKFLILLFNILCLFPVLAADNHGVS...")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `standardize` | `bool` | `True` | Standardize PDB (remove non-standard, fix naming) |
| `keep_hydrogens` | `bool` | `False` | Keep hydrogen atoms |

### Protein Featurization Modes

```python
result = protein.featurize(
    mode="all",                              # str or list of modes
    graph_kwargs={"level": "residue"},       # graph options
    surface_kwargs={},                       # surface options
    backbone_kwargs={"k_neighbors": 30},     # backbone options
)
```

| Mode | Output Key | Description |
|------|-----------|-------------|
| `"graph"` | `"graph"` | Residue/atom-level graph (node_features, edge_index, ...) |
| `"backbone"` | `"backbone"` | Backbone features for inverse folding (dihedrals, kNN, local frames) |
| `"surface"` | `"surface"` | Molecular surface mesh with per-vertex features |
| `"sequence"` | `"sequence"` | Amino acid sequence string or chain dict |
| `"all"` | all above | All modes combined |

Lazy properties are also available:

```python
protein.sequence   # str (single chain) or Dict[str, str] (multi-chain)
protein.graph      # residue-level graph (auto-computed)
protein.surface    # surface mesh (auto-computed)
```

---

### Protein Graph Mode — Residue Level

```python
result = protein.featurize(
    mode="graph",
    graph_kwargs={"level": "residue", "distance_cutoff": 8.0},
)
graph = result["graph"]
```

#### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | tuple of 8 | `Tensor` | Residue scalar features (total 81-dim) |
| `node_vector_features` | tuple of 3 | `Tensor` | Residue vector features (total 31 vectors x 3) |
| `edge_index` | `(2, E)` | `int64` | Sparse edge pairs (source, target) |
| `edge_features` | tuple of 2 | `Tensor` | Edge scalar features (total 39-dim) |
| `edge_vector_features` | tuple of 1 | `Tensor` | Edge vector features (8 vectors x 3) |
| `coords` | `(L, 2, 3)` | `float32` | CA + sidechain centroid coordinates |
| `distance_cutoff` | `float` | — | Cutoff used (default: 8.0 A) |
| `level` | `str` | — | `"residue"` |

Edge construction: all residue pairs (i, j) where any of the 4 distances (CA-CA, SC-SC, CA-SC, SC-CA) < `distance_cutoff`.

#### Node Scalar Features `(L, 81)` — tuple of 8 tensors

| Index | Tensor | Dim | Features |
|-------|--------|-----|----------|
| `[0:21]` | residue_one_hot | 21 | One-hot: 20 amino acids + UNK |
| `[21:23]` | terminal_flags | 2 | is_N_terminal, is_C_terminal |
| `[23:33]` | self_distance | 10 | Intra-residue pairwise distances among N, CA, C, O, SC (upper triangle) |
| `[33:53]` | degree_feature | 20 | cos/sin of 10 angles: phi, psi, omega, chi1-chi5, backbone_curvature, backbone_torsion |
| `[53:58]` | has_chi_angles | 5 | Binary flags: has chi1, chi2, chi3, chi4, chi5 |
| `[58:68]` | sasa | 10 | SASA: total, polar, apolar, mainchain, sidechain (abs/350 + relative) |
| `[68:76]` | rf_distance | 8 | Forward/reverse neighbor distances: fwd(CA-CA, SC-SC, CA-SC, SC-CA) + rev(same) |
| `[76:81]` | physicochemical | 5 | Residue properties: hydrophobicity (Kyte-Doolittle), volume (Zamyatnin), charge, flexibility, polarity |

#### Node Vector Features `(L, 31, 3)` — tuple of 3 tensors

| Index | Tensor | Vectors | Features |
|-------|--------|---------|----------|
| `[0:20]` | self_vector | 20 | Intra-residue pairwise direction vectors among N, CA, C, O, SC |
| `[20:28]` | rf_vector | 8 | Forward/reverse neighbor direction vectors (CA-CA, SC-SC, CA-SC, SC-CA x 2) |
| `[28:31]` | local_frames | 3 | Local N-CA-C coordinate frame (3 orthonormal basis vectors) |

#### Edge Scalar Features `(E, 39)` — tuple of 2 tensors

| Index | Tensor | Dim | Features |
|-------|--------|-----|----------|
| `[0:4]` | distance | 4 | CA-CA, SC-SC, CA-SC, SC-CA distances (Angstrom) |
| `[4:39]` | relative_position | 35 | One-hot sequence separation: d=0, 1, ..., 32, >32, cross-chain, UNK |

#### Edge Vector Features `(E, 8, 3)` — tuple of 1 tensor

| Index | Tensor | Vectors | Features |
|-------|--------|---------|----------|
| `[0:8]` | interaction_vectors | 8 | CA_i->CA_j, CA_j->CA_i, CA_i->SC_j, CA_j->SC_i, SC_i->CA_j, SC_j->CA_i, SC_i->SC_j, SC_j->SC_i |

---

### Protein Graph Mode — Atom Level

```python
result = protein.featurize(
    mode="graph",
    graph_kwargs={"level": "atom", "distance_cutoff": 4.0},
)
graph = result["graph"]
```

#### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | `(A,)` | `int64` | Atom token ID (0-186, for `nn.Embedding`) — same as `atom_tokens` |
| Node token features | 3 tensors | `int64` | Integer tokens for embedding layers |
| Node scalar features | 11 tensors | `float32` | Continuous per-atom features (total 11-dim) |
| `coords` | `(A, 3)` | `float32` | Atom 3D coordinates |
| `edge_index` | `(2, E)` | `int64` | Sparse edge pairs (source, target) |
| Edge features | 4 tensors | `float32` | Per-edge features (total 6-dim) |
| `distance_cutoff` | `float` | — | Cutoff used (default: 4.0 Å) |
| `level` | `str` | — | `"atom"` |

Edge construction: all atom pairs within `distance_cutoff` Angstrom.

#### Node Token Features `(A,)` — int64, for `nn.Embedding`

| Key | Vocab Size | Description |
|-----|-----------|-------------|
| `atom_tokens` | 187 | Residue-atom pair token (e.g. ALA-CA, GLY-N). Use `nn.Embedding(187, d)` |
| `residue_token` | 22 | Residue type per atom (20 AA + Metal + UNK) |
| `atom_element` | 19 | Element type per atom (H, C, N, O, S, P, Se, metals, UNK) |

#### Node Scalar Features — float32, total 11-dim

| Index | Key | Dim | Range | Description |
|-------|-----|-----|-------|-------------|
| `[0]` | `sasa` | 1 | [0, ~) | Per-atom absolute SASA (Å²) |
| `[1]` | `relative_sasa` | 1 | [0, 1] | SASA / residue_max_sasa (Tien et al. 2013) |
| `[2]` | `b_factor` | 1 | [0, 1] | Normalized B-factor (B / 100, capped at 1.0) |
| `[3]` | `b_factor_zscore` | 1 | (~) | Per-chain B-factor z-score: (b - chain_mean) / chain_std |
| `[4]` | `is_backbone` | 1 | {0, 1} | 1.0 if backbone atom (N, CA, C, O), 0.0 if sidechain |
| `[5]` | `formal_charge` | 1 | [-0.5, 1] | Partial charge at physiological pH |
| `[6]` | `is_hbond_donor` | 1 | {0, 1} | 1.0 if H-bond donor (backbone N except PRO + sidechain donors) |
| `[7]` | `is_hbond_acceptor` | 1 | {0, 1} | 1.0 if H-bond acceptor (backbone O + sidechain acceptors) |
| `[8:11]` | `secondary_structure` | 3 | {0, 1} | One-hot [helix, sheet, coil] from phi/psi Ramachandran |

`formal_charge` values: ASP OD1/OD2 = -0.5, GLU OE1/OE2 = -0.5, LYS NZ = +1.0, ARG NH1/NH2 = +0.5, all others = 0.0

#### Node Index Features — int64

| Key | Description |
|-----|-------------|
| `residue_number` | PDB residue sequence number (as in PDB file) |
| `residue_count` | Sequential 0-based residue index (for sequence separation) |

#### Edge Features — total 6-dim

| Index | Key | Shape | Type | Description |
|-------|-----|-------|------|-------------|
| `[0]` | `edge_distances` | `(E,)` | `float32` | Euclidean distance (Å) |
| `[1]` | `same_residue` | `(E,)` | `float32` | 1.0 if both atoms in same residue |
| `[2]` | `sequence_separation` | `(E,)` | `float32` | \|residue_i - residue_j\|, capped at 32 |
| `[3:6]` | `unit_vector` | `(E, 3)` | `float32` | Normalized direction vector src → dst |

#### Metadata

| Key | Type | Description |
|-----|------|-------------|
| `atom_name` | `list[str]` | PDB atom names (e.g. "CA", "CB", "OG") |
| `chain_label` | `list[str]` | Chain identifiers (e.g. "A", "B") |

---

### Protein Backbone Mode

For inverse folding models (ProteinMPNN, ESM-IF, GVP, PiFold).

```python
result = protein.featurize(
    mode="backbone",
    backbone_kwargs={"k_neighbors": 30},
)
backbone = result["backbone"]
```

#### Output — Node Features

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `backbone_coords` | `(L, 4, 3)` | `float32` | N, CA, C, O coordinates |
| `cb_coords` | `(L, 3)` | `float32` | Virtual CB (ProteinMPNN geometry) |
| `dihedrals` | `(L, 3)` | `float32` | phi, psi, omega (radians, chain-boundary-aware) |
| `dihedrals_sincos` | `(L, 6)` | `float32` | sin/cos encoding: [sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)] |
| `dihedrals_mask` | `(L, 3)` | `bool` | True where dihedral is valid (invalid positions zeroed in sincos) |
| `orientation_frames` | `(L, 3, 3)` | `float32` | N-CA-C local coordinate frames |
| `residue_types` | `(L,)` | `int64` | Residue type (0-20) |
| `chain_ids` | `(L,)` | `int64` | Integer chain ID |
| `residue_mask` | `(L,)` | `bool` | True if all 4 backbone atoms present |

#### Output — kNN Graph (E = L * k)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `edge_index` | `(2, E)` | `int64` | kNN edges over CA atoms |
| `edge_dist` | `(E,)` | `float32` | CA-CA Euclidean distance |
| `edge_unit_vec` | `(E, 3)` | `float32` | Unit direction vector i -> j |
| `edge_seq_sep` | `(E,)` | `int64` | Sequence separation \|i-j\| (0 for cross-chain) |
| `edge_same_chain` | `(E,)` | `bool` | True if same chain |
| `edge_rbf` | `(E, 16)` | `float32` | Gaussian RBF distance encoding (16 basis, 0-20 A) |
| `edge_local_pos` | `(E, 3)` | `float32` | CA_j position in residue i's local frame (SE(3)-invariant) |
| `edge_rel_orient` | `(E, 3, 3)` | `float32` | Relative rotation R_i^T @ R_j |

#### Output — Metadata

| Key | Type | Description |
|-----|------|-------------|
| `num_residues` | `int` | Total residue count (L) |
| `num_chains` | `int` | Total chain count |
| `k_neighbors` | `int` | k used for kNN graph |

---

### Protein Surface Mode

```python
result = protein.featurize(mode="surface", surface_kwargs={
    "mode": "mesh",           # "mesh" (default) or "point_cloud"
    "grid_density": 2.5,
    "include_features": True,
})
surface = result["surface"]
```

#### Output (mesh mode)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `points` | `(V, 3)` | `ndarray` | Vertex positions |
| `verts` | `(V, 3)` | `ndarray` | Alias for points |
| `faces` | `(F, 3)` | `ndarray` | Triangle face indices (mesh mode only) |
| `normals` | `(V, 3)` | `ndarray` | Vertex normals |
| `hydropathy` | `(V,)` | `ndarray` | Hydrophobicity per vertex |
| `charge` | `(V,)` | `ndarray` | Electrostatic potential per vertex |
| `curvature` | `(V,)` | `ndarray` | Surface curvature per vertex |
| `shape_index` | `(V,)` | `ndarray` | Shape index descriptor per vertex |
| `residue_ids` | `(V,)` | `ndarray` | Nearest residue index per vertex |

Surface parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_density` | 2.5 | Grid resolution (higher = finer mesh) |
| `threshold` | 0.5 | Isosurface threshold |
| `sharpness` | 1.5 | Atom field sharpness |
| `include_features` | True | Compute MaSIF-style features |
| `mode` | `"mesh"` | `"mesh"` (marching cubes) or `"point_cloud"` (SAS sampling) |
| `n_points_per_atom` | 100 | Points per atom (point cloud mode) |
| `probe_radius` | 1.4 | Solvent probe radius (point cloud mode) |

---

### Protein Sequence Mode

```python
result = protein.featurize(mode="sequence")
seq = result["sequence"]
# Single chain: str "MKFLIL..."
# Multi-chain: Dict[str, str] {"A": "MKFLIL...", "B": "GRPEWK..."}
```

---

### Pocket Featurization

Extract and featurize the binding pocket around a ligand.

```python
pocket = protein.featurize_pocket(
    ligand=ligand_mol,   # RDKit Mol, file path, or Ligand object
    cutoff=6.0,          # Angstrom
    mode="graph",
    graph_kwargs={"level": "residue"},
)
# Returns same structure as protein.featurize() for the pocket subset
```

---

## Ligand

### Ligand Initialization

```python
from plmol import Ligand

# From SMILES
ligand = Ligand.from_smiles("CCO", add_hs=False)

# From SDF file
ligand = Ligand.from_sdf("ligand.sdf")

# Generate 3D conformer (needed for surface)
ligand.generate_conformer()
```

### Ligand Featurization Modes

```python
result = ligand.featurize(
    mode="all",                    # str or list of modes
    graph_kwargs={},               # graph options
    surface_kwargs={},             # surface options
    fingerprint_kwargs={},         # fingerprint options
    generate_conformer=False,      # auto-generate 3D if missing
    add_hs=None,                   # hydrogen override
)
```

| Mode | Output Key | Description |
|------|-----------|-------------|
| `"graph"` | `"graph"` | Dense adjacency graph (node_features, adjacency, bond_mask, ...) |
| `"fingerprint"` | `"fingerprint"` | Descriptors + ECFP4/6, MACCS, RDKit FP, AtomPair, ErG |
| `"surface"` | `"surface"` | Molecular surface mesh |
| `"smiles"` | `"smiles"` | Canonical SMILES string |
| `"sequence"` | `"sequence"` | Same as SMILES (ligand alias) |
| `"all"` | all above | All modes combined |

Lazy properties:

```python
ligand.smiles       # str
ligand.sequence     # str (alias for smiles)
ligand.graph        # dict (auto-computed)
ligand.surface      # dict (auto-computed, needs conformer)
ligand.fingerprint  # dict (auto-computed)
```

---

### Ligand Graph Mode

`Ligand.featurize(mode="graph")` returns a **dense adjacency** representation.

```python
result = ligand.featurize(mode="graph")
graph = result["graph"]
```

#### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | `(N, 98)` | `ndarray float32` | Per-atom feature vector |
| `adjacency` | `(N, N, 37)` | `ndarray float32` | Dense adjacency (bond + pair channels) |
| `bond_mask` | `(N, N)` | `ndarray bool` | True where chemical bond exists |
| `distance_matrix` | `(N, N)` | `ndarray float32` | Euclidean distance (0 if no 3D) |
| `distance_bounds` | `(N, N, 2)` | `ndarray float32` | DG lower/upper distance bounds |
| `coords` | `(N, 3)` | `ndarray float32` | 3D coordinates (0 if no conformer) |

Sparse conversion:

```python
from plmol import LigandFeaturizer
edge_index, edge_features = LigandFeaturizer.adjacency_to_bond_edges(graph["adjacency"])
# edge_index: (2, E)  edge_features: (E, 37)
```

#### Node Features `(N, 98)`

| Index | Group | Dim | Features |
|-------|-------|-----|----------|
| `[0:22]` | Atom type | 22 | One-hot: H, C, N, O, S, P, F, Cl, Br, I, Se, Zn, Mg, Ca, Fe, Mn, Cu, Co, Ni, Na, K, UNK |
| `[22:28]` | Formal charge | 6 | One-hot: -2, -1, 0, 1, 2, UNK |
| `[28:34]` | Hybridization | 6 | One-hot: SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED |
| `[34:37]` | Flags | 3 | is_aromatic, is_in_ring, radical_electrons |
| `[37:42]` | Total Hs | 5 | One-hot: 0, 1, 2, 3, 4 |
| `[42:49]` | Degree | 7 | One-hot: 0, 1, 2, 3, 4, 5, 6 |
| `[49:52]` | Atom properties | 3 | mass, vdw_radius, electronegativity (normalized) |
| `[52:60]` | Stereochemistry | 8 | chiral_CW, chiral_CCW, chiral_unspec, potential_chiral, has_stereo_bond, is_aromatic, is_SP2, is_SP |
| `[60:62]` | Partial charges | 2 | Gasteiger charge (shifted), abs_charge |
| `[62:68]` | Physical properties | 6 | mass, vdw_radius, covalent_radius, ionization_energy, polarizability, lone_pairs |
| `[68:73]` | Topological | 5 | eccentricity, closeness_centrality, betweenness_centrality, dist_to_heteroatom, dist_to_ring |
| `[73:78]` | SMARTS patterns | 5 | h_acceptor, h_donor, hydrophobic, positive, negative |
| `[78:94]` | Extended neighborhood | 16 | Bond type counts (1-hop: single/double/triple/aromatic, 2-hop: same 4, 3-hop: same 4, total: same 4) |
| `[94:96]` | Crippen contributions | 2 | Per-atom logP, molar refractivity (Wildman-Crippen) |
| `[96:97]` | TPSA contribution | 1 | Per-atom topological polar surface area |
| `[97:98]` | Labute ASA | 1 | Per-atom approximate surface area (Labute) |

#### Adjacency Channels `(N, N, 37)`

Channels `[0:27]` are bond features (nonzero only where `bond_mask` is True).
Channels `[27:37]` are pair features (defined for all atom pairs).

| Index | Group | Dim | Features |
|-------|-------|-----|----------|
| `[0:4]` | Bond type | 4 | One-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC |
| `[4:10]` | Bond stereo | 6 | One-hot: ANY, CIS, E, NONE, TRANS, Z |
| `[10:15]` | Bond direction | 5 | One-hot: NONE, BEGINWEDGE, BEGINDASH, ENDDOWNRIGHT, ENDUPRIGHT |
| `[15:20]` | Bond properties | 5 | is_aromatic, is_conjugated, is_in_ring, is_rotatable, bond_order |
| `[20:21]` | Bond distance | 1 | 3D bond length (normalized) |
| `[21:27]` | Bond topological | 6 | betweenness, is_bridge, ring_fusion, dist_to_heteroatom, dist_to_ring, graph_distance |
| `[27:33]` | Shortest path dist | 6 | One-hot: d=1, d=2, d=3, d=4, d=5, d>=6 |
| `[33:34]` | Euclidean distance | 1 | 3D Euclidean distance (normalized) |
| `[34:37]` | Pair context | 3 | same_ring, same_fragment, same_aromatic_system |

---

### Ligand Descriptor / Fingerprint Mode

```python
result = ligand.featurize(mode="fingerprint")
fp = result["fingerprint"]
```

#### Output (default)

| Key | Shape | Description |
|-----|-------|-------------|
| `descriptors` | `(62,)` | Normalized molecular descriptors |
| `maccs` | `(167,)` | MACCS structural keys |
| `ecfp4` | `(2048,)` | ECFP4 (Morgan radius=2, chirality-aware) |
| `ecfp4_feature` | `(2048,)` | ECFP4 feature-invariant variant |
| `ecfp6` | `(2048,)` | ECFP6 (Morgan radius=3, chirality-aware) |
| `rdkit` | `(2048,)` | RDKit path-based fingerprint |
| `atom_pair` | `(2048,)` | Atom pair fingerprint |
| `topological_torsion` | `(2048,)` | Topological torsion fingerprint |
| `erg` | `(315,)` | ErG pharmacophore fingerprint |

Select specific fingerprints:

```python
result = ligand.featurize(
    mode="fingerprint",
    fingerprint_kwargs={"include_fps": ["ecfp4", "maccs", "peoe_vsa"]},
)
```

Additional fingerprints available via `include_fps`:

| Key | Shape | Description |
|-----|-------|-------------|
| `ecfp4_count` | `(2048,)` | ECFP4 count fingerprint |
| `ecfp6_feature` | `(2048,)` | ECFP6 feature-invariant variant |
| `pharmacophore2d` | `(1024,)` | 2D pharmacophore fingerprint |
| `avalon` | `(2048,)` | Avalon fingerprint |
| `peoe_vsa` | `(14,)` | PEOE_VSA (charge-partitioned surface area) |
| `slogp_vsa` | `(12,)` | SlogP_VSA (LogP-partitioned surface area) |
| `smr_vsa` | `(10,)` | SMR_VSA (molar refractivity-partitioned surface area) |
| `mqn` | `(42,)` | Molecular Quantum Numbers |

#### Descriptors `(62,)`

All values normalized to [0, 1].

| Index | Group | Dim | Features |
|-------|-------|-----|----------|
| `[0:5]` | Basic properties | 5 | mw, logp, tpsa, n_rotatable_bonds, flexibility |
| `[5:7]` | H-bonding | 2 | hbd, hba |
| `[7:12]` | Counts | 5 | n_atoms, n_bonds, n_rings, n_aromatic_rings, heteroatom_ratio |
| `[12:16]` | Topological indices | 4 | balaban_j, bertz_ct, chi0, chi1 |
| `[16:20]` | Kier-Hall | 4 | hall_kier_alpha, kappa1, kappa2, kappa3 |
| `[20:24]` | Electronic | 4 | mol_mr, labute_asa, num_radical_electrons, num_valence_electrons |
| `[24:29]` | Ring subtypes | 5 | saturated_rings, aliphatic_rings, saturated_heterocycles, aliphatic_heterocycles, aromatic_heterocycles |
| `[29:32]` | Misc | 3 | num_heteroatoms, formal_charge, chi0n |
| `[32:37]` | Drug-likeness | 5 | lipinski_violations, passes_lipinski, qed, num_heavy_atoms, frac_csp3 |
| `[37:40]` | Ring structure | 3 | n_ring_systems, max_ring_size, avg_ring_size |
| `[40:44]` | Charge distribution | 4 | max_partial_charge, min_partial_charge, max_abs_partial_charge, min_abs_partial_charge |
| `[44:50]` | ADMET filters | 6 | veber_violations, ghose_violations, egan_violations, muegge_violations, pfizer_375_alert, gsk_4400_pass |
| `[50:52]` | Structural alerts | 2 | pains_alert_count, brenk_alert_count |
| `[52:56]` | Structural complexity | 4 | num_amide_bonds, num_stereocenters, num_spiro_atoms, num_bridgehead_atoms |
| `[56:57]` | Solubility | 1 | esol_logs (Delaney equation) |
| `[57:62]` | 3D shape | 5 | npr1, npr2, asphericity, eccentricity, radius_of_gyration |

---

### Ligand Surface Mode

```python
ligand.generate_conformer()  # needed if from SMILES
result = ligand.featurize(mode="surface")
surface = result["surface"]
```

| Key | Shape | Description |
|-----|-------|-------------|
| `points` | `(V, 3)` | Vertex positions |
| `faces` | `(F, 3)` | Triangle face indices |
| `normals` | `(V, 3)` | Vertex normals |
| `features` | `(V, C)` | Per-vertex chemical features |

---

## Complex

### Complex Initialization

```python
from plmol import Complex

# From files
cx = Complex.from_files("protein.pdb", "ligand.sdf")

# From objects/mixed inputs
cx = Complex.from_inputs(
    protein="protein.pdb",       # path or Protein object
    ligand="CCO",                # SMILES, path, RDKit Mol, or Ligand object
    standardize=True,
    add_hs=False,
)

# Swap components
cx.set_ligand("new_ligand.sdf")
cx.set_protein("new_protein.pdb")
```

### Combined Featurization

```python
result = cx.featurize(
    requests="all",  # "ligand", "protein", "interaction", or "all"
    ligand_kwargs={"mode": ["graph", "fingerprint"]},
    protein_kwargs={"mode": ["graph", "sequence"]},
    interaction_kwargs={"distance_cutoff": 6.0},
)
# result["ligand"]      -> ligand features
# result["protein"]     -> protein features
# result["interaction"] -> interaction graph
```

Individual access:

```python
cx.ligand(mode="graph")
cx.protein(mode="backbone", backbone_kwargs={"k_neighbors": 30})  # not available via Complex yet
cx.interaction(distance_cutoff=6.0, pocket_cutoff=8.0)
```

### Interaction Features

```python
interaction = cx.interaction(distance_cutoff=6.0)
```

| Key | Type | Description |
|-----|------|-------------|
| `edges` | `Tensor (2, E)` | Protein-ligand heavy atom pairs (pharmacophore interactions) |
| `edge_features` | `Tensor (E, 74)` | Interaction feature vectors |
| `interactions` | `List[Interaction]` | Detailed interaction objects |
| `metadata` | `dict` | Distance cutoff, atom counts |

**Interaction types**: `hydrogen_bond`, `pi_stacking`, `salt_bridge`, `hydrophobic`, `halogen_bond`, `metal_coordination`, `cation_pi`

#### Edge Features `(E, 74)`

| Index | Group | Dim | Features |
|-------|-------|-----|----------|
| `[0:7]` | Interaction type | 7 | One-hot: hydrogen_bond, salt_bridge, pi_stacking, cation_pi, hydrophobic, halogen_bond, metal_coordination |
| `[7:11]` | Geometry | 4 | Distance (normalized), angle, has_valid_angle, angle_type |
| `[11:31]` | Element types | 20 | Protein element one-hot (10) + ligand element one-hot (10) |
| `[31:43]` | Hybridization | 12 | Protein hybridization (6) + ligand hybridization (6) |
| `[43:45]` | Formal charges | 2 | Protein charge, ligand charge (normalized) |
| `[45:47]` | Aromatic | 2 | Protein is_aromatic, ligand is_aromatic |
| `[47:51]` | Ring/degree | 4 | is_in_ring (2) + degree (2) |
| `[51:72]` | Residue type | 21 | Protein residue one-hot |
| `[72]` | Backbone | 1 | Protein atom is_backbone |
| `[73]` | Strength | 1 | Gaussian decay from ideal distance: exp(-0.5 * ((d - ideal) / 0.5)^2) |

#### Contact Edges (optional)

```python
# Include all heavy atom pairs within cutoff (for GNN to learn novel interactions)
graph = featurizer.get_interaction_graph(include_contacts=True, contact_cutoff=4.5)
```

| Key | Type | Description |
|-----|------|-------------|
| `contact_edges` | `Tensor (2, E_c)` | All protein-ligand heavy atom pairs within cutoff |
| `contact_distances` | `Tensor (E_c,)` | Pairwise distances |
| `num_contacts` | `int` | Number of contact edges |

---

## Geometry Functions

Stateless pure functions for geometric computations. Importable from `plmol.protein`.

```python
from plmol.protein import (
    calculate_dihedral,
    calculate_local_frames,
    calculate_backbone_curvature,
    calculate_backbone_torsion,
    calculate_virtual_cb,
    calculate_self_distances_vectors,
    rbf_encode,
)
```

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `calculate_dihedral(coords)` | `(N, M, 3)` | `(N, M)` | Dihedral angles from atom coords |
| `calculate_local_frames(coords)` | `(L, MAX_ATOMS, 3)` | `(L, 3, 3)` | N-CA-C local coordinate frames |
| `calculate_backbone_curvature(coords, terminal_flags)` | `(L, MAX_ATOMS, 3)` | `(L,)` | CA-based backbone curvature |
| `calculate_backbone_torsion(coords, terminal_flags)` | `(L, MAX_ATOMS, 3)` | `(L,)` | CA-based backbone torsion |
| `calculate_virtual_cb(coords)` | `(L, MAX_ATOMS, 3)` | `(L, 3)` | Virtual CB position |
| `calculate_self_distances_vectors(coords)` | `(L, MAX_ATOMS, 3)` | `(L, 10), (L, 20, 3)` | Intra-residue distances & vectors |
| `rbf_encode(distances, d_min, d_max, num_rbf)` | `(*)` | `(*, num_rbf)` | Gaussian RBF encoding |

---

## Backbone Featurizer

High-level backbone feature functions. Importable from `plmol.protein`.

```python
from plmol.protein import (
    compute_backbone_features,
    compute_edge_frame_features,
)
```

| Function | Description |
|----------|-------------|
| `compute_backbone_features(coords, residues, residue_types, k_neighbors)` | Full backbone assembly (20 output keys) |
| `compute_edge_frame_features(ca_coords, orientation_frames, edge_index)` | SE(3)-invariant edge features |

`compute_backbone_features` is what `Protein.featurize(mode="backbone")` calls internally.

---

## Low-Level Featurizers

For advanced use cases, the underlying featurizer classes are directly accessible.

```python
from plmol import (
    # Protein
    ProteinFeaturizer,         # Main protein featurizer (PDB parse + cache)
    ResidueFeaturizer,         # Residue-level features
    AtomFeaturizer,            # Atom-level features
    HierarchicalFeaturizer,    # Atom-residue hierarchical (ESM + attention)
    ESMFeaturizer,             # ESM3/ESMC embeddings
    PDBStandardizer,           # PDB cleanup

    # Ligand
    MoleculeFeaturizer,        # Molecular features + fingerprints
    MoleculeGraphFeaturizer,   # Graph node/edge features
    LigandFeaturizer,          # Ligand-specific wrapper (graph + surface + FP)

    # Interaction
    PLInteractionFeaturizer,   # Protein-ligand interaction edges
)
```

### ESMFeaturizer

```python
from plmol import ESMFeaturizer

esm = ESMFeaturizer(model_type="esmc", model_name="esmc_600m", device="cuda")
embeddings = esm.extract("MKFLIL...")
# embeddings["embeddings"]: (L, 1152)
# embeddings["bos"]: (1152,)
# embeddings["eos"]: (1152,)
```

| Model | Name | Embedding Dim |
|-------|------|--------------|
| ESMC | `esmc_300m` | 960 |
| ESMC | `esmc_600m` | 1152 |
| ESM3 | `esm3-open` | 1536 |

### HierarchicalFeaturizer

Produces atom-residue hierarchical features with ESM embeddings.

```python
from plmol import HierarchicalFeaturizer

hf = HierarchicalFeaturizer()
data = hf.featurize("protein.pdb")
# data.atom_tokens:        (N_atoms,) int tensor
# data.atom_coords:        (N_atoms, 3)
# data.residue_features:   (L, 76)
# data.atom_to_residue:    (N_atoms,) mapping
# data.esmc_embeddings:    (L, 1152)
# data.esm3_embeddings:    (L, 1536)
```

---

## Constants

```python
from plmol import constants

# Element types
constants.ATOM_TYPES                         # ['H', 'C', 'N', 'O', 'S', ...]
constants.NUM_HEAVY_ELEMENT_TYPES            # 10

# Amino acids
constants.AMINO_ACID_3TO1                    # {'ALA': 'A', ...}
constants.RESIDUE_TYPES                      # ['ALA', ..., 'VAL', 'Other']
constants.NUM_RESIDUE_TYPES                  # 21
constants.MAX_ATOMS_PER_RESIDUE              # 15

# Interactions
constants.INTERACTION_TYPES                  # ['h_bond', 'pi_stacking', ...]
constants.NUM_INTERACTION_TYPES              # 7
constants.IDEAL_DISTANCES                    # {'h_bond': 2.8, ...}

# Default cutoffs
constants.DEFAULT_ATOM_GRAPH_DISTANCE_CUTOFF     # 4.0 A
constants.DEFAULT_RESIDUE_GRAPH_DISTANCE_CUTOFF  # 8.0 A
constants.DEFAULT_BACKBONE_KNN_NEIGHBORS         # 30

# Surface/Voxel
constants.SURFACE_DEFAULT_GRID_DENSITY       # 2.5
constants.VOXEL_DEFAULT_RESOLUTION           # 1.0 A/voxel
constants.VOXEL_DEFAULT_BOX_SIZE             # 24
```

---

## Scripts & Examples

### Batch Scripts

```bash
# Batch protein featurization
python scripts/batch_protein_featurize.py --input_dir pdbs/ --output_dir features/

# Batch ligand featurization
python scripts/batch_ligand_featurize.py --input_dir sdfs/ --output_dir features/
```

### Example Files

| File | Description |
|------|-------------|
| `examples/protein_ligand_demo.py` | Full protein + ligand workflow |
| `examples/ligand_views_demo.py` | Ligand graph, fingerprint, surface demo |
| `examples/check_graph_dims.py` | Validate graph tensor dimensions |
| `examples/usage_example.ipynb` | Interactive notebook tutorial |
| `examples/10gs_protein.pdb` | Example PDB (human glutathione S-transferase) |
| `examples/10gs_ligand.sdf` | Example ligand (S-hexylglutathione) |

---

## Directory Structure

```
plmol/
├── __init__.py                  # Top-level exports (Protein, Ligand, Complex, ...)
├── base.py                      # BaseMolecule abstract class
├── cache.py                     # LRU caching utility
├── complex.py                   # Complex class
├── errors.py                    # PlmolError, InputError, DependencyError, FeatureError
├── specs.py                     # FeatureSpec, LIGAND_SPEC, PROTEIN_SPEC, INTERACTION_SPEC
├── constants/
│   ├── amino_acids.py           # Amino acid mappings & tokens
│   ├── elements.py              # Element types & periodic table
│   ├── interactions.py          # Interaction types & ideal distances
│   ├── physical_properties.py   # VdW radius, mass, etc.
│   ├── runtime.py               # Default parameters (cutoffs, grid density)
│   └── smarts_patterns.py       # Pharmacophore SMARTS patterns
├── protein/
│   ├── core.py                  # Protein class
│   ├── protein_featurizer.py    # ProteinFeaturizer (parse + cache)
│   ├── residue_featurizer.py    # ResidueFeaturizer (residue features)
│   ├── atom_featurizer.py       # AtomFeaturizer (atom features)
│   ├── geometry.py              # Stateless geometric functions (7 functions)
│   ├── backbone_featurizer.py   # Backbone features for inverse folding (4 functions)
│   ├── hierarchical_featurizer.py  # HierarchicalFeaturizer + HierarchicalProteinData
│   ├── esm_featurizer.py        # ESM3/ESMC embedding extraction
│   ├── pdb_standardizer.py      # PDB standardization
│   └── utils.py                 # PDBParser and utilities
├── ligand/
│   ├── core.py                  # Ligand class
│   ├── base.py                  # MoleculeFeaturizer
│   ├── featurizer.py            # LigandFeaturizer
│   └── graph.py                 # MoleculeGraphFeaturizer
├── interaction/
│   ├── pli_featurizer.py        # PLInteractionFeaturizer
│   └── pocket_extractor.py      # Pocket extraction
├── featurizers/
│   ├── surface.py               # Unified surface building
│   └── voxel.py                 # Unified voxelization
├── surface/                     # Surface mesh generation kernels
├── voxel/                       # Voxel computation
├── io/
│   └── loaders.py               # load_protein_input, load_ligand_input
└── cli/                         # Command-line interface
```
