# Protein API Reference

## Initialization

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

## Featurization Modes

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

Lazy properties:

```python
protein.sequence   # str (single chain) or Dict[str, str] (multi-chain)
protein.graph      # residue-level graph (auto-computed)
protein.surface    # surface mesh (auto-computed)
```

---

## Graph Mode -- Residue Level

```python
result = protein.featurize(
    mode="graph",
    graph_kwargs={"level": "residue", "distance_cutoff": 8.0, "knn_cutoff": None},
)
graph = result["graph"]
```

### graph_kwargs

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | `str` | `"residue"` | `"residue"` or `"atom"` |
| `distance_cutoff` | `float` | `8.0` | Distance cutoff for edges (A) |
| `knn_cutoff` | `Optional[int]` | `None` | k-nearest neighbors. If given, union with distance edges for connectivity |

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | tuple of 8 | `Tensor` | Residue scalar features (total 81-dim) |
| `node_vector_features` | tuple of 3 | `Tensor` | Residue vector features (total 31 vectors x 3) |
| `edge_index` | `(2, E)` | `int64` | Sparse edge pairs (source, target) |
| `edge_features` | tuple of 2 | `Tensor` | Edge scalar features (total 39-dim) |
| `edge_vector_features` | tuple of 1 | `Tensor` | Edge vector features (8 vectors x 3) |
| `coords` | `(L, 2, 3)` | `float32` | CA + sidechain centroid coordinates |
| `distance_cutoff` | `float` | -- | Cutoff used (default: 8.0 A) |
| `knn_cutoff` | `Optional[int]` | -- | kNN cutoff used (None if not set) |
| `level` | `str` | -- | `"residue"` |

Edge construction: all residue pairs (i, j) where any of the 4 distances (CA-CA, SC-SC, CA-SC, SC-CA) < `distance_cutoff`. When `knn_cutoff` is set, kNN edges (based on minimum of 4 distance matrices) are unioned with distance edges.

### Node Scalar Features `(L, 81)` -- tuple of 8 tensors

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

### Node Vector Features `(L, 31, 3)` -- tuple of 3 tensors

| Index | Tensor | Vectors | Features |
|-------|--------|---------|----------|
| `[0:20]` | self_vector | 20 | Intra-residue pairwise direction vectors among N, CA, C, O, SC |
| `[20:28]` | rf_vector | 8 | Forward/reverse neighbor direction vectors (CA-CA, SC-SC, CA-SC, SC-CA x 2) |
| `[28:31]` | local_frames | 3 | Local N-CA-C coordinate frame (3 orthonormal basis vectors) |

### Edge Scalar Features `(E, 39)` -- tuple of 2 tensors

| Index | Tensor | Dim | Features |
|-------|--------|-----|----------|
| `[0:4]` | distance | 4 | CA-CA, SC-SC, CA-SC, SC-CA distances (Angstrom) |
| `[4:39]` | relative_position | 35 | One-hot sequence separation: d=0, 1, ..., 32, >32, cross-chain, UNK |

### Edge Vector Features `(E, 8, 3)` -- tuple of 1 tensor

| Index | Tensor | Vectors | Features |
|-------|--------|---------|----------|
| `[0:8]` | interaction_vectors | 8 | CA_i->CA_j, CA_j->CA_i, CA_i->SC_j, CA_j->SC_i, SC_i->CA_j, SC_j->CA_i, SC_i->SC_j, SC_j->SC_i |

---

## Graph Mode -- Atom Level

```python
result = protein.featurize(
    mode="graph",
    graph_kwargs={"level": "atom", "distance_cutoff": 4.0, "knn_cutoff": None},
)
graph = result["graph"]
```

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | `(A,)` | `int64` | Atom token ID (0-186, for `nn.Embedding`) -- same as `atom_tokens` |
| Node token features | 3 tensors | `int64` | Integer tokens for embedding layers |
| Node scalar features | 11 tensors | `float32` | Continuous per-atom features (total 11-dim) |
| `coords` | `(A, 3)` | `float32` | Atom 3D coordinates |
| `edge_index` | `(2, E)` | `int64` | Sparse edge pairs (source, target) |
| Edge features | 4 tensors | `float32` | Per-edge features (total 6-dim) |
| `distance_cutoff` | `float` | -- | Cutoff used (default: 4.0 A) |
| `knn_cutoff` | `Optional[int]` | -- | kNN cutoff used (None if not set) |
| `level` | `str` | -- | `"atom"` |

Edge construction: all atom pairs within `distance_cutoff`. When `knn_cutoff` is set, kNN edges are unioned with distance edges via `torch.topk`.

### Node Token Features `(A,)` -- int64, for `nn.Embedding`

| Key | Vocab Size | Description |
|-----|-----------|-------------|
| `atom_tokens` | 187 | Residue-atom pair token (e.g. ALA-CA, GLY-N). Use `nn.Embedding(187, d)` |
| `residue_token` | 22 | Residue type per atom (20 AA + Metal + UNK) |
| `atom_element` | 19 | Element type per atom (H, C, N, O, S, P, Se, metals, UNK) |

### Node Scalar Features -- float32, total 11-dim

| Index | Key | Dim | Range | Description |
|-------|-----|-----|-------|-------------|
| `[0]` | `sasa` | 1 | [0, ~) | Per-atom absolute SASA (A^2) |
| `[1]` | `relative_sasa` | 1 | [0, 1] | SASA / residue_max_sasa (Tien et al. 2013) |
| `[2]` | `b_factor` | 1 | [0, 1] | Normalized B-factor (B / 100, capped at 1.0) |
| `[3]` | `b_factor_zscore` | 1 | (~) | Per-chain B-factor z-score: (b - chain_mean) / chain_std |
| `[4]` | `is_backbone` | 1 | {0, 1} | 1.0 if backbone atom (N, CA, C, O), 0.0 if sidechain |
| `[5]` | `formal_charge` | 1 | [-0.5, 1] | Partial charge at physiological pH |
| `[6]` | `is_hbond_donor` | 1 | {0, 1} | 1.0 if H-bond donor |
| `[7]` | `is_hbond_acceptor` | 1 | {0, 1} | 1.0 if H-bond acceptor |
| `[8:11]` | `secondary_structure` | 3 | {0, 1} | One-hot [helix, sheet, coil] from phi/psi Ramachandran |

### Edge Features -- total 6-dim

| Index | Key | Shape | Type | Description |
|-------|-----|-------|------|-------------|
| `[0]` | `edge_distances` | `(E,)` | `float32` | Euclidean distance (A) |
| `[1]` | `same_residue` | `(E,)` | `float32` | 1.0 if both atoms in same residue |
| `[2]` | `sequence_separation` | `(E,)` | `float32` | \|residue_i - residue_j\|, capped at 32 |
| `[3:6]` | `unit_vector` | `(E, 3)` | `float32` | Normalized direction vector src -> dst |

### Metadata

| Key | Type | Description |
|-----|------|-------------|
| `atom_name` | `list[str]` | PDB atom names (e.g. "CA", "CB", "OG") |
| `chain_label` | `list[str]` | Chain identifiers (e.g. "A", "B") |

---

## Backbone Mode

For inverse folding models (ProteinMPNN, ESM-IF, GVP, PiFold).

```python
result = protein.featurize(
    mode="backbone",
    backbone_kwargs={"k_neighbors": 30},
)
backbone = result["backbone"]
```

### Node Features

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `backbone_coords` | `(L, 4, 3)` | `float32` | N, CA, C, O coordinates |
| `cb_coords` | `(L, 3)` | `float32` | Virtual CB (ProteinMPNN geometry) |
| `dihedrals` | `(L, 3)` | `float32` | phi, psi, omega (radians, chain-boundary-aware) |
| `dihedrals_sincos` | `(L, 6)` | `float32` | sin/cos encoding |
| `dihedrals_mask` | `(L, 3)` | `bool` | True where dihedral is valid |
| `orientation_frames` | `(L, 3, 3)` | `float32` | N-CA-C local coordinate frames |
| `residue_types` | `(L,)` | `int64` | Residue type (0-20) |
| `chain_ids` | `(L,)` | `int64` | Integer chain ID |
| `residue_mask` | `(L,)` | `bool` | True if all 4 backbone atoms present |

### kNN Graph (E = L * k)

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

### Metadata

| Key | Type | Description |
|-----|------|-------------|
| `num_residues` | `int` | Total residue count (L) |
| `num_chains` | `int` | Total chain count |
| `k_neighbors` | `int` | k used for kNN graph |

---

## Surface Mode

```python
result = protein.featurize(mode="surface", surface_kwargs={
    "mode": "mesh",           # "mesh" (default) or "point_cloud"
    "grid_density": 2.5,
    "include_features": True,
})
surface = result["surface"]
```

### Output (mesh mode)

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

### Surface Parameters

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

## Voxel Mode

```python
result = protein.featurize(mode="voxel", voxel_kwargs={
    "resolution": 1.0,
    "box_size": 24,
})
voxel = result["voxel"]
```

Channels (16): occupancy, atom type (6), charge, hydrophobicity, HBD, HBA, aromaticity, pos/neg ionizable, backbone, b_factor.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `center` | None | Grid center (3,). None = protein centroid |
| `resolution` | 1.0 | Angstrom per voxel |
| `box_size` | 24 | Grid dimension per axis. None for adaptive |
| `padding` | 4.0 | Padding (A) when box_size is None |
| `sigma_scale` | 0.5 | VdW radius multiplier for Gaussian sigma |
| `cutoff_sigma` | 2.0 | Gaussian cutoff in sigma units |

---

## Sequence Mode

```python
result = protein.featurize(mode="sequence")
seq = result["sequence"]
# Single chain: str "MKFLIL..."
# Multi-chain: Dict[str, str] {"A": "MKFLIL...", "B": "GRPEWK..."}
```

---

## Pocket Featurization

Extract and featurize the binding pocket around a ligand.

```python
pocket = protein.featurize_pocket(
    ligand=ligand_mol,   # RDKit Mol, file path, or Ligand object
    distance_cutoff=6.0, # Angstrom
    mode="graph",
    graph_kwargs={"level": "residue"},
)
# Returns same structure as protein.featurize() for the pocket subset
```

---

## Low-Level Featurizers

```python
from plmol import (
    ProteinFeaturizer,         # Main protein featurizer (PDB parse + cache)
    ResidueFeaturizer,         # Residue-level features
    AtomFeaturizer,            # Atom-level features
    HierarchicalFeaturizer,    # Atom-residue hierarchical (ESM + attention)
    ESMFeaturizer,             # ESM3/ESMC embeddings
    PDBStandardizer,           # PDB cleanup
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
