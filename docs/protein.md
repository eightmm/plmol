# Protein API Reference

## Initialization

```python
from plmol import Protein

# From PDB file (recommended)
protein = Protein.from_pdb("protein.pdb", standardize=True, keep_hydrogens=False)

# From mmCIF/PDBx file
protein = Protein.from_mmcif("protein.cif", standardize=True, keep_hydrogens=False)

# From sequence (ESM/Foldseek style - no structure)
protein = Protein.from_sequence("MKFLILLFNILCLFPVLAADNHGVS...")

# Auto-detect PDB vs mmCIF from file extension
protein = Protein.from_structure("protein.cif", chain_id="A")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `standardize` | `bool` | `True` | Standardize PDB (remove non-standard, fix naming) |
| `keep_hydrogens` | `bool` | `False` | Keep hydrogen atoms |

## SASA Backends

SASA feeds the residue 12-dim block, the atom graph's `burial_index`,
`sasa`, `relative_sasa` and `is_polar_sasa`, the surface burial channel and
voxel channel 15.

```python
from plmol import set_sasa_backend, resolve_sasa_backend

resolve_sasa_backend()        # "freesasa" when it is installed
set_sasa_backend("native")    # plmol's own Shrake-Rupley
set_sasa_backend("auto")      # the default: freesasa, else native
```

| Backend | Algorithm | Radii |
|---------|-----------|-------|
| `freesasa` | Lee-Richards, 20 slices | freesasa's ProtOr, atom-type dependent |
| `native` | Shrake-Rupley, 100 points | plmol's `VDW_RADIUS`, element based |

The native occlusion test is exact: it finds the atom pairs whose spheres can
overlap and tests each atom's sample directions against each neighbour, so no
neighbour cap can make an area come out too large. `shrake_rupley` therefore no
longer takes `max_neighbours`, and `DEFAULT_SASA_NEIGHBOURS` and
`SURFACE_BURIAL_KNN` are gone with it.

The cap they set was never observed to bite. At most 18 atoms can reach a given
sample point on a 3260-atom protein, under the 24 that were allowed, and the
areas came out identical at 16, 24 and 48 even on a cluster packed tight enough
that 78% of sample points had more than 24 atoms in reach -- a point is buried
by its nearest neighbours or not at all. What is gone is the caveat, and the
saturation warning that came with it.

**freesasa stays the default.** It is not slower and it is what plmol's
published feature values were computed with. The native path exists so the
library degrades honestly rather than silently: before it, a missing freesasa
turned the residue SASA block into zeros and every `burial_index` into 0.5,
and those were handed back as features.

Agreement between the two, measured on a 3260-atom protein:

| Comparison | Correlation |
|------------|-------------|
| native vs freesasa's own Shrake-Rupley, same radii | 0.994 |
| native vs freesasa default (Lee-Richards, ProtOr radii) | 0.982 per atom, +2% total area |
| residue absolute-area columns | > 0.99 |

The `relative*` columns agree less closely (0.78–0.99) because freesasa
normalises polar, apolar and main-chain areas by separate per-class reference
values, while the native path normalises everything by the residue's
`RESIDUE_MAX_SASA`. Polar/apolar classification is identical: freesasa's
classifier calls N, O and S polar, and the native element rule reproduces that
exactly on protein atoms.

## Spatial Backends

The surface curvature and the atom-to-point mapping ask for k nearest
neighbours. scipy's `cKDTree` answers that when it is installed; a uniform grid
in numpy answers it when it is not.

```python
from plmol import set_spatial_backend, resolve_spatial_backend

resolve_spatial_backend()        # "scipy" when it is installed
set_spatial_backend("native")    # plmol's own uniform grid
set_spatial_backend("auto")      # the default: scipy, else native
```

**scipy stays the default where it is installed**, because a KD-tree is the
right structure for this and scipy's is threaded C. Measured on a 3260-atom
protein whose surface has 15465 points:

| | scipy | native |
|---|---|---|
| `surface` mode, end to end | 276 ms | 452 ms |
| the 39 surface feature columns | — | agree to 4e-06, correlation 1.000000 |

Nothing else in the library goes through a neighbour index. SASA and the
surface point cloud share an exact sphere-occlusion test that needs no tree;
metal coordination and the pocket extractor walk their distances directly.

`plmol[spatial]` installs scipy. Without it every mode still runs, on the grid.

## Featurization Modes

```python
result = protein.featurize(
    mode="all",                              # str or list of modes
    graph_kwargs={"level": "residue"},       # graph options
    surface_kwargs={},                       # surface options
    voxel_kwargs={},                         # voxel options
    backbone_kwargs={"k_neighbors": 30},     # backbone options
)
```

| Mode | Output Key | Description |
|------|-----------|-------------|
| `"graph"` | `"graph"` | Residue/atom-level graph (node_features, edge_index, ...) |
| `"atom_graph"` | `"atom_graph"` | Atom-level graph; the mode spelling of `graph_kwargs={"level": "atom"}`. Not part of `"all"` |
| `"embedding"` | `"embedding"` | Per-residue protein language model embeddings (ESM, Ankh, ProtT5). Not part of `"all"` |
| `"backbone"` | `"backbone"` | Backbone features for inverse folding (dihedrals, kNN, local frames) |
| `"surface"` | `"surface"` | dMaSIF point cloud with per-vertex features |
| `"voxel"` | `"voxel"` | 16-channel 3D voxel grid |
| `"sequence"` | `"sequence"` | Amino acid sequence string or chain dict |
| `"all"` | all above | All modes combined |

Lazy properties:

```python
protein.sequence   # str (single chain) or Dict[str, str] (multi-chain)
protein.graph      # residue-level graph (auto-computed)
protein.surface    # surface point cloud (auto-computed)
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
| `node_features` | tuple of 8 | `Tensor` | Residue scalar features (total 83-dim) |
| `node_vector_features` | tuple of 3 | `Tensor` | Residue vector features (total 31 vectors x 3) |
| `edge_index` | `(2, E)` | `int64` | Sparse edge pairs (source, target) |
| `edge_features` | tuple of 2 | `Tensor` | Edge scalar features (total 39-dim) |
| `edge_vector_features` | tuple of 1 | `Tensor` | Edge vector features (8 vectors x 3) |
| `coords` | `(L, 2, 3)` | `float32` | CA + sidechain centroid coordinates |
| `distance_cutoff` | `float` | -- | Cutoff used (default: 8.0 A) |
| `knn_cutoff` | `Optional[int]` | -- | kNN cutoff used (None if not set) |
| `level` | `str` | -- | `"residue"` |

Edge construction: all residue pairs (i, j) where any of the 4 distances (CA-CA, SC-SC, CA-SC, SC-CA) < `distance_cutoff`. When `knn_cutoff` is set, kNN edges (based on minimum of 4 distance matrices) are unioned with distance edges.

### Node Scalar Features `(L, 83)` -- tuple of 8 tensors

| Index | Tensor | Dim | Features |
|-------|--------|-----|----------|
| `[0:21]` | residue_one_hot | 21 | One-hot: 20 amino acids + UNK |
| `[21:23]` | terminal_flags | 2 | is_N_terminal, is_C_terminal |
| `[23:33]` | self_distance | 10 | Intra-residue pairwise distances among N, CA, C, O, SC (upper triangle) |
| `[33:53]` | degree_feature | 20 | cos/sin of 10 angles: phi, psi, omega, chi1-chi5, backbone_curvature, backbone_torsion |
| `[53:58]` | has_chi_angles | 5 | Binary flags: has chi1, chi2, chi3, chi4, chi5 |
| `[58:70]` | sasa | 12 | SASA: total, polar, apolar, mainchain, sidechain (abs/RESIDUE_MAX_SASA + relative), burial_index (1.0 - relativeTotal), polar_apolar_ratio |
| `[70:78]` | rf_distance | 8 | Forward/reverse neighbor distances: fwd(CA-CA, SC-SC, CA-SC, SC-CA) + rev(same) |
| `[78:83]` | physicochemical | 5 | Residue properties: hydrophobicity (Kyte-Doolittle), volume (Zamyatnin), charge, flexibility, polarity |

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

## Embedding Mode

Per-residue embeddings from a protein language model. One call shape for every
model family, so swapping ESM for Ankh changes a string and nothing else.

```python
from plmol import Protein, list_protein_language_models, plm_dim

list_protein_language_models()
# ['ankh-base', 'ankh-large', 'esm2_t12_35m', 'esm2_t33_650m',
#  'esm3-open', 'esmc_300m', 'esmc_600m', 'prot_t5_xl']

result = protein.featurize(
    mode="embedding",
    embedding_kwargs={"model": "ankh-base", "device": "auto"},
)["embedding"]
```

`"embedding"` is not part of `mode="all"`, so `"all"` never downloads a model.

### Models

| Name | Dim | Backend | Install |
|------|-----|---------|---------|
| `esmc_300m` | 960 | ESM SDK | `pip install 'plmol[esm]'` |
| `esmc_600m` | 1152 | ESM SDK | `pip install 'plmol[esm]'` |
| `esm3-open` | 1536 | ESM SDK | `pip install 'plmol[esm]'` |
| `ankh-base` | 768 | transformers | `pip install 'plmol[plm]'` |
| `ankh-large` | 1536 | transformers | `pip install 'plmol[plm]'` |
| `esm2_t12_35m` | 480 | transformers | `pip install 'plmol[plm]'` |
| `esm2_t33_650m` | 1280 | transformers | `pip install 'plmol[plm]'` |
| `prot_t5_xl` | 1024 | transformers | `pip install 'plmol[plm]'` |

Neither backend is a hard dependency. Asking for a model whose package is
missing raises `DependencyError` naming the extra to install; `plm_dim(name)`
reports the width without loading anything.

Models are cached per `(name, device)`, so featurizing many proteins in a loop
loads the weights once rather than once per protein.

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `embeddings` | `(L, D)` | `float32` | One row per residue, special tokens removed |
| `bos` | `(D,)` | `float32` | Start-of-sequence token, zeros for models without one |
| `eos` | `(D,)` | `float32` | End-of-sequence token, zeros for models without one |
| `model` | `str` | — | The model that produced this |
| `dim` | `int` | — | D |
| `sequence` | `str` | — | The sequence embedded, for checking alignment |

Row `i` of `embeddings` is residue `i` of `sequence`. Special-token counts come
from each model's card and are recorded in the registry; a model returning an
unexpected number of rows raises `InputError` rather than silently misaligning.

### Chains

Chains are separate molecules, so embedding them together inserts a junction
that does not exist. Pass `by_chain=True` to embed each one on its own.

```python
result = protein.featurize(
    mode="embedding",
    embedding_kwargs={"model": "esmc_600m", "by_chain": True},
)["embedding"]

result["by_chain"]["A"]["embeddings"]   # (L_A, D)
result["by_chain"]["B"]["embeddings"]   # (L_B, D)
```

### Without a structure

```python
from plmol import embed_sequence, embed_sequences

embed_sequence("MKTIIALSYIFCLVFA", model="ankh-base")
embed_sequences({"A": "MKTII", "B": "AAAGG"}, model="esmc_600m")
```

`embed_sequences` loads the model once for the whole mapping.

### Registering another model

```python
from plmol import PLMSpec, register_plm

register_plm(PLMSpec(
    name="my-model", backend="huggingface", dim=1024,
    model_id="org/my-protein-lm", family="esm2",
))
```

## Graph Mode -- Atom Level

Two equivalent spellings; the second matches `NucleicAcid` and the ligand graph
views.

```python
result = protein.featurize(
    mode="graph",
    graph_kwargs={"level": "atom", "distance_cutoff": 4.0, "knn_cutoff": None},
)
graph = result["graph"]

result = protein.featurize(
    mode="atom_graph",
    graph_kwargs={"distance_cutoff": 4.0, "knn_cutoff": None},
)
graph = result["atom_graph"]
```

Asking for `mode=["graph", "atom_graph"]` returns a residue graph under
`"graph"` and an atom graph under `"atom_graph"`. `"atom_graph"` is not part of
`mode="all"`, which keeps `"all"` at its previous cost.

For a shape shared with every other plmol graph, see
[Graph View API](graph_view.md): `as_graph` folds the atom graph's token ids
and loose per-edge arrays into `node_tokens`, `node_features` and
`edge_features`.

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | `(A,)` | `int64` | Atom token ID (0-186, for `nn.Embedding`) -- same as `atom_tokens` |
| `atom_to_residue` | `(A,)` | `int64` | Maps each atom to 0-indexed residue index (= `residue_count`) |
| `residue_atom_indices` | `List[List[int]]` | -- | Atom indices per residue (reverse of `atom_to_residue`) |
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
| `[2]` | `burial_index` | 1 | [0, 1] | Burial index (1.0 = fully buried, 0.0 = fully exposed) |
| `[3]` | `is_polar_sasa` | 1 | {0, 1} | 1.0 if polar SASA atom (freesasa classifier) |
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

dMaSIF-style point cloud surface with PCA curvature and chemical features.

```python
result = protein.featurize(mode="surface", surface_kwargs={
    "include_features": True,
    "n_points_per_atom": 100,
})
surface = result["surface"]
```

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `points` | `(V, 3)` | `ndarray` | Surface point positions |
| `verts` | `(V, 3)` | `ndarray` | Alias for points |
| `normals` | `(V, 3)` | `ndarray` | Outward surface normals |
| `features` | `(V, 39)` | `ndarray` | Per-vertex feature vector |
| `feature_names` | `list[str]` | -- | Column names for features |

### Surface Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_features` | True | Compute dMaSIF-style features |
| `n_points_per_atom` | 100 | Points per atom for SAS sampling |
| `probe_radius` | 1.4 | Solvent probe radius (A) |

---

## Voxel Mode

```python
result = protein.featurize(mode="voxel", voxel_kwargs={
    "resolution": 1.0,
    "box_size": 24,
})
voxel = result["voxel"]
```

Channels (16): occupancy, atom type (6), charge, hydrophobicity, HBD, HBA, aromaticity, pos/neg ionizable, backbone, burial_index.

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
from plmol.protein import (
    ProteinFeaturizer,         # Main protein featurizer (PDB parse + cache)
    ResidueFeaturizer,         # Residue-level features
    AtomFeaturizer,            # Atom-level features
    HierarchicalFeaturizer,    # Atom-residue hierarchical (ESM + attention)
    ESMFeaturizer,             # ESM3/ESMC embeddings
    PDBStandardizer,           # PDB cleanup
)
from plmol.parsers import PDBParser  # Low-level PDB parsing
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
