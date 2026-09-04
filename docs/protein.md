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

## SASA

SASA feeds the residue 11-dim block, the atom graph's `burial_index`, `sasa`,
`relative_sasa` and `is_polar_sasa`, the surface burial channel and voxel
channel 15.

plmol computes it itself: Shrake-Rupley, 100 sample points per atom, a 1.4 A
probe, and its own element-based `VDW_RADIUS` table. The occlusion test
considers every overlapping neighbour rather than a fixed number of nearest
ones, so no sampling cap can make an area come out too large. Areas are cached
on the coordinates, so a Protein asked for `graph`, `atom_graph`, `voxel` and
`surface` computes them once rather than four times.

**freesasa was removed in 0.4.0.** It had been the default when installed;
this path is 1.3 to 2.0 times faster on every mode that uses SASA, and one
implementation the library owns beats two that disagree.

| Mode | 0.3.x with freesasa | 0.4.0 | |
|------|--------------------|-------|---|
| `graph` | 130 ms | 64 ms | 2.0x |
| `atom_graph` | 100 ms | 33 ms | 3.0x |
| `surface` | 185 ms | 114 ms | 1.6x |
| `voxel` | 128 ms | 56 ms | 2.3x |
| `complex` (all) | 207 ms | 142 ms | 1.5x |

Minimum of nine interleaved runs on a 3260-atom protein, idle machine.

Values from 0.3.x and earlier were freesasa's, computed with Lee-Richards and
ProtOr radii. Per atom the two correlate at r=0.982 and the totals differ by
2%; most of that is the radius table rather than the algorithm, since matching
against freesasa's own Shrake-Rupley leaves r=0.979. Every SASA-derived column
moved at 0.4.0 and nothing else did.

Polar and apolar are split on the element -- N, O and S are polar -- read from
the parser's element field rather than the first letter of the atom name, so
the SE of a selenomethionine is selenium. That rule agreed with freesasa's
classifier on 100% of a 3260-atom structure while both existed.

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

**scipy stays the default where it is installed.** freesasa was removed in
0.4.0 because plmol's own path was faster; scipy is not, and the gap is
algorithmic rather than a matter of tuning -- a tree prunes to about
`k log n` candidates per query where a uniform grid has to enumerate about
`8k` of them. Measured on a 3260-atom protein whose surface has 15465 points,
idle machine:

| | scipy | native |
|---|---|---|
| `knn`, k=80, on the point cloud | 12.5 ms | 152.4 ms |
| `surface` mode, end to end | 116 ms | 329 ms |
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
| `"cavity"` | `"cavity"` | Enclosed spaces found without a ligand, largest first |
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
| `node_features` | tuple of 8 | array | Residue scalar features (total 82-dim) |
| `node_vector_features` | tuple of 3 | array | Residue vector features (total 31 vectors x 3) |
| `edge_index` | `(2, E)` | `int64` | Sparse edge pairs (source, target) |
| `edge_features` | tuple of 2 | array | Edge scalar features (total 39-dim) |
| `edge_vector_features` | tuple of 1 | array | Edge vector features (8 vectors x 3) |
| `coords` | `(L, 2, 3)` | `float32` | CA + sidechain centroid coordinates |
| `distance_cutoff` | `float` | -- | Cutoff used (default: 8.0 A) |
| `knn_cutoff` | `Optional[int]` | -- | kNN cutoff used (None if not set) |
| `level` | `str` | -- | `"residue"` |

Edge construction: all residue pairs (i, j) where any of the 4 distances (CA-CA, SC-SC, CA-SC, SC-CA) < `distance_cutoff`. When `knn_cutoff` is set, kNN edges (based on minimum of 4 distance matrices) are unioned with distance edges.

### Node Scalar Features `(L, 82)` -- tuple of 8 arrays

| Index | Array | Dim | Features |
|-------|--------|-----|----------|
| `[0:21]` | residue_one_hot | 21 | One-hot: 20 amino acids + UNK |
| `[21:23]` | terminal_flags | 2 | is_N_terminal, is_C_terminal |
| `[23:33]` | self_distance | 10 | Intra-residue pairwise distances among N, CA, C, O, SC (upper triangle) |
| `[33:53]` | degree_feature | 20 | cos/sin of 10 angles: phi, psi, omega, chi1-chi5, backbone_curvature, backbone_torsion. An angle that would span two residues no peptide bond joins — a chain boundary, or the two sides of a missing loop — is zero, the same encoding the first and last residue already carry |
| `[53:58]` | has_chi_angles | 5 | Binary flags: has chi1, chi2, chi3, chi4, chi5 |
| `[58:69]` | sasa | 11 | polar, apolar, mainchain, sidechain over `RESIDUE_MAX_SASA` (what fraction of the residue's surface each class is); relativeTotal, relativePolar, relativeApolar, relativeMainChain, relativeSideChain over `RESIDUE_MAX_CLASS_SASA` (how exposed each class is against how exposed it could be); burial_index (1.0 - relativeTotal); polar_apolar_ratio |
| `[69:77]` | rf_distance | 8 | Forward/reverse neighbor distances: fwd(CA-CA, SC-SC, CA-SC, SC-CA) + rev(same) |
| `[77:82]` | physicochemical | 5 | Residue properties: hydrophobicity (Kyte-Doolittle), volume (Zamyatnin), charge, flexibility, polarity |

> **The SASA columns depend on how the structure is oriented.** plmol samples
> each atom's sphere on a lattice fixed in space rather than one carried with
> the molecule, so rotating a structure moves them. How far it moves them is
> one sample point: at the default 100 points a point is worth about 1.2 A², and
> over eight random rotations of the example protein the per-atom range averages
> 1.41 A² (median 1.21, worst 8.45) while the total moves by 0.95%. At 1000
> points the per-atom range averages 0.27 A² and the total moves by 0.14%.
> Translating a structure changes nothing.
>
> Relative figures overstate this, because the atoms with the largest ratios are
> the ones with almost no surface. 493 of 3260 atoms come out exactly zero in one
> orientation and non-zero in another, and the largest area any of them reaches
> is 4.8 A². Among genuinely exposed atoms — mean area above 10 A² — the spread
> is 18% at 100 points and 3% at 1000. In the residue block the relative columns
> `[63:68]` move by up to 0.14, and `polar_apolar_ratio` `[68]` swings the whole
> 0–1 range on 10 of 416 residues — again residues with no measurable surface,
> where the ratio is `0/1e-8` when nothing survives the occlusion test and `1.0`
> when one polar sample point does.
>
> This is the discretisation floor of Shrake–Rupley rather than a defect in this
> implementation; freesasa, Biopython and MDTraj all default to 100 points too.
> `burial_index` and `relative_sasa` on the atom graph, the surface burial
> channel and the voxel's inherit the same behaviour. The featurizers do not
> expose `n_points`; `plmol.sasa.shrake_rupley` and
> `plmol.sasa.native_structure_result` do.
>
> Orienting the lattice by a frame taken from the coordinates would make the
> answer deterministic but no more accurate, and the frame is only as stable as
> the gap between the inertia eigenvalues — small for a globular protein, zero
> for a symmetric oligomer — so it would trade orientation dependence for a
> worse perturbation dependence. What would cure it is an analytic area, the
> exact area of a sphere outside the union of its neighbours by Gauss–Bonnet,
> which is both rotation-invariant and exact and which would change every SASA
> value plmol reports.
>
> A guard on the ratio does not cure it either, and this was measured rather
> than assumed. Refusing to divide until the residue has at least *k* sample
> points' worth of area, for *k* from 1 to 20: at k=1 all ten residues still
> swing; at k=20 — which zeroes the ratio for 156 of the 416 residues — one
> still swings the full range and six swing more than half. The count is not
> even monotone in *k*, because the threshold becomes a boundary of its own that
> the rotation crosses, trading one discontinuity for another.
>
> Sub-precision movement is not the cause: shifting every atom by 0.0004 Å,
> which re-rounds all three decimals a PDB file stores, changes none of these
> columns at all.

### Node Vector Features `(L, 31, 3)` -- tuple of 3 arrays

| Index | Array | Vectors | Features |
|-------|--------|---------|----------|
| `[0:20]` | self_vector | 20 | Intra-residue pairwise direction vectors among N, CA, C, O, SC |
| `[20:28]` | rf_vector | 8 | Forward/reverse neighbor direction vectors (CA-CA, SC-SC, CA-SC, SC-CA x 2) |
| `[28:31]` | local_frames | 3 | Local N-CA-C coordinate frame (3 orthonormal basis vectors) |

### Edge Scalar Features `(E, 39)` -- tuple of 2 arrays

| Index | Array | Dim | Features |
|-------|--------|-----|----------|
| `[0:4]` | distance | 4 | CA-CA, SC-SC, CA-SC, SC-CA distances (Angstrom) |
| `[4:39]` | relative_position | 35 | One-hot sequence separation: d=0, 1, ..., 32, >32, cross-chain, UNK |

### Edge Vector Features `(E, 8, 3)` -- tuple of 1 array

| Index | Array | Vectors | Features |
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
| `atom_to_residue` | `(A,)` | `int64` | Maps each atom to 0-indexed residue index (= `residue_count`); chain, residue number and insertion code all start a new residue, so 100 and 100A are two |
| `residue_atom_indices` | `List[List[int]]` | -- | Atom indices per residue (reverse of `atom_to_residue`) |
| Node token features | 3 arrays | `int64` | Integer tokens for embedding layers |
| Node scalar features | 11 arrays | `float32` | Continuous per-atom features (total 11-dim) |
| `coords` | `(A, 3)` | `float32` | Atom 3D coordinates |
| `edge_index` | `(2, E)` | `int64` | Sparse edge pairs (source, target) |
| Edge features | 4 arrays | `float32` | Per-edge features (total 6-dim) |
| `distance_cutoff` | `float` | -- | Cutoff used (default: 4.0 A) |
| `knn_cutoff` | `Optional[int]` | -- | kNN cutoff used (None if not set) |
| `level` | `str` | -- | `"atom"` |

Edge construction: all atom pairs within `distance_cutoff`. When `knn_cutoff` is set, kNN edges are unioned with distance edges.

### Node Token Features `(A,)` -- int64, for `nn.Embedding`

| Key | Vocab Size | Description |
|-----|-----------|-------------|
| `atom_tokens` | 187 | Residue-atom pair token (e.g. ALA-CA, GLY-N). Use `nn.Embedding(187, d)` |
| `residue_token` | 22 | Residue type per atom (20 AA + Metal + UNK) |
| `atom_element` | 19 | Element type per atom (H, C, N, O, S, P, Se, metals, UNK) |

### Node Scalar Features

The raw `atom_graph` dict returns these as **separate keys**, not as one
array. There are eleven of them:

| Key | Dim | Range | Description |
|-----|-----|-------|-------------|
| `sasa` | 1 | [0, ~) | Per-atom absolute SASA (A^2) |
| `relative_sasa` | 1 | [0, 1] | SASA / residue_max_sasa (Tien et al. 2013) |
| `burial_index` | 1 | [0, 1] | Burial index (1.0 = fully buried, 0.0 = fully exposed) |
| `is_polar_sasa` | 1 | {0, 1} | 1.0 when the element is N, O or S |
| `is_backbone` | 1 | {0, 1} | 1.0 if backbone atom (N, CA, C, O), 0.0 if sidechain |
| `formal_charge` | 1 | [-0.5, 1] | Partial charge at physiological pH |
| `is_hbond_donor` | 1 | {0, 1} | 1.0 if H-bond donor |
| `is_hbond_acceptor` | 1 | {0, 1} | 1.0 if H-bond acceptor |
| `secondary_structure` | 3 | {0, 1} | One-hot [helix, sheet, coil] from phi/psi Ramachandran. A residue whose neighbour is not actually peptide-bonded to it -- a chain end, or either side of a missing loop -- stays coil |

`as_graph` concatenates them into `node_features`, **10-dim**, in this order.
`relative_sasa` is left out because `burial_index` is exactly `1 -
relative_sasa`; it stays available under its own key in the raw dict.

| Index | Key | Dim |
|-------|-----|-----|
| `[0:1]` | `burial_index` | 1 |
| `[1:2]` | `formal_charge` | 1 |
| `[2:3]` | `is_backbone` | 1 |
| `[3:4]` | `is_hbond_acceptor` | 1 |
| `[4:5]` | `is_hbond_donor` | 1 |
| `[5:6]` | `is_polar_sasa` | 1 |
| `[6:7]` | `sasa` | 1 |
| `[7:10]` | `secondary_structure` | 3 |

### Edge Features -- total 6-dim

| Index | Key | Shape | Type | Description |
|-------|-----|-------|------|-------------|
| `[0]` | `edge_distances` | `(E,)` | `float32` | Euclidean distance (A) |
| `[1]` | `same_residue` | `(E,)` | `float32` | 1.0 if both atoms in same residue |
| `[2]` | `sequence_separation` | `(E,)` | `float32` | \|residue number_i - residue number_j\| along one chain, capped at 32. Two atoms on different chains have no sequence relationship and get the cap |
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

## Cavities

Where the enclosed spaces are, with nothing bound. `extract_pocket` answers
"which residues line this ligand"; this answers the question you have before
there is a ligand.

```python
from plmol import Protein, detect_cavities

cavities = Protein.from_pdb("apo.pdb").featurize(mode="cavity")["cavity"]
cavities["volume"][0]        # the largest, in cubic Angstrom
cavities["center"][0]        # a docking box centre
```

| Key | Shape | Description |
|-----|-------|-------------|
| `num_cavities` | `int` | How many were found |
| `center` | `(C, 3)` `float32` | Centroid of each cavity's grid points |
| `volume` | `(C,)` `float32` | Cubic Angstrom |
| `buriedness` | `(C,)` `float32` | Mean enclosed-axis count, 0 to 7 |
| `extent` | `(C, 3)` `float32` | Bounding box side lengths |
| `num_lining_residues` | `(C,)` `int64` | Residues touching the cavity |
| `points` | list of `(M, 3)` | The grid points themselves |
| `lining_atom_indices`, `lining_residues` | lists | What lines each cavity |

Rows are sorted by volume, so the top `k` is a slice.

### How it works

The method is LIGSITE's. The structure goes on a 1 Angstrom grid and every
point inside an atom's van der Waals sphere plus the probe is marked. From each
remaining point, seven axes are scanned -- the three grid axes and the four
body diagonals. An axis is *enclosed* when there is structure in both
directions along it within `scan_length`, meaning the point sits between two
walls rather than out in bulk solvent. Points enclosed on `psp_threshold` axes
are cavity; adjacent cavity points are one cavity.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `resolution` | 1.0 A | Finer resolves smaller pockets, at the cube of the ratio |
| `psp_threshold` | 5 of 7 | 7 finds only sealed voids, 3 also catches open grooves |
| `scan_length` | 8.0 A | How far to look for the far wall; the middle of a wider cavity is invisible to a shorter scan |
| `min_points` | 10 | Smaller clusters are dropped as noise |

On a 3260-atom protein this takes about 120 ms and finds 13 cavities.

What it is not: cavity detection ranks by volume, not by druggability. The
largest cavity is where a ligand usually sits -- on the bundled 10gs structure
it is the one the crystallographic ligand occupies, and its lining covers 92%
of the residues `extract_pocket` finds from that ligand -- but a large cleft
can merge neighbouring grooves.

Tightening it is not free. Measured on 10gs against the residues
`extract_pocket` finds from the ligand:

| `psp_threshold` | `scan_length` | Rank of the ligand's cavity | Its coverage of the pocket |
|---|---|---|---|
| 4 | 8 A | 0 | 100% |
| **5** | **8 A** | **0** | **92%** |
| 6 | 8 A | 0 | 50% |
| 6 | 10 A | 0 | 88% |
| 7 | 8 A | 2 | 33% |

At 6 the site is still ranked first but half of it stops counting as enclosed,
and a longer scan is needed to get it back. At 7 the pocket is not enclosed
enough to rank first at all. Raise the threshold with `scan_length` together,
or leave both alone.

```python
# featurize_cavity is featurize_pocket for a structure with nothing bound
pocket = Protein.from_pdb("apo.pdb").featurize_cavity(0, mode="graph")
pocket["graph"]      # the lining residues, as any other residue graph
pocket["cavity"]     # the cavity itself, one row
```

## Pocket (ligand-guided)

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
# data.atom_tokens:        (N_atoms,) int array
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
