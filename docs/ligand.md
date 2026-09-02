# Ligand API Reference

## Initialization

```python
from plmol import Ligand

# From SMILES
ligand = Ligand.from_smiles("CCO", add_hs=False)

# From SDF file
ligand = Ligand.from_sdf("ligand.sdf")

# Generate 3D conformer (needed for surface/voxel)
ligand.generate_conformer()
```

## Featurization Modes

```python
result = ligand.featurize(
    mode="all",                    # str or list of modes; "all" uses default modes
    graph_kwargs={},               # graph options
    surface_kwargs={},             # surface options
    fingerprint_kwargs={},         # fingerprint options
    generate_conformer=False,      # auto-generate 3D if missing
    add_hs=None,                   # hydrogen override
)
```

Mode strings are normalized via `normalize_modes()` from `plmol.specs`. Invalid modes raise `InputError`.

| Mode | Output Key | Description |
|------|-----------|-------------|
| `"graph"` | `"graph"` | Dense adjacency graph (node_features, adjacency, bond_mask, ...) |
| `"bond_graph"` | `"bond_graph"` | Bond-wise (line) graph: bonds are nodes, shared atoms are edges |
| `"fingerprint"` | `"fingerprint"` | Descriptors + ECFP/Morgan, MACCS, RDKit FP, AtomPair, ErG |
| `"descriptor"` | `"descriptor"` | 62-dim normalized descriptor vector + descriptor names |
| `"fragment"` | `"fragment"` | Fragmentation result (rotatable-bond by default; BRICS optional) |
| `"surface"` | `"surface"` | dMaSIF point cloud surface (requires 3D conformer) |
| `"voxel"` | `"voxel"` | 16-channel 3D voxel grid (requires 3D conformer) |
| `"morgan"` | `"fingerprint"` + `"morgan"` | Backward-compatible alias for Morgan/ECFP4; prefer `fingerprint_kwargs={"include_fps": ["morgan"]}` |
| `"smiles"` | `"smiles"` | Canonical SMILES string |
| `"sequence"` | `"sequence"` | Same as SMILES (ligand alias) |
| `"all"` | graph + fingerprint + smiles + sequence | Default modes. surface/voxel/fragment/descriptor/morgan must be explicitly requested |

`LigandFeaturizer` follows the same mode names and also exposes lower-level getters for direct graph, surface, voxel, descriptor, and fingerprint calls.

Lazy properties:

```python
ligand.smiles       # str
ligand.sequence     # str (alias for smiles)
ligand.graph        # dict (auto-computed)
ligand.surface      # dict (auto-computed, needs conformer)
ligand.fingerprint  # dict (auto-computed)
ligand.fragment     # dict (auto-computed)
```

---

## Graph Mode

`Ligand.featurize(mode="graph")` returns a **dense adjacency** representation.

```python
result = ligand.featurize(mode="graph", graph_kwargs={
    "distance_cutoff": None,   # Optional 3D distance cutoff for spatial edges
    "knn_cutoff": None,        # Optional k-nearest neighbors for spatial edges
})
graph = result["graph"]
```

### graph_kwargs

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distance_cutoff` | `Optional[float]` | `None` | 3D distance cutoff for spatial edges. None = bond edges only |
| `knn_cutoff` | `Optional[int]` | `None` | k-nearest neighbors for spatial edges. Unioned with distance edges |
| `generate_conformer` | `bool` | `True` | Generate transient 3D coordinates for graph pair features when missing. Set `False` for zero coords |

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | `(N, 98)` | `float32` | Per-atom feature vector |
| `adjacency` | `(N, N, 37)` | `float32` | Dense adjacency (bond + pair channels) |
| `bond_mask` | `(N, N)` | `bool` | True where chemical bond exists |
| `distance_matrix` | `(N, N)` | `float32` | Euclidean distance (0 if no 3D) |
| `distance_bounds` | `(N, N, 2)` | `float32` | DG lower/upper distance bounds |
| `coords` | `(N, 3)` | `float32` | 3D coordinates (0 if no conformer) |
| `atom_to_fragment` | `(N,)` | `int64` | Atom → fragment index mapping |
| `fragment_atom_indices` | `List[List[int]]` | — | Fragment → atom indices (reverse) |
| `fragment_adjacency` | `(F, F)` | `int64` | Fragment-level connectivity |
| `num_fragments` | `int` | — | Number of fragments (F) |
| `molecule_features` | `(62,)` | `float32` | Whole-molecule descriptors |

Sparse conversion:

```python
from plmol import LigandFeaturizer
edge_index, edge_features = LigandFeaturizer.adjacency_to_bond_edges(graph["adjacency"])
# edge_index: (2, E)  edge_features: (E, 37)
```

### Implementation

The graph featurizer is split across three files:

- **`graph.py`** (`MoleculeGraphFeaturizer`): Orchestrator class that composes atom and edge features via mixins.
- **`graph_atom_features.py`** (`AtomFeatureMixin`): Per-atom feature extraction (ring analysis, stereochemistry, partial charges, physical properties, topological features, SMARTS matching, neighborhood statistics).
- **`graph_edge_features.py`** (`EdgeFeatureMixin`): Bond-level and pairwise feature extraction (bond features, pair features, distance matrices, distance bounds).

The public API is unchanged: `MoleculeGraphFeaturizer.featurize(mol)` returns `(node_dict, edge_dict, adjacency_matrix)`.

### Node Features `(N, 98)`

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
| `[78:94]` | Extended neighborhood | 16 | 1-hop and 2-hop neighborhood statistics (8 features each) |
| `[94:96]` | Crippen contributions | 2 | Per-atom logP, molar refractivity (Wildman-Crippen) |
| `[96:97]` | TPSA contribution | 1 | Per-atom topological polar surface area |
| `[97:98]` | Labute ASA | 1 | Per-atom approximate surface area (Labute) |

### Adjacency Channels `(N, N, 37)`

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

## Bond Graph Mode

Inverts the atom-wise graph. Every bond becomes a node, and two bond nodes are
connected when they share an atom — so the roles of atoms and bonds are swapped
relative to `graph` mode.

```python
bond_graph = ligand.featurize(mode="bond_graph")["bond_graph"]
```

`bond_graph` accepts the same `graph_kwargs` as `graph`, because it is derived
from that output. Bond node features are read directly from the atom graph's
dense adjacency rather than recomputed, so the two views always agree.

`"bond_graph"` is not part of `mode="all"`; request it explicitly.

### Output

`B` = number of bonds, `E` = number of bond-graph edges.

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | `(B, 37)` | `float32` | Per-bond features, i.e. `adjacency[begin, end]` from the atom graph |
| `edge_index` | `(2, E)` | `int64` | Bond pairs sharing an atom, both directions |
| `edge_features` | `(E, 100)` | `float32` | Shared atom's 98-dim features + `[cos(theta), theta/pi]` |
| `edge_shared_atom` | `(E,)` | `int64` | Atom index bridging each edge |
| `bond_index` | `(B, 2)` | `int64` | Atom pair each bond node came from |
| `atom_to_bonds` | `List[List[int]]` | — | Atom → incident bond indices (reverse of `bond_index`) |
| `coords` | `(B, 3)` | `float32` | Bond midpoints (0 if no conformer) |
| `adjacency` | `(B, B)` | `bool` | Bond-node connectivity |
| `num_bonds` | `int` | — | B |
| `num_bond_edges` | `int` | — | E |

Bond nodes follow RDKit bond index order. Edge count satisfies
`E = 2 * sum over atoms of C(degree, 2)`.

### Angle Features

`edge_features[:, 98:]` holds the angle the two bonds subtend at the shared atom:
`cos(theta)` in `[-1, 1]` and `theta / pi` in `[0, 1]`. Both are zero when the
molecule has no 3D conformer — SMILES input generates one by default, so pass
`graph_kwargs={"generate_conformer": False}` to opt out.

### Edge Cases

| Input | Result |
|-------|--------|
| No bonds (`"C"`, `"[Na+]"`) | `B = 0`, `E = 0`, empty arrays with correct trailing dims |
| One bond (`"CO"`) | `B = 1`, `E = 0` |
| Disconnected fragments (`"CC.CC"`) | Bonds in different fragments share no atoms, so no edges between them |
| Rings | Ring-closure bonds are ordinary bond nodes; cyclopropane gives K3 |

### Low-Level Function

```python
from plmol import build_bond_graph, MoleculeFeaturizer

graph = ligand.featurize(mode="graph")["graph"]
bond_graph = build_bond_graph(
    MoleculeFeaturizer(smiles).get_rdkit_mol(),
    adjacency=graph["adjacency"],
    node_features=graph["node_features"],
    coords=graph["coords"],
)
```

The molecule must be the one the atom graph was built from. `graph` mode
canonicalizes atom order, so passing a differently ordered copy raises
`InputError` on an atom-count mismatch and would otherwise map bonds wrongly.

## Fingerprint Mode

```python
result = ligand.featurize(mode="fingerprint")
fp = result["fingerprint"]
```

Fingerprint generation is handled by `FingerprintGenerator` (`ligand/fingerprint_generator.py`), which uses lazy imports for optional dependencies (Pharm2D, Avalon).

### Default Output

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

`"morgan"` is accepted as an alias for `"ecfp4"` in `include_fps`:

```python
result = ligand.featurize(
    mode="fingerprint",
    fingerprint_kwargs={"include_fps": ["morgan"]},
)
ecfp4 = result["fingerprint"]["ecfp4"]
```

`mode="morgan"` remains as a legacy convenience mode. It now also returns the
standard `"fingerprint"` key containing only descriptors + `ecfp4`.

### Additional Fingerprints (via `include_fps`)

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

### Descriptors `(62,)`

All values normalized to [0, 1].

For descriptors without fingerprints:

```python
desc = ligand.featurize(mode="descriptor")["descriptor"]
values = desc["descriptors"]          # (62,)
names = desc["descriptor_names"]      # length 62
```

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
| `[37:40]` | Ring structure | 3 | n_atom_rings, max_ring_size, avg_ring_size |
| `[40:44]` | Charge distribution | 4 | max_partial_charge, min_partial_charge, max_abs_partial_charge, min_abs_partial_charge |
| `[44:50]` | ADMET filters | 6 | veber_violations, ghose_violations, egan_violations, muegge_violations, pfizer_375_alert, gsk_4400_pass |
| `[50:52]` | Structural alerts | 2 | pains_alert_count, brenk_alert_count |
| `[52:56]` | Structural complexity | 4 | num_amide_bonds, num_stereocenters, num_spiro_atoms, num_bridgehead_atoms |
| `[56:57]` | Solubility | 1 | esol_logs (Delaney equation) |
| `[57:62]` | 3D shape | 5 | npr1, npr2, asphericity, eccentricity, radius_of_gyration |

---

## Fragment Mode

Fragments a molecule into a fragment-level graph. The default method cuts
rotatable bonds (SMARTS: `[!$(*#*)&!D1]-!@[!$(*#*)&!D1]`) and produces rigid
substructures connected by flexible bonds. `method="brics"` cuts RDKit BRICS
retrosynthetic bonds instead, which often gives medicinal-chemistry building
blocks.

```python
result = ligand.featurize(
    mode="fragment",
    fragment_kwargs={"method": "rotatable"},  # or "brics"
)
frag = result["fragment"]
```

### Output

| Key | Type | Description |
|-----|------|-------------|
| `fragment_smiles` | `List[str]` | SMILES string for each fragment |
| `atom_to_fragment` | `ndarray (N,)` int64 | Maps each atom index to its fragment index |
| `fragment_atom_indices` | `List[List[int]]` | Atom indices per fragment (reverse of `atom_to_fragment`) |
| `fragment_adjacency` | `ndarray (F, F)` int64 | Symmetric binary adjacency between fragments |
| `fragment_features` | `ndarray (F, 62)` float32 | Per-fragment RDKit descriptors (same 62-dim space as molecule-level `descriptors`) |
| `num_fragments` | `int` | Number of fragments (F) |
| `fragment_method` | `str` | Fragmentation method used: `"rotatable"` or `"brics"` |
| `num_cleaved_bonds` | `int` | Number of bonds cut by the selected method |
| `num_rotatable_bonds` | `int` | Number of rotatable bonds detected (`method="rotatable"`) |
| `num_brics_bonds` | `int` | Number of BRICS bonds detected (`method="brics"`) |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"rotatable"` | Fragmentation method: `"rotatable"` or `"brics"` |
| `min_fragment_size` | `int` | `1` | Fragments smaller than this are merged into their largest neighbour |

`min_fragment_size` is available through `LigandFeaturizer` or the low-level function:

```python
from plmol import LigandFeaturizer
featurizer = LigandFeaturizer(mol)
frag = featurizer.get_fragment(method="brics", min_fragment_size=3)
```

### Low-Level Function

```python
from rdkit import Chem
from plmol.ligand import fragment_by_brics, fragment_on_rotatable_bonds

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
rotatable = fragment_on_rotatable_bonds(mol, min_fragment_size=1)
brics = fragment_by_brics(mol, min_fragment_size=1)
```

Fragment uses `rdkit_utils.prepare_mol` for molecule preparation and computes per-fragment descriptors via `MoleculeFeaturizer.get_descriptors()`.

### Edge Cases

- **No rotatable bonds** (e.g., benzene): Returns 1 fragment containing the whole molecule.
- **Single-atom molecule**: Returns 1 fragment, empty adjacency `(1, 1)`.
- **`min_fragment_size > 1`**: Small fragments are iteratively merged into their largest adjacent fragment. All atoms remain mapped.

---

## Hierarchical Mappings (Atom <-> Fragment <-> Molecule)

The graph dict embeds a 3-level hierarchy enabling atom->fragment pooling and fragment->molecule pooling in downstream ML models:

```
molecule_features (62,)           <- whole molecule descriptors
    ^ aggregate fragment_features
fragment_features (F, 62)         <- per-fragment descriptors (in fragment dict)
fragment_adjacency (F, F)         <- fragment connectivity
    ^ aggregate via fragment_atom_indices
    v lookup via atom_to_fragment
node_features (N, 98)             <- per-atom features
adjacency (N, N, 37)              <- atom connectivity
```

This mirrors the protein side's `atom_to_residue` / `residue_atom_indices` convention.

---

## Surface Mode

dMaSIF-style point cloud surface. Requires a 3D conformer. Use `generate_conformer=True` or call `ligand.generate_conformer()` first.

```python
result = ligand.featurize(mode="surface", generate_conformer=True)
surface = result["surface"]
```

| Key | Shape | Description |
|-----|-------|-------------|
| `points` | `(V, 3)` | Surface point positions |
| `normals` | `(V, 3)` | Outward surface normals |
| `features` | `(V, 30)` | Per-vertex chemical features |
| `feature_names` | `list[str]` | Column names for features |

---

## Utility Module: `rdkit_utils`

Centralized RDKit utility functions used across the ligand pipeline (`plmol/rdkit_utils.py`):

| Function | Description |
|----------|-------------|
| `prepare_mol(mol_or_smiles, add_hs, canonicalize)` | Prepare molecule from SMILES or Mol with optional canonicalization and hydrogens |
| `ensure_3d_conformer(mol, random_seed, optimize)` | Return molecule with 3D conformer (ETKDGv3 + MMFF), generating one if needed |
| `has_3d(mol)` | Check whether molecule has a 3D conformer (uses `Is3D()` on the conformer, not just conformer existence) |
| `canonicalize_mol(mol)` | Reorder atoms to canonical order, preserving coordinates |
| `get_positions(mol)` | Extract 3D coordinates as `(N, 3)` array |

---

## Low-Level Featurizers

### LigandFeaturizer

Wraps all ligand representations including voxel mode.

```python
from plmol import LigandFeaturizer

featurizer = LigandFeaturizer("CCO")

# Individual representations
graph = featurizer.get_graph(standardized=True)
fp = featurizer.get_morgan_fingerprint()
frag = featurizer.get_fragment(method="brics", min_fragment_size=1)
surface = featurizer.get_surface(generate_conformer=True)
voxel = featurizer.get_voxel(generate_conformer=True)

# Batch featurize (mode="all" uses graph + fingerprint + smiles + sequence)
result = featurizer.featurize(
    mode=["graph", "fingerprint", "voxel"],
    voxel_kwargs={"resolution": 1.0, "box_size": 24},
    generate_conformer=True,
)
```

#### Voxel Mode (LigandFeaturizer only)

```python
result = featurizer.featurize(mode="voxel", generate_conformer=True)
voxel = result["voxel"]
```

Channels (16): occupancy, atom type (6), charge, hydrophobicity, HBD, HBA, aromaticity, pos/neg ionizable, hybridization, ring.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `center` | None | Grid center (3,). None = ligand centroid |
| `resolution` | 1.0 | Angstrom per voxel |
| `box_size` | 24 | Grid dimension per axis. None for adaptive |
| `charge_method` | `"gasteiger"` | `"gasteiger"` or `"mmff94"` |

### MoleculeFeaturizer

Descriptors and fingerprints. Delegates fingerprint generation to `FingerprintGenerator` (`ligand/fingerprint_generator.py`).

```python
from plmol import MoleculeFeaturizer

featurizer = MoleculeFeaturizer("CCO")
features = featurizer.get_features()           # descriptors + fingerprints
node, edge, adj = featurizer.get_graph()       # graph representation
descriptors = featurizer.get_descriptors()     # 62-dim descriptor tensor
ecfp4 = featurizer.get_morgan_fingerprint()    # ECFP4 (2048-dim)

# Functional style (pass molecule per call)
featurizer = MoleculeFeaturizer()
features = featurizer.get_features("CCO")
node, edge, adj = featurizer.get_graph("CCO", distance_cutoff=5.0, knn_cutoff=8)
```

### MoleculeGraphFeaturizer

Graph node/edge features (used internally by `MoleculeFeaturizer`). Composed of `AtomFeatureMixin` (`graph_atom_features.py`) and `EdgeFeatureMixin` (`graph_edge_features.py`).

```python
from plmol import MoleculeGraphFeaturizer
```
