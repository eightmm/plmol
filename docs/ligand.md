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
    mode="all",                    # str or list of modes
    graph_kwargs={},               # graph options
    surface_kwargs={},             # surface options
    fingerprint_kwargs={},         # fingerprint options
    voxel_kwargs={},               # voxel options
    generate_conformer=False,      # auto-generate 3D if missing
    add_hs=None,                   # hydrogen override
)
```

| Mode | Output Key | Description |
|------|-----------|-------------|
| `"graph"` | `"graph"` | Dense adjacency graph (node_features, adjacency, bond_mask, ...) |
| `"fingerprint"` | `"fingerprint"` | Descriptors + ECFP4/6, MACCS, RDKit FP, AtomPair, ErG |
| `"surface"` | `"surface"` | Molecular surface mesh |
| `"voxel"` | `"voxel"` | 3D voxel grid |
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

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | `(N, 98)` | `float32` | Per-atom feature vector |
| `adjacency` | `(N, N, 37)` | `float32` | Dense adjacency (bond + pair channels) |
| `bond_mask` | `(N, N)` | `bool` | True where chemical bond exists |
| `distance_matrix` | `(N, N)` | `float32` | Euclidean distance (0 if no 3D) |
| `distance_bounds` | `(N, N, 2)` | `float32` | DG lower/upper distance bounds |
| `coords` | `(N, 3)` | `float32` | 3D coordinates (0 if no conformer) |

Sparse conversion:

```python
from plmol import LigandFeaturizer
edge_index, edge_features = LigandFeaturizer.adjacency_to_bond_edges(graph["adjacency"])
# edge_index: (2, E)  edge_features: (E, 37)
```

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

## Fingerprint Mode

```python
result = ligand.featurize(mode="fingerprint")
fp = result["fingerprint"]
```

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

## Surface Mode

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

## Voxel Mode

```python
result = ligand.featurize(mode="voxel", generate_conformer=True)
voxel = result["voxel"]
```

Channels (16): occupancy, atom type (6), charge, hydrophobicity, HBD, HBA, aromaticity, pos/neg ionizable, hybridization, ring.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `center` | None | Grid center (3,). None = ligand centroid |
| `resolution` | 1.0 | Angstrom per voxel |
| `box_size` | 24 | Grid dimension per axis. None for adaptive |
| `charge_method` | `"gasteiger"` | `"gasteiger"` or `"mmff94"` |

---

## Low-Level Featurizers

```python
from plmol import (
    MoleculeFeaturizer,        # Molecular features + fingerprints
    MoleculeGraphFeaturizer,   # Graph node/edge features
    LigandFeaturizer,          # Ligand-specific wrapper (graph + surface + FP)
)
```

### MoleculeFeaturizer (Object-Oriented)

```python
from plmol import MoleculeFeaturizer

featurizer = MoleculeFeaturizer("CCO")
features = featurizer.get_feature()           # descriptors + fingerprints
node, edge, adj = featurizer.get_graph()      # graph representation
descriptors = featurizer.get_descriptors()    # 62-dim descriptor tensor
ecfp4 = featurizer.get_morgan_fingerprint()   # ECFP4 (2048-dim)
```

### MoleculeFeaturizer (Functional)

```python
featurizer = MoleculeFeaturizer()
features = featurizer.get_feature("CCO")
node, edge, adj = featurizer.get_graph("CCO", distance_cutoff=5.0, knn_cutoff=8)
```
