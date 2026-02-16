# Complex API Reference

## Initialization

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

## Combined Featurization

```python
result = cx.featurize(
    requests="all",  # "ligand", "protein", "interaction", or "all"
    ligand_kwargs={"mode": ["graph", "fingerprint"]},
    protein_kwargs={"mode": ["graph", "sequence"]},
    interaction_kwargs={"distance_cutoff": 6.0, "knn_cutoff": None},
)
# result["ligand"]      -> ligand features
# result["protein"]     -> protein features
# result["interaction"] -> interaction graph
```

Individual access:

```python
cx.ligand(mode="graph")
cx.protein(mode="backbone")
cx.interaction(distance_cutoff=6.0, pocket_cutoff=8.0, knn_cutoff=None)
```

---

## Interaction Features

```python
interaction = cx.interaction(
    distance_cutoff=6.0,     # Max distance for interaction detection (A)
    pocket_cutoff=None,      # Optional pocket extraction cutoff
    knn_cutoff=None,         # Optional bipartite kNN for contact edges
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distance_cutoff` | `float` | `4.5` | Max distance for interaction detection |
| `pocket_cutoff` | `Optional[float]` | `None` | If set, extract pocket first, then detect interactions |
| `knn_cutoff` | `Optional[int]` | `None` | Bipartite kNN: each protein atom's k nearest ligand atoms + each ligand atom's k nearest protein atoms. Unioned with distance-based edges |

### Output

| Key | Type | Description |
|-----|------|-------------|
| `edges` | `Tensor (2, E)` | Protein-ligand heavy atom pairs (pharmacophore interactions) |
| `edge_features` | `Tensor (E, 74)` | Interaction feature vectors |
| `interactions` | `List[Interaction]` | Detailed interaction objects |
| `num_interactions` | `int` | Total interaction count |
| `interaction_counts` | `dict` | Per-type interaction counts |
| `num_protein_atoms` | `int` | Number of protein heavy atoms |
| `num_ligand_atoms` | `int` | Number of ligand heavy atoms |
| `distance_cutoff` | `float` | Distance cutoff used |
| `knn_cutoff` | `Optional[int]` | kNN cutoff used |
| `feature_dim` | `int` | Edge feature dimension (74) |
| `metadata` | `dict` | Interaction type indices, pharmacophore indices, element types, residue types |

### Interaction Types

| Type | Detection | Typical Distance |
|------|----------|-----------------|
| `hydrogen_bond` | Donor-acceptor pairs + D-H-A angle | < 3.5 A |
| `salt_bridge` | Positive-negative charge pairs | < 4.0 A |
| `pi_stacking` | Aromatic ring pairs + ring angle | < 5.5 A |
| `cation_pi` | Charged atom + aromatic ring | < 6.0 A |
| `hydrophobic` | Hydrophobic atom pairs | < 4.5 A |
| `halogen_bond` | Halogen + acceptor + C-X-A angle | < 3.5 A |
| `metal_coordination` | Metal ion + coordinating atom | < 2.8 A |

### Edge Features `(E, 74)`

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

### Contact Edges (optional)

```python
# Direct PLInteractionFeaturizer usage for contact edges
from plmol import PLInteractionFeaturizer

featurizer = PLInteractionFeaturizer(protein_mol, ligand_mol, distance_cutoff=4.5, knn_cutoff=8)
graph = featurizer.get_interaction_graph(include_contacts=True, contact_cutoff=4.5, knn_cutoff=8)
```

| Key | Type | Description |
|-----|------|-------------|
| `contact_edges` | `Tensor (2, E_c)` | All protein-ligand heavy atom pairs within cutoff (union with kNN if set) |
| `contact_distances` | `Tensor (E_c,)` | Pairwise distances |
| `num_contacts` | `int` | Number of contact edges |

---

## PLInteractionFeaturizer (Low-Level)

Direct access to the interaction featurizer for fine-grained control.

```python
from plmol import PLInteractionFeaturizer

featurizer = PLInteractionFeaturizer(
    protein_mol=protein_mol,
    ligand_mol=ligand_mol,
    distance_cutoff=4.5,
    knn_cutoff=None,
)

# Detect specific interaction types
hbonds = featurizer.detect_hydrogen_bonds()
salt_bridges = featurizer.detect_salt_bridges()
pi_stacking = featurizer.detect_pi_stacking()
hydrophobic = featurizer.detect_hydrophobic()

# All interactions
all_interactions = featurizer.detect_all_interactions()

# Edge tensors
edges, edge_features = featurizer.get_interaction_edges()

# Distance-based edges (all pairs within cutoff, with optional kNN)
dist_edges, dist_features = featurizer.get_distance_based_edges(distance_cutoff=4.5, knn_cutoff=8)

# Full graph with metadata
graph = featurizer.get_interaction_graph(include_contacts=True, knn_cutoff=8)

# Atom features
protein_pharm, ligand_pharm = featurizer.get_atom_pharmacophore_features()
protein_chem, ligand_chem = featurizer.get_atom_chemical_features()
protein_coords, ligand_coords = featurizer.get_heavy_atom_coords()

# Summary
print(featurizer.get_interaction_summary())
```

---

## Pocket Extraction

```python
from plmol.interaction import extract_pocket

pocket_list = extract_pocket(
    pdb_path="protein.pdb",
    ligand=ligand_mol,       # RDKit Mol
    distance_cutoff=6.0,     # A
)

for pocket_info in pocket_list:
    pocket_mol = pocket_info.pocket_mol    # RDKit Mol of pocket residues
    residues = pocket_info.residue_ids     # List of (chain, resnum) tuples
```
