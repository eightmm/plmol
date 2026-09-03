# Complex API Reference

## Initialization

```python
from plmol import MolecularComplex

# Alias for backward compatibility
Complex = MolecularComplex

# From files
cx = MolecularComplex.from_files("protein.pdb", "ligand.sdf")

# From objects/mixed inputs
cx = MolecularComplex.from_inputs(
    protein="protein.pdb",       # path or Protein object
    ligand="CCO",                # SMILES, path, RDKit Mol, or Ligand object
    standardize=True,
    add_hs=False,
)

# From mmCIF file (auto-detects protein, nucleic acid, and ligand residues)
cx = MolecularComplex.from_mmcif(
    "structure.cif",
    standardize=True,
    extract_ligands=True,
    ligand_resname=None,
    ligand_chain=None,
)

# Arbitrary molecule combinations
cx = MolecularComplex(molecules={
    "protein": protein_obj,
    "ligand": ligand_obj,
    "nucleic_acid": nucleic_acid_obj,
})

# Swap components
cx.molecules["ligand"] = new_ligand
cx.molecules["protein"] = new_protein
```

### Cache Invalidation

`MolecularComplex` tracks the identity of the underlying ligand mol object (`id(ligand_obj._rdmol)`). When the ligand in `molecules["ligand"]` is replaced or the ligand mol object is mutated, the cache is automatically cleared on the next featurization call.

## Combined Featurization

```python
result = cx.featurize(
    requests="all",  # present components among "ligand", "protein", "nucleic_acid", "interaction"
    ligand_kwargs={"mode": ["graph", "bond_graph", "fragment_graph", "fingerprint"]},
    protein_kwargs={"mode": ["graph", "sequence"]},
    nucleic_acid_kwargs={"mode": ["sequence", "graph"]},
    interaction_kwargs={"distance_cutoff": 6.0, "knn_cutoff": None},
)
# result["ligand"]           -> ligand features
# result["protein"]          -> protein features
# result["nucleic_acid"]     -> nucleic acid features (if present)
# result["interaction"]      -> protein-ligand interaction graph
```

Individual access:

```python
cx.molecules["ligand"].featurize(mode="graph")          # also: bond_graph, fragment_graph
cx.molecules["protein"].featurize(mode="backbone")      # also: atom_graph
cx.molecules["nucleic_acid"].featurize(mode="sequence")
cx.interaction(distance_cutoff=6.0, pocket_cutoff=8.0, knn_cutoff=None)
```

`requests="all"` featurizes only components that are present. Interaction features
are included when both a ligand and structure-backed protein are available.

---

## Interaction Features

```python
interaction = cx.interaction(
    distance_cutoff=6.0,     # Max distance for the contact edges (A)
    pocket_cutoff=None,      # Optional pocket extraction cutoff
    knn_cutoff=None,         # Optional bipartite kNN for contact edges
    include_contacts=False,  # Include raw distance/contact edges
    contact_cutoff=None,     # Optional cutoff for contact edges
    include_coords=True,     # Include protein/ligand heavy atom coordinates
    include_metal_sites=True,# Include metal-site summary arrays
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distance_cutoff` | `float` | `4.5` | Max distance for the **contact** edges. Pharmacophore interactions use their own per-type range — see below |
| `pocket_cutoff` | `Optional[float]` | `None` | If set, extract pocket first, then detect interactions |
| `knn_cutoff` | `Optional[int]` | `None` | Bipartite kNN: each protein atom's k nearest ligand atoms + each ligand atom's k nearest protein atoms. Unioned with distance-based edges |
| `include_contacts` | `bool` | `False` | Add raw protein-ligand contact edges and distances |
| `contact_cutoff` | `Optional[float]` | `None` | Contact-edge cutoff. Defaults to `distance_cutoff` when omitted |
| `include_coords` | `bool` | `True` | Add heavy-atom coordinate arrays for protein and ligand |
| `include_metal_sites` | `bool` | `True` | Add detected protein metal sites and encoded metal summary arrays |

### Output

| Key | Type | Description |
|-----|------|-------------|
| `edges` | `(2, E)` | Protein-ligand heavy atom pairs (pharmacophore interactions) |
| `edge_features` | `(E, 79)` | Interaction feature vectors |
| `interactions` | `List[Interaction]` | Detailed interaction objects |
| `num_interactions` | `int` | Total interaction count |
| `interaction_counts` | `dict` | Per-type interaction counts |
| `num_protein_atoms` | `int` | Number of protein heavy atoms |
| `num_ligand_atoms` | `int` | Number of ligand heavy atoms |
| `ligand_atom_order` | `(N,)` | Ligand graph node → this block's ligand index. See below |
| `distance_cutoff` | `float` | Contact-edge cutoff used. Not the pharmacophore ranges |
| `knn_cutoff` | `Optional[int]` | kNN cutoff used |
| `feature_dim` | `int` | Edge feature dimension (79) |
| `metadata` | `dict` | Interaction type indices, pharmacophore indices, element types, residue types |
| `protein_coords` | `(P, 3)` | Protein heavy atom coordinates, when `include_coords=True` |
| `ligand_coords` | `(L, 3)` | Ligand heavy atom coordinates, when `include_coords=True` |
| `metal_sites` | `List[MetalSite]` | Detected protein metal coordination sites, when `include_metal_sites=True` |
| `metal_features` | `dict` | Encoded metal-site arrays from `encode_metal_features()` |

### Joining the interaction block to the ligand graph

The two blocks count the ligand's atoms differently. `graph` mode
canonicalizes the atom order — that is what lets the bond and fragment graphs
line up with it — while the interaction featurizer indexes the molecule as it
was handed over, which for a file is the file's own order. Both are
self-consistent, and an interaction edge's ligand endpoint is **not** a ligand
graph node index.

`ligand_atom_order` is the gather that lines them up:

```python
result = complex.featurize(requests=["ligand", "interaction"])
order = result["interaction"]["ligand_atom_order"]

# the same atoms, in the graph's numbering
result["interaction"]["ligand_coords"][order]      # == result["ligand"]["graph"]["coords"]

# an interaction edge's ligand endpoint, as a graph node
import numpy as np
to_node = np.full(len(order), -1)
to_node[order] = np.arange(len(order))
nodes = to_node[result["interaction"]["edges"][1]]
```

An entry is `-1` where the graph node has no counterpart, which happens only
when the molecule carries explicit hydrogens: the graph keeps them and the
interaction block is heavy atoms alone.

### `distance_cutoff` does not widen interaction detection

Each interaction type is detected at its own physically motivated range, from
`INTERACTION_TYPES`:

| Type | Range (Å) |
|------|-----------|
| `hydrogen_bond` | 3.5 |
| `salt_bridge` | 4.0 |
| `pi_stacking` | 5.5 |
| `cation_pi` | 6.0 |
| `hydrophobic` | 4.5 |
| `halogen_bond` | 3.5 |
| `metal_coordination` | 2.8 |

`distance_cutoff` bounds the optional **contact** edges instead. On the
example complex, raising it from 3.5 to 8.0 takes the contact edges from 29
to 1849 and leaves the pharmacophore interactions at 53 throughout. If you
want a wider hydrogen bond you have to change its entry in
`INTERACTION_TYPES`, not this parameter.

### Interaction Types

| Type | Detection | Typical Distance |
|------|----------|-----------------|
| `hydrogen_bond` | Donor-acceptor pairs + D-H-A angle | < 3.5 A |
| `salt_bridge` | Positive-negative charge pairs | < 4.0 A |
| `pi_stacking` | Aromatic ring pairs + ring angle | < 5.5 A |
| `cation_pi` | Charged atom + aromatic ring + cation-to-ring-normal angle filter (< 30 deg) | < 6.0 A |
| `hydrophobic` | Hydrophobic atom pairs | < 4.5 A |
| `halogen_bond` | Halogen + acceptor + C-X-A angle | < 3.5 A |
| `metal_coordination` | Metal ion (from protein pocket HETATM) + coordinating ligand atom | < 2.8 A |

Metal coordination interactions are detected by `PLInteractionFeaturizer`.
`MolecularComplex.interaction()` also returns a separate metal-site summary using
`detect_metal_sites()` and `encode_metal_features()` when metal atoms are present
in the protein/pocket molecule.

### Edge Features `(E, 79)`

| Index | Group | Dim | Features |
|-------|-------|-----|----------|
| `[0:7]` | Interaction type | 7 | One-hot: hydrogen_bond, salt_bridge, pi_stacking, cation_pi, hydrophobic, halogen_bond, metal_coordination |
| `[7:11]` | Geometry | 4 | Distance (normalized), angle, has_valid_angle, angle_type |
| `[11:31]` | Element types | 20 | Protein element one-hot (10) + ligand element one-hot (10) |
| `[31:43]` | Hybridization | 12 | Protein hybridization (6) + ligand hybridization (6) |
| `[43:45]` | Formal charges | 2 | Protein charge, ligand charge (normalized) |
| `[45:47]` | Aromatic | 2 | Protein is_aromatic, ligand is_aromatic |
| `[47:51]` | Ring/degree | 4 | is_in_ring (2) + degree (2) |
| `[51:72]` | Residue type | 21 | Protein residue one-hot (20 standard + Other) |
| `[72]` | Backbone | 1 | Protein atom is_backbone |
| `[73]` | Strength | 1 | Gaussian decay from ideal distance: exp(-0.5 * ((d - ideal) / 0.5)^2) |
| `[74:76]` | Cross-contact density | 2 | Number of atoms from the other entity within 4.0 A of each endpoint (protein, ligand), normalized by /10 |
| `[76:78]` | Endpoint min distance | 2 | Min distance from each endpoint to its nearest partner atom (protein, ligand), normalized by /cutoff |
| `[78]` | Relative pocket distance | 1 | Interaction distance / max pairwise distance in the pocket |

### Contact Edges (optional)

```python
# Direct PLInteractionFeaturizer usage for contact edges
from plmol import PLInteractionFeaturizer

featurizer = PLInteractionFeaturizer(protein_mol, ligand_mol, distance_cutoff=4.5, knn_cutoff=8)
graph = featurizer.get_interaction_graph(include_contacts=True, contact_cutoff=4.5, knn_cutoff=8)
```

| Key | Type | Description |
|-----|------|-------------|
| `contact_edges` | `(2, E_c)` | All protein-ligand heavy atom pairs within cutoff (union with kNN if set) |
| `contact_distances` | `(E_c,)` | Pairwise distances |
| `num_contacts` | `int` | Number of contact edges |

---

## PLInteractionFeaturizer (Low-Level)

Direct access to the interaction featurizer for fine-grained control.

```python
from plmol import PLInteractionFeaturizer
from plmol.interaction import detect_metal_sites, encode_metal_features

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

# Edge arrays
edges, edge_features = featurizer.get_interaction_edges()

# Distance-based edges (all pairs within cutoff, with optional kNN)
dist_edges, dist_features = featurizer.get_distance_based_edges(distance_cutoff=4.5, knn_cutoff=8)

# Full graph with metadata
graph = featurizer.get_interaction_graph(include_contacts=True, knn_cutoff=8)

# Atom features
protein_pharm, ligand_pharm = featurizer.get_atom_pharmacophore_features()
protein_chem, ligand_chem = featurizer.get_atom_chemical_features()
protein_coords, ligand_coords = featurizer.get_heavy_atom_coords()

# Metal-site summaries (requires metal atoms in protein_mol metadata)
metal_sites = detect_metal_sites(atom_coords, atom_metadata, metal_indices)
metal_features = encode_metal_features(metal_sites, n_residues=num_residues)

# Summary
print(featurizer.get_interaction_summary())
```

---

## Pocket Extraction

Pocket extraction preserves metal HETATM records (Zn, Fe, Mg, Ca, Mn, Cu, Co, Ni) that are within the distance cutoff of the ligand. This enables accurate metal coordination detection in interaction featurization.

```python
from plmol.interaction import extract_pocket

pocket_list = extract_pocket(
    pdb_path="protein.pdb",
    ligand=ligand_mol,       # RDKit Mol
    distance_cutoff=6.0,     # A
)

for pocket_info in pocket_list:
    pocket_mol = pocket_info.pocket_mol    # RDKit Mol of pocket residues + nearby metals
    residues = pocket_info.pocket_residues # List of (chain, resnum, resname) tuples
    metals = pocket_info.metal_records     # List of metal HETATM records within cutoff
    num_atoms = pocket_info.num_atoms
    num_residues = pocket_info.num_residues
```
