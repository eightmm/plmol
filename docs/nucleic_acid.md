# Nucleic Acid API Reference

## Initialization

```python
from plmol import NucleicAcid

# From PDB file (recommended)
na = NucleicAcid.from_pdb("dna.pdb", chain_id="A")

# From mmCIF/PDBx file
na = NucleicAcid.from_mmcif("structure.cif", chain_id="A")

# Auto-detect format from file extension
na = NucleicAcid.from_structure("structure.cif", chain_id="A")

# From sequence string (no structure needed)
na = NucleicAcid.from_sequence("ATGCATGC", na_type="DNA")
na = NucleicAcid.from_sequence("AUGCAUGC", na_type="RNA")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain_id` | `str` | `None` | Specific chain to extract; None uses all nucleic acid atoms |
| `na_type` | `str` | `None` (auto-detect) | `"DNA"` or `"RNA"` for sequence initialization |

## Featurization Modes

```python
result = na.featurize(
    mode="all",                              # str or list of modes
    graph_kwargs={"distance_cutoff": 8.0},   # graph options
    atom_graph_kwargs={"distance_cutoff": 4.5},  # atom graph options
)
```

| Mode | Output Key | Description |
|------|-----------|-------------|
| `"sequence"` | `"sequence"` | Sequence-level features (tokens, GC content, purine/pyrimidine classification) |
| `"graph"` | `"graph"` | Residue-level graph with backbone torsion angles and base properties |
| `"backbone"` | `"backbone"` | Sugar-phosphate backbone coordinates (7 atoms per residue) |
| `"atom_graph"` | `"atom_graph"` | Atom-level graph with token-based features |
| `"all"` | all above | All modes combined |

Lazy properties:

```python
na.sequence       # str (1-letter nucleotide codes)
na.chain_type     # str ("DNA", "RNA", "MIXED", or "UNKNOWN")
na.graph          # residue-level graph (auto-computed)
```

---

## Sequence Mode

```python
result = na.featurize(mode="sequence")
seq = result["sequence"]
```

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `tokens` | `(N,)` | `int64` | Nucleotide token ID (0-8, for `nn.Embedding`) |
| `is_purine` | `(N,)` | `float32` | 1.0 if purine (A, G), 0.0 if pyrimidine (C, T/U) |
| `is_pyrimidine` | `(N,)` | `float32` | 1.0 if pyrimidine, 0.0 if purine |
| `gc_content` | scalar | `float` | GC content fraction (0.0 to 1.0) |
| `res_names` | `list[str]` | -- | 3-letter residue names (DA, DT, DG, DC for DNA; A, U, G, C for RNA) |

---

## Graph Mode -- Residue Level

Residue-level graph with backbone torsion angles (alpha, beta, gamma, delta, epsilon, zeta, chi) and base properties.

```python
result = na.featurize(
    mode="graph",
    graph_kwargs={"distance_cutoff": 8.0},
)
graph = result["graph"]
```

### graph_kwargs

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distance_cutoff` | `float` | `8.0` | Distance cutoff for spatial edges (A); sequential edges always included |

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `nucleotide_type` | `(N,)` | `int64` | Nucleotide token (NUCLEOTIDE_TOKEN index) |
| `one_hot` | `(N, NUM_NUCLEOTIDE_TYPES)` | `float32` | One-hot encoding of nucleotide type |
| `is_purine` | `(N,)` | `float32` | Purine classification |
| `is_pyrimidine` | `(N,)` | `float32` | Pyrimidine classification |
| `is_dna` | `(N,)` | `float32` | 1.0 if DNA, 0.0 if RNA |
| `torsions` | `(N, 7)` | `float32` | Backbone torsion angles: [alpha, beta, gamma, delta, epsilon, zeta, chi] (radians) |
| `sugar_pucker` | `(N,)` | `float32` | Sugar pucker mode: 1.0 = C3'-endo, 0.0 = C2'-endo, 0.5 = unknown |
| `mol_weight` | `(N,)` | `float32` | Molecular weight normalized to ~[0, 1] |
| `n_hbond_donors` | `(N,)` | `float32` | Number of hydrogen bond donors |
| `n_hbond_acceptors` | `(N,)` | `float32` | Number of hydrogen bond acceptors |
| `coords` | `(N, 3)` | `float32` | C1' atom coordinates (or centroid if missing) |
| `edge_index` | `(2, E)` | `int64` | Edge pairs (source, target) |
| `edge_attr` | `(E, 3)` | `float32` | Edge features: [is_sequential, distance, inverse_distance] |
| `num_nodes` | scalar | `int` | Number of residues (N) |

Edge construction: sequential edges (i, i+1) always included. Spatial edges added for all pairs within `distance_cutoff`.

### Nucleotide Types

| Token | DNA | RNA | Description |
|-------|-----|-----|-------------|
| 0 | DA | A | Adenine |
| 1 | DT | U | Thymine (DNA) / Uracil (RNA) |
| 2 | DG | G | Guanine |
| 3 | DC | C | Cytosine |
| 4-7 | -- | -- | (reserved for modified nucleotides) |
| 8 | UNK_NT | UNK_NT | Unknown/unrecognized nucleotide |

---

## Backbone Mode

Sugar-phosphate backbone geometry with atom coordinates per residue.

```python
result = na.featurize(mode="backbone")
backbone = result["backbone"]
```

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `backbone_coords` | `(N, 7, 3)` | `float32` | Backbone atom coordinates: [P, O5', C5', C4', C3', O3', C1'] (NaN if missing) |
| `backbone_atom_names` | `list[str]` | -- | Atom names in order: ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'"] |
| `num_residues` | scalar | `int` | Number of residues (N) |

The 7 atoms represent:
- **P**: Phosphorus (bridging phosphodiester bond)
- **O5', O3'**: 5' and 3' oxygen atoms (phosphodiester linkage)
- **C5', C4', C3'**: Ribose/deoxyribose carbon atoms (5-membered sugar ring)
- **C1'**: Anomeric carbon (glycosidic bond attachment point)

---

## Backbone Torsion Angles

The 7 backbone torsion angles (in radians) are:

| Index | Name | Atoms | Description |
|-------|------|-------|-------------|
| 0 | alpha (α) | O3'(i-1) - P(i) - O5'(i) - C5'(i) | Phosphodiester linkage torsion |
| 1 | beta (β) | P(i) - O5'(i) - C5'(i) - C4'(i) | 5' oxygen torsion |
| 2 | gamma (γ) | O5'(i) - C5'(i) - C4'(i) - C3'(i) | Exocyclic C5' torsion |
| 3 | delta (δ) | C5'(i) - C4'(i) - C3'(i) - O3'(i) | Ring closure torsion (determines sugar pucker) |
| 4 | epsilon (ε) | C4'(i) - C3'(i) - O3'(i) - P(i+1) | 3' to next phosphorus torsion |
| 5 | zeta (ζ) | C3'(i) - O3'(i) - P(i+1) - O5'(i+1) | Phosphodiester linkage torsion (next) |
| 6 | chi (χ) | O4'(i) - C1'(i) - N9/N1(i) - C4/C2(i) | Glycosidic bond (base rotation) |

**Sugar pucker determination**: Inferred from delta torsion:
- **C3'-endo** (1.0): delta ≈ 85° (commonly found in A-form DNA/RNA)
- **C2'-endo** (0.0): delta ≈ 145° (commonly found in B-form DNA)

---

## Atom Graph Mode

Atom-level graph with token-based features.

```python
result = na.featurize(
    mode="atom_graph",
    atom_graph_kwargs={"distance_cutoff": 4.5},
)
atom_graph = result["atom_graph"]
```

### atom_graph_kwargs

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distance_cutoff` | `float` | `4.5` | Distance cutoff for edges (A) |

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `coords` | `(A, 3)` | `float32` | Atom 3D coordinates |
| `residue_token` | `(A,)` | `int64` | Nucleotide type for parent residue (NUCLEOTIDE_TOKEN) |
| `atom_to_residue` | `(A,)` | `int64` | Maps each atom to 0-indexed residue index |
| `residue_atom_indices` | `List[List[int]]` | -- | Atom indices per residue (reverse mapping) |
| `edge_index` | `(2, E)` | `int64` | Sparse edge pairs (source, target) |
| `edge_distances` | `(E,)` | `float32` | Euclidean distance (A) for each edge |
| `num_atoms` | scalar | `int` | Total atom count (A) |

Edge construction: all atom pairs within `distance_cutoff`.

### Standard Nucleotide Atoms

Heavy atoms (hydrogens excluded) in standard order per nucleotide:

**DNA/RNA Backbone**: P, O1P, O2P, O5', C5', C4', O4', C3', O3', C1'

**DNA Bases**:
- **Adenine (A/DA)**: N9, C4, N3, C2, N1, C6, N6, C5, N7, C8
- **Guanine (G/DG)**: N9, C4, N3, C2, N2, N1, C6, O6, C5, N7, C8
- **Cytosine (C/DC)**: N1, C2, O2, N3, C4, N4, C5, C6
- **Thymine (T) / Uracil (U)**: N1, C2, O2, N3, C4, O4, C5, C6

---

## Code Examples

### Basic Featurization

```python
from plmol import NucleicAcid
import numpy as np

# Load from PDB
na = NucleicAcid.from_pdb("dna.pdb")

# Get all features
result = na.featurize(mode="all")

# Access individual features
seq_tokens = result["sequence"]["tokens"]
graph = result["graph"]
backbone = result["backbone"]
atom_graph = result["atom_graph"]

# Get properties
print(f"Type: {na.chain_type}")  # "DNA", "RNA", or "MIXED"
print(f"Sequence: {na.sequence}")
print(f"GC Content: {result['sequence']['gc_content']:.2%}")
```

### Graph-Based Analysis

```python
# Residue-level graph
graph = na.featurize(mode="graph", graph_kwargs={"distance_cutoff": 8.0})["graph"]

# Extract node features
nucleotide_types = graph["nucleotide_type"]  # (N,)
is_purine = graph["is_purine"]               # (N,)
torsions = graph["torsions"]                 # (N, 7)
coords = graph["coords"]                     # (N, 3)

# Extract edge features
edge_index = graph["edge_index"]             # (2, E)
edge_attr = graph["edge_attr"]               # (E, 3)

# Find sequential edges (i, i+1)
is_sequential = edge_attr[:, 0] > 0.5
sequential_edges = edge_index[:, is_sequential]

# Distance-based edges
distances = edge_attr[:, 1]
spatial_edges = edge_index[distances <= 8.0]
```

### Sequence Initialization (No Structure)

```python
# Create NA from sequence without 3D structure
na = NucleicAcid.from_sequence("GCTAGCTAG", na_type="DNA")

# Sequence features available without PDB
seq_feat = na.featurize(mode="sequence")["sequence"]
print(f"Length: {len(seq_feat['tokens'])}")
print(f"GC: {seq_feat['gc_content']:.2%}")

# Graph/backbone modes require PDB
try:
    na.featurize(mode="graph")
except InputError:
    print("Graph features require PDB file")
```

### Backbone Geometry Analysis

```python
# Extract backbone coordinates
backbone = na.featurize(mode="backbone")["backbone"]

backbone_coords = backbone["backbone_coords"]  # (N, 7, 3)
atom_names = backbone["backbone_atom_names"]  # ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'"]

# Compute inter-residue distances (P to P)
p_coords = backbone_coords[:, 0, :]  # Phosphorus coordinates
p_distances = np.linalg.norm(p_coords[1:] - p_coords[:-1], axis=1)

# Find sugar pucker from torsion angles
graph = na.featurize(mode="graph")["graph"]
sugar_pucker = graph["sugar_pucker"]
c3_endo_count = (sugar_pucker == 1.0).sum()
c2_endo_count = (sugar_pucker == 0.0).sum()
print(f"C3'-endo: {c3_endo_count}, C2'-endo: {c2_endo_count}")
```

### Atom-Level Features

```python
# Atom graph for detailed structural analysis
atom_graph = na.featurize(mode="atom_graph", atom_graph_kwargs={"distance_cutoff": 4.5})["atom_graph"]

atom_coords = atom_graph["coords"]            # (A, 3)
residue_tokens = atom_graph["residue_token"]  # (A,)
atom_to_residue = atom_graph["atom_to_residue"]  # (A,)
edge_index = atom_graph["edge_index"]         # (2, E)
edge_distances = atom_graph["edge_distances"] # (E,)

# Group atoms by residue
residue_atom_indices = atom_graph["residue_atom_indices"]
for res_idx, atom_indices in enumerate(residue_atom_indices):
    res_atoms = atom_coords[atom_indices]
    print(f"Residue {res_idx}: {len(atom_indices)} atoms")
```

### mmCIF File Handling

```python
# Load from mmCIF (requires gemmi)
na = NucleicAcid.from_mmcif("structure.cif", chain_id="A")

# or auto-detect format
na = NucleicAcid.from_structure("structure.cif")

# Features extracted identically
result = na.featurize(mode="graph")
```

---

## Low-Level Featurizers

```python
from plmol.nucleic_acid import NucleicFeaturizer

featurizer = NucleicFeaturizer("dna.pdb", chain_id="A")

# Direct access to feature functions
seq_features = featurizer.get_sequence_features()
graph = featurizer.get_graph(distance_cutoff=8.0)
backbone = featurizer.get_backbone()
atom_graph = featurizer.get_atom_graph(distance_cutoff=4.5)
```

---

## Constants and Enumerations

```python
from plmol.constants import (
    DNA_RESIDUES,              # {"DA", "DT", "DG", "DC"}
    RNA_RESIDUES,              # {"A", "U", "G", "C"}
    NUCLEOTIDE_TOKEN,          # {"DA": 0, "DT": 1, "DG": 2, "DC": 3, ...}
    NUCLEOTIDE_3TO1,           # {"DA": "A", "DT": "T", ...}
    NUCLEOTIDE_PROPERTIES,     # {res_name: (mol_weight, n_donors, n_acceptors, ...)}
    NUM_NUCLEOTIDE_TYPES,      # 9 (includes UNK_NT)
    STANDARD_NUCLEOTIDE_ATOMS, # {res_name: [atom_names]}
    NUCLEOTIDE_MAX_SASA,       # {res_name: float} (reference SASA values)
    PURINES,                   # {"A", "DA", "G", "DG"}
    PYRIMIDINES,               # {"C", "DC", "T", "DT", "U"}
)
```

---

## Integration with Complex

Use nucleic acids within a multi-molecule complex:

```python
from plmol import Complex, Ligand, NucleicAcid, Protein

protein = Protein.from_pdb("protein.pdb")
ligand = Ligand.from_sdf("ligand.sdf")
na = NucleicAcid.from_pdb("dna.pdb")

# Add nucleic acid to complex with protein and ligand.
complex = Complex.from_inputs(protein=protein, ligand=ligand, nucleic_acid=na)

# Featurize with interaction features
result = complex.featurize(
    requests=["protein", "ligand", "nucleic_acid", "interaction"]
)
```


## Watson-Crick Pairing

A graph built from distance alone cannot tell a base pair from any other close
contact. `graph` mode reports the pairing alongside its edges, so `edge_attr`
keeps its width of 3.

| Key | Shape | Description |
|-----|-------|-------------|
| `pair_index` | `(2, P)` `int64` | Purine row, then its pyrimidine |
| `pair_kind` | `(P,)` `int64` | 0 = A·T, 1 = A·U, 2 = G·C |
| `pair_c1_distance` | `(P,)` `float32` | C1'-C1' separation; about 10.5 A for a real pair |
| `pair_plane_angle` | `(P,)` `float32` | Degrees between the two base planes |
| `is_paired` | `(N,)` `float32` | Per nucleotide |

A pair is recognised by its hydrogen bonds, not by sequence: the purine's N1 to
the pyrimidine's N3 is the anchor, at least one of the pair's other canonical
bonds has to be within 3.5 A as well, and the bases must be within 60 degrees
of coplanar. Stacked bases are coplanar too, which is why the bond length is
checked first. Each base takes one partner, the one with the shortest anchor.

```python
from plmol import find_base_pairs

pairs = find_base_pairs(residues)      # residue dicts, as the featurizer builds them
pairs[0].kind, pairs[0].hbond_distances, pairs[0].c1_distance
```
