# Graph View API Reference

One shape for every plmol graph, plus batching and dimension lookup.

```python
from plmol import as_graph, collate, feature_dims
```

## Why

plmol's graph outputs grew per molecule type and disagree on almost everything.
Across the seven graph views only `coords` is shared, and even that sits
beside wildly different node and edge conventions:

| View | Edges | Edge features | Node features |
|------|-------|---------------|---------------|
| protein `graph` (residue) | `edge_index` | tuple of 2 arrays + vector tuple | tuple of 8 arrays |
| protein `atom_graph` | `edge_index` | none — 4 loose per-edge arrays | `(N,)` token ids |
| ligand `graph` | none — dense `(N, N, 37)` `adjacency` | in the adjacency | `(N, 94)` |
| ligand `bond_graph` | `edge_index` | `(E, 96)` | `(B, 29)` |
| ligand `fragment_graph` | `edge_index` | `(E, 31)` | `(F, 62)` |
| nucleic `graph` | `edge_index` | `(E, 3)` `edge_attr` | 9 loose per-nucleotide arrays |
| nucleic `atom_graph` | `edge_index` | `(E,)` `edge_distances` | token ids only |

Writing one GNN that consumes any of them means branching seven ways.
`as_graph` removes that branch.

Nothing here changes what `featurize` returns; this is a view on top of it.

## `as_graph(view, source=None)`

Normalizes any of the views above.

```python
graph = protein.featurize(mode="graph")["graph"]
g = as_graph(graph)
```

### Output

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `node_features` | `(N, F)` | `float32` | Continuous node features. `(N, 0)` when a view's nodes are purely token-valued |
| `node_tokens` | `(N, T)` or `None` | `int64` | Embedding inputs, e.g. the protein atom graph's atom/residue/element tokens |
| `node_vector_features` | `(N, V, 3)` or `None` | `float32` | Kept unflattened for SE(3)-equivariant models |
| `edge_index` | `(2, E)` | `int64` | Source and destination node indices |
| `edge_features` | `(E, C)` | `float32` | Per-edge features |
| `edge_vector_features` | `(E, W, 3)` or `None` | `float32` | As above, per edge |
| `coords` | `(N, 3)` | `float32` | Node positions |
| `num_nodes`, `num_edges` | `int` | — | Counts |
| `source` | `str` | — | Which view this came from |

Everything comes back as numpy arrays, whatever shape the view arrived in.
`to_torch(graph)` converts the whole thing for a model that wants tensors.

### Conversions applied

- **Dense ligand `adjacency`** is unrolled with the same bond mask the library
  uses elsewhere (first four channels are the bond-type one-hot), then reduced
  to the 29 channels that describe a bonded pair. Eight of the ten pair
  channels are degenerate on a bond: the six shortest-path bins are always
  `d=1`, `same_fragment` is always 1, and the euclidean distance is the bond
  length channel 20 already carries (measured correlation 1.000000). See
  [Bond View Channels](ligand.md#bond-view-channels).
  `LigandFeaturizer.adjacency_to_bond_edges` stays faithful at 37 channels; it
  is a literal conversion, while `as_graph` is a curated model-ready view.
- **Protein residue tuples** are concatenated along the feature axis. Vector
  features stay in their own `(N, V, 3)` array rather than being flattened.
- **Nucleic acid `graph`** loose per-nucleotide arrays are concatenated in a
  fixed order: `one_hot`, `is_purine`, `is_pyrimidine`, `is_dna`, `torsions`,
  `sugar_pucker`, `mol_weight`, `n_hbond_donors`, `n_hbond_acceptors` →
  `(N, 23)`; `nucleotide_type` → `node_tokens (N, 1)`; `edge_attr` → edges.
- **Nucleic acid `atom_graph`** has no continuous node features, so
  `node_features` is `(N, 0)` and `residue_token` becomes `node_tokens`.
- **Protein atom graph** loose arrays are concatenated in a fixed order.
  Per-edge: `edge_distances`, `same_residue`, `sequence_separation`,
  `unit_vector` → `(E, 6)`. Per-node: `burial_index`, `formal_charge`,
  `is_backbone`, `is_hbond_acceptor`, `is_hbond_donor`, `is_polar_sasa`,
  `sasa`, `secondary_structure` → `(N, 10)`. Tokens:
  `atom_tokens`, `residue_token`, `atom_element` → `(N, 3)`.
  `relative_sasa` is deliberately excluded: `burial_index` is exactly
  `1 - relative_sasa` (measured correlation -1.000000, maximum difference 0).
  The raw `atom_graph` dict still exposes it under that name.

`as_graph` is idempotent: normalizing an already normalized graph returns it
unchanged. `backbone`, `surface` and `voxel` are not graphs — they have no
edges — and are rejected with `InputError`.

## `collate(views)`

Batches graphs into one disconnected graph, the layout PyTorch Geometric's
`Batch` uses, so the result drops into models written against it.

```python
batch = collate([as_graph(g) for g in graphs])
# or pass raw featurize output; it is normalized on the way in
batch = collate([lig.featurize(mode="graph")["graph"] for lig in ligands])
```

Node indices of each graph are offset by the running node count, so
`edge_index` stays valid across the batch.

### Additional keys

| Key | Shape | Description |
|-----|-------|-------------|
| `batch` | `(N_total,)` `int64` | Which graph each node came from |
| `ptr` | `(num_graphs + 1,)` `int64` | Node offsets; graph `i` owns `ptr[i]:ptr[i+1]` |
| `num_graphs` | `int` | Number of graphs batched |

Graphs whose `node_features` or `edge_features` widths disagree raise
`InputError` — a ligand `graph` and a ligand `bond_graph` cannot be batched
together.

## `feature_dims(molecule, mode)`

Answers the `in_channels` question in code instead of in documentation.

```python
dims = feature_dims("ligand", "graph")
# {"node_features": 94, "edge_features": 29}

model = MyGNN(in_channels=dims["node_features"], edge_dim=dims["edge_features"])
```

| Molecule | Mode | Dimensions |
|----------|------|-----------|
| `ligand` | `graph` | node 94, edge 29 |
| `ligand` | `bond_graph` | node 29, edge 96 |
| `ligand` | `fragment_graph` | node 62, edge 31 |
| `ligand` | `descriptor` | descriptors 62 |
| `protein` | `graph` | node 82, node_vector 31, edge 39, edge_vector 8 |
| `protein` | `atom_graph` | node 10, node_tokens 3, edge 6 |
| `nucleic_acid` | `graph` | node 23, node_tokens 1, edge 3 |
| `nucleic_acid` | `atom_graph` | node 0 (token-only: use `node_tokens`), node_tokens 1, edge 1 |

These are the widths of `as_graph(view)`, not of the raw `featurize` output —
ligand `graph` has no `edge_features` key at all, it has a dense
`adjacency (N, N, 37)`. Vector entries give the *number* of vectors; each
is 3-dimensional. The table
lives in `FEATURE_DIMS` and a test asserts every entry against real
featurization output, so it cannot drift from the code.

## Example: one model over three views

```python
from plmol import Ligand, as_graph, collate, feature_dims

ligands = [Ligand.from_smiles(s) for s in smiles_list]

for mode in ["graph", "bond_graph", "fragment_graph"]:
    batch = collate([lig.featurize(mode=mode)[mode] for lig in ligands])
    dims = feature_dims("ligand", mode)
    # Same array names and layout in every iteration; only the widths change.
    model = MyGNN(dims["node_features"], dims["edge_features"])
    out = model(batch["node_features"], batch["edge_index"],
                batch["edge_features"], batch["batch"])
```
