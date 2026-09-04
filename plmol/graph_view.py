"""One graph shape for every plmol view, plus batching.

plmol's graph outputs grew per molecule type and disagree on almost everything:
the ligand graph is a dense ``(N, N, C)`` adjacency, the protein residue graph
splits features into tuples of scalar and vector tensors, the protein atom graph
carries token ids plus a dozen loose per-atom and per-edge arrays, and the bond
and fragment graphs use ``edge_index``/``edge_features``. Across all of them
only ``coords`` and ``node_features`` are shared, and even those do not mean the
same thing.

``as_graph`` maps any of them onto one shape so a single model can consume them,
``collate`` batches those into one disconnected graph the way PyTorch Geometric
does, and ``feature_dims`` answers the ``in_channels`` question that otherwise
has to be read out of the documentation.

Nothing here changes what ``featurize`` returns; this is a view on top of it.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .arrays import FLOAT, INT
from .errors import InputError
from .ligand.graph_edge_features import bond_view_channels

# Per-edge keys of the protein atom graph, in the order they are concatenated
# into ``edge_features``. Documented so column indices are stable.
_ATOM_GRAPH_EDGE_KEYS = (
    "edge_distances",
    "same_residue",
    "sequence_separation",
    "unit_vector",
)

# Per-node continuous keys of the protein atom graph, likewise ordered.
# ``relative_sasa`` is deliberately absent: burial_index is exactly
# ``1 - relative_sasa`` (measured correlation -1.000000, maximum difference 0),
# so emitting both would put a column and its complement in the same vector.
# The raw ``atom_graph`` dict still exposes ``relative_sasa`` for callers that
# want it under that name.
_ATOM_GRAPH_NODE_KEYS = (
    "burial_index",
    "formal_charge",
    "is_backbone",
    "is_hbond_acceptor",
    "is_hbond_donor",
    "is_polar_sasa",
    "sasa",
    "secondary_structure",
)

# Token-valued node keys, kept as integers for nn.Embedding.
_ATOM_GRAPH_TOKEN_KEYS = ("atom_tokens", "residue_token", "atom_element")

# Nucleic acid residue graph: per-nucleotide keys, in concatenation order.
_NA_RESIDUE_NODE_KEYS = (
    "one_hot",
    "is_purine",
    "is_pyrimidine",
    "is_dna",
    "torsions",
    "sugar_pucker",
    "mol_weight",
    "n_hbond_donors",
    "n_hbond_acceptors",
)
_NA_RESIDUE_TOKEN_KEYS = ("nucleotide_type",)

# Nucleic acid atom graph: nodes are token-valued, edges carry a distance.
# Per-atom continuous keys of the nucleic acid atom graph, in concatenation
# order. There were none until 0.4.x: node_features came out of this view with
# a width of zero while the protein atom graph carried ten.
_NA_ATOM_NODE_KEYS = (
    "burial_index",
    "is_backbone",
    "is_polar_sasa",
    "relative_sasa",
    "sasa",
)

_NA_ATOM_TOKEN_KEYS = ("residue_token",)
_NA_ATOM_EDGE_KEYS = ("edge_distances",)

_CANONICAL_KEYS = (
    "node_features",
    "node_tokens",
    "node_vector_features",
    "edge_index",
    "edge_features",
    "edge_vector_features",
    "coords",
    "num_nodes",
    "num_edges",
    "source",
)


def as_graph(view: Dict[str, Any], source: Optional[str] = None) -> Dict[str, Any]:
    """Normalize any plmol graph view onto one shape.

    Args:
        view: A ``graph``, ``atom_graph``, ``bond_graph`` or ``fragment_graph``
            dict as returned by ``featurize``.
        source: Optional label recorded in the result; inferred when omitted.

    Returns:
        Dict with:
            - ``node_features`` ``(N, F)`` float32, continuous node features.
              Empty ``(N, 0)`` for views whose nodes are purely token-valued.
            - ``node_tokens`` ``(N, T)`` int64 or None, embedding inputs.
            - ``node_vector_features`` ``(N, V, 3)`` float32 or None, kept
              separate because SE(3)-equivariant models need them unflattened.
            - ``edge_index`` ``(2, E)`` int64.
            - ``edge_features`` ``(E, C)`` float32.
            - ``edge_vector_features`` ``(E, W, 3)`` float32 or None.
            - ``coords`` ``(N, 3)`` float32.
            - ``num_nodes``, ``num_edges``, ``source``.

    Everything comes back as numpy arrays regardless of whether the input held
    numpy arrays, since ``Ligand.featurize`` converts to numpy while the other
    classes do not.

    Raises:
        InputError: If the view is not a recognized plmol graph.
    """
    if not isinstance(view, dict):
        raise InputError(f"as_graph expects a graph dict, got {type(view).__name__}.")

    kind = source or _infer_source(view)
    if kind == "dense":
        return _from_dense_adjacency(view, source or "ligand_graph")
    if kind == "atom_graph":
        return _from_protein_atom_graph(view, source or "protein_atom_graph")
    if kind == "residue_graph":
        return _from_tuple_graph(view, source or "protein_residue_graph")
    if kind == "na_residue_graph":
        return _from_na_residue_graph(view, source or "nucleic_residue_graph")
    if kind == "na_atom_graph":
        return _from_na_atom_graph(view, source or "nucleic_atom_graph")
    if kind == "edge_index":
        return _from_edge_index_graph(view, source or "edge_index_graph")
    raise InputError(
        "Unrecognized graph view. Expected a dense 'adjacency', an 'edge_index' "
        "with 'edge_features', or one of the protein/nucleic acid graph modes. "
        "Note that 'backbone' and 'surface' are not graphs and have no edges."
    )


def _infer_source(view: Dict[str, Any]) -> Optional[str]:
    if "adjacency" in view and _ndim(view["adjacency"]) == 3:
        return "dense"
    if "edge_index" not in view:
        return None
    if "atom_tokens" in view:
        return "atom_graph"
    if isinstance(view.get("node_features"), tuple):
        return "residue_graph"
    if "torsions" in view and "one_hot" in view:
        return "na_residue_graph"
    if "residue_token" in view and "edge_distances" in view:
        return "na_atom_graph"
    if "edge_features" in view:
        return "edge_index"
    return None


# ---------------------------------------------------------------------------
# Per-view adapters
# ---------------------------------------------------------------------------


def _from_dense_adjacency(view: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Ligand ``graph``: dense (N, N, C) adjacency to sparse edges.

    Uses the same bond mask as ``LigandFeaturizer.adjacency_to_bond_edges``:
    the first four channels are the bond-type one-hot.
    """
    adjacency = _array(view["adjacency"])
    if adjacency.ndim != 3 or adjacency.shape[-1] < 4:
        raise InputError("adjacency must be [N, N, C] with C >= 4.")

    mask = view.get("bond_mask")
    if mask is None:
        mask = adjacency[..., :4].sum(axis=-1) > 0
    else:
        mask = _array(mask).astype(bool)
    mask = mask.copy()
    np.fill_diagonal(mask, False)

    src, dst = np.nonzero(mask)
    node_features = _float(view["node_features"])
    # Only the channels that describe a bonded pair; the rest of the pair block
    # is degenerate once the dense adjacency is unrolled into bonds.
    channels = bond_view_channels(adjacency.shape[-1])
    return _pack(
        node_features=node_features,
        node_tokens=None,
        node_vector_features=None,
        edge_index=np.stack([src, dst], axis=0),
        edge_features=adjacency[src, dst][:, channels].astype(FLOAT),
        edge_vector_features=None,
        coords=_coords(view, node_features.shape[0]),
        source=source,
    )


def _from_protein_atom_graph(view: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Protein ``atom_graph``: token ids plus loose per-atom and per-edge arrays."""
    edge_index = _array(view["edge_index"]).astype(INT)
    num_edges = int(edge_index.shape[1])
    num_nodes = int(_array(view["coords"]).shape[0])

    return _pack(
        node_features=_concat_columns(view, _ATOM_GRAPH_NODE_KEYS, num_nodes),
        node_tokens=_concat_tokens(view, _ATOM_GRAPH_TOKEN_KEYS, num_nodes),
        node_vector_features=None,
        edge_index=edge_index,
        edge_features=_concat_columns(view, _ATOM_GRAPH_EDGE_KEYS, num_edges),
        edge_vector_features=None,
        coords=_coords(view, num_nodes),
        source=source,
    )


def _from_tuple_graph(view: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Protein residue ``graph``: tuples of scalar and vector tensors."""
    node_features = _concat_tuple(view.get("node_features"), dim=-1)
    return _pack(
        node_features=node_features,
        node_tokens=None,
        node_vector_features=_concat_tuple(view.get("node_vector_features"), dim=1),
        edge_index=_array(view["edge_index"]).astype(INT),
        edge_features=_concat_tuple(view.get("edge_features"), dim=-1),
        edge_vector_features=_concat_tuple(view.get("edge_vector_features"), dim=1),
        coords=_coords(view, node_features.shape[0]),
        source=source,
    )


def _from_na_residue_graph(view: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Nucleic acid ``graph``: per-nucleotide arrays plus ``edge_attr``."""
    edge_index = _array(view["edge_index"]).astype(INT)
    num_nodes = int(_array(view["coords"]).shape[0])
    return _pack(
        node_features=_concat_columns(view, _NA_RESIDUE_NODE_KEYS, num_nodes),
        node_tokens=_concat_tokens(view, _NA_RESIDUE_TOKEN_KEYS, num_nodes),
        node_vector_features=None,
        edge_index=edge_index,
        edge_features=_float(view["edge_attr"]).reshape(int(edge_index.shape[1]), -1),
        edge_vector_features=None,
        coords=_coords(view, num_nodes),
        source=source,
    )


def _from_na_atom_graph(view: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Nucleic acid ``atom_graph``: token-valued nodes, distance-valued edges."""
    edge_index = _array(view["edge_index"]).astype(INT)
    num_edges = int(edge_index.shape[1])
    num_nodes = int(_array(view["coords"]).shape[0])
    return _pack(
        node_features=_concat_columns(view, _NA_ATOM_NODE_KEYS, num_nodes),
        node_tokens=_concat_tokens(view, _NA_ATOM_TOKEN_KEYS, num_nodes),
        node_vector_features=None,
        edge_index=edge_index,
        edge_features=_concat_columns(view, _NA_ATOM_EDGE_KEYS, num_edges),
        edge_vector_features=None,
        coords=_coords(view, num_nodes),
        source=source,
    )


def _from_edge_index_graph(view: Dict[str, Any], source: str) -> Dict[str, Any]:
    """``bond_graph`` / ``fragment_graph`` and anything already in this shape."""
    node_features = _float(view["node_features"])
    return _pack(
        node_features=node_features,
        node_tokens=None,
        node_vector_features=None,
        edge_index=_array(view["edge_index"]).astype(INT),
        edge_features=_float(view["edge_features"]),
        edge_vector_features=None,
        coords=_coords(view, node_features.shape[0]),
        source=source,
    )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def collate(views: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch normalized graphs into one disconnected graph.

    Node indices of each graph are offset by the running node count, so
    ``edge_index`` stays valid across the batch, and a ``batch`` vector records
    which graph each node came from. This is the layout PyTorch Geometric's
    ``Batch`` uses, so the result drops into models written against it.

    Args:
        views: Graphs from :func:`as_graph`. Raw ``featurize`` outputs are
            normalized on the way in.

    Returns:
        The canonical keys plus ``batch`` ``(N_total,)``, ``ptr``
        ``(num_graphs + 1,)`` and ``num_graphs``.

    Raises:
        InputError: If the sequence is empty or the graphs disagree on a
            feature width.
    """
    if not views:
        raise InputError("collate needs at least one graph.")

    graphs = [v if v.get("source") and "num_nodes" in v else as_graph(v) for v in views]
    _check_widths(graphs)

    node_offset = 0
    node_features, node_tokens, node_vectors, coords = [], [], [], []
    edge_index, edge_features, edge_vectors = [], [], []
    batch, ptr = [], [0]

    for graph_idx, graph in enumerate(graphs):
        n = graph["num_nodes"]
        node_features.append(graph["node_features"])
        coords.append(graph["coords"])
        if graph["node_tokens"] is not None:
            node_tokens.append(graph["node_tokens"])
        if graph["node_vector_features"] is not None:
            node_vectors.append(graph["node_vector_features"])

        edge_index.append(graph["edge_index"] + node_offset)
        edge_features.append(graph["edge_features"])
        if graph["edge_vector_features"] is not None:
            edge_vectors.append(graph["edge_vector_features"])

        batch.append(np.full(n, graph_idx, dtype=INT))
        node_offset += n
        ptr.append(node_offset)

    def stack(parts, dim=0):
        return np.concatenate(parts, axis=dim) if parts else None

    return {
        "node_features": stack(node_features),
        "node_tokens": stack(node_tokens) if len(node_tokens) == len(graphs) else None,
        "node_vector_features": (
            stack(node_vectors) if len(node_vectors) == len(graphs) else None
        ),
        "edge_index": stack(edge_index, dim=1),
        "edge_features": stack(edge_features),
        "edge_vector_features": (
            stack(edge_vectors) if len(edge_vectors) == len(graphs) else None
        ),
        "coords": stack(coords),
        "batch": stack(batch),
        "ptr": np.array(ptr, dtype=INT),
        "num_nodes": node_offset,
        "num_edges": int(sum(g["num_edges"] for g in graphs)),
        "num_graphs": len(graphs),
        "source": graphs[0]["source"],
    }


def _check_widths(graphs: List[Dict[str, Any]]) -> None:
    for key in ("node_features", "edge_features"):
        widths = {int(g[key].shape[-1]) for g in graphs}
        if len(widths) > 1:
            raise InputError(
                f"Cannot batch graphs with different {key} widths: {sorted(widths)}. "
                "Graphs from different views or molecule types are not compatible."
            )


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

#: Feature widths per molecule type and mode, as produced by :func:`as_graph`.
#: Kept in code rather than only in the docs so models can ask instead of
#: hardcoding; a test asserts these against real featurization output.
FEATURE_DIMS: Dict[str, Dict[str, Dict[str, int]]] = {
    "ligand": {
        "graph": {"node_features": 94, "edge_features": 29},
        "bond_graph": {"node_features": 29, "edge_features": 96},
        "fragment_graph": {"node_features": 62, "edge_features": 31},
        "descriptor": {"descriptors": 62},
    },
    "nucleic_acid": {
        "graph": {"node_features": 23, "node_tokens": 1, "edge_features": 3},
        "atom_graph": {"node_features": 5, "node_tokens": 1, "edge_features": 1},
    },
    "protein": {
        "graph": {
            "node_features": 82,
            "node_vector_features": 31,
            "edge_features": 39,
            "edge_vector_features": 8,
        },
        "atom_graph": {"node_features": 10, "node_tokens": 3, "edge_features": 6},
    },
}


def feature_dims(molecule: str, mode: str) -> Dict[str, int]:
    """Feature widths for a molecule type and mode, as :func:`as_graph` emits them.

    Args:
        molecule: ``"ligand"``, ``"protein"`` or ``"nucleic_acid"``.
        mode: A featurization mode, e.g. ``"graph"`` or ``"bond_graph"``.

    Returns:
        Mapping of canonical key to its width, e.g.
        ``{"node_features": 94, "edge_features": 37}``. Vector entries give the
        number of vectors, each of which is 3-dimensional.

    Raises:
        InputError: If the molecule type or mode has no recorded dimensions.
    """
    try:
        return dict(FEATURE_DIMS[molecule][mode])
    except KeyError:
        known = {m: sorted(v) for m, v in FEATURE_DIMS.items()}
        raise InputError(
            f"No recorded dimensions for molecule={molecule!r} mode={mode!r}. Known: {known}"
        ) from None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ndim(value: Any) -> int:
    return len(getattr(value, "shape", ()))


def _array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _float(value: Any) -> np.ndarray:
    return _array(value).astype(FLOAT)


def _coords(view: Dict[str, Any], num_nodes: int) -> np.ndarray:
    coords = view.get("coords")
    if coords is None:
        return np.zeros((num_nodes, 3), dtype=FLOAT)
    return _float(coords).reshape(num_nodes, -1)[:, :3]


def _pack(**kwargs: Any) -> Dict[str, Any]:
    kwargs["num_nodes"] = int(kwargs["node_features"].shape[0])
    kwargs["num_edges"] = int(kwargs["edge_index"].shape[1])
    return {key: kwargs[key] for key in _CANONICAL_KEYS}


def _concat_tuple(value: Any, dim: int) -> Optional[np.ndarray]:
    """Concatenate a tuple of tensors; pass a lone tensor through."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        return np.concatenate([_float(v) for v in value], axis=dim)
    return _float(value)


def _concat_columns(view: Dict[str, Any], keys: Iterable[str], rows: int) -> np.ndarray:
    """Stack the named per-row arrays into one float matrix."""
    columns = []
    for key in keys:
        value = view.get(key)
        if value is None:
            continue
        tensor = _float(value)
        columns.append(tensor.reshape(rows, -1) if tensor.ndim > 1 else tensor.reshape(rows, 1))
    if not columns:
        return np.zeros((rows, 0), dtype=FLOAT)
    return np.concatenate(columns, axis=-1)


def _concat_tokens(view: Dict[str, Any], keys: Iterable[str], rows: int) -> Optional[np.ndarray]:
    columns = [
        _array(view[key]).astype(INT).reshape(rows, 1) for key in keys if view.get(key) is not None
    ]
    return np.concatenate(columns, axis=-1) if columns else None
