"""Fragment-level graph construction for small molecules.

Coarsens the atom-wise molecular graph: every fragment becomes a node, and two
fragment nodes are connected by the bond that was cut between them. Together
with ``graph`` and ``bond_graph`` this completes the atom -> bond -> fragment ->
molecule hierarchy.

Fragment node features and edge features are read out of the fragmentation
result and the atom graph's dense adjacency rather than recomputed, so all
views stay numerically consistent.
"""

from typing import Any, Dict, Optional

import numpy as np
from rdkit import Chem

from ..arrays import FLOAT, INT
from ..errors import InputError
from .graph_edge_features import bond_view_channels

# Geometry channels appended to the cleaved bond's feature vector.
FRAGMENT_GEOMETRY_FEATURE_DIM = 2


def build_fragment_graph(
    mol: Chem.Mol,
    fragment_result: Dict[str, Any],
    adjacency: Any,
    coords: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the fragment-level graph that corresponds to an atom-wise graph.

    Args:
        mol: RDKit molecule the atom graph and the fragmentation came from. It
            must be that exact molecule: ``graph`` mode canonicalizes atom
            order, so a differently ordered copy maps fragments wrongly.
        fragment_result: Output of ``fragment_molecule`` and friends. Must carry
            ``fragment_features``, so it has to come from a call with
            ``compute_features=True``.
        adjacency: Dense atom adjacency ``(N, N, C)`` from ``graph`` mode. Edge
            features are taken from the cleaved bond's row.
        coords: Optional atom coordinates ``(N, 3)``. Fragment centroids and the
            geometry channels are computed when 3D coordinates are present and
            zero otherwise.

    Returns:
        Dict with:
            - ``node_features`` ``(F, 62)``: per-fragment descriptors.
            - ``edge_index`` ``(2, E)``: fragment pairs joined by a cleaved
              bond, both directions.
            - ``edge_features`` ``(E, C + 2)``: the informative channels of the
              cleaved bond, followed by ``[centroid distance, cleaved bond
              length]`` in Angstrom.
            - ``edge_cleaved_bond`` ``(E, 2)``: the atom pair cut for each edge.
            - ``coords`` ``(F, 3)``: fragment centroids.
            - ``adjacency`` ``(F, F)`` bool: fragment connectivity.
            - ``atom_to_fragment`` ``(N,)``, ``fragment_atom_indices``,
              ``fragment_smiles``, ``num_fragments``, ``num_fragment_edges``.

    Raises:
        InputError: If the atom graph does not match the molecule, or the
            fragmentation result was produced without fragment features.
    """
    adjacency = np.asarray(adjacency)
    channels = bond_view_channels(adjacency.shape[-1])
    edge_feature_dim = len(channels) + FRAGMENT_GEOMETRY_FEATURE_DIM

    num_atoms = mol.GetNumAtoms()
    atom_to_fragment = np.asarray(fragment_result["atom_to_fragment"], dtype=np.int64)

    if adjacency.shape[0] != num_atoms or atom_to_fragment.shape[0] != num_atoms:
        raise InputError(
            f"Atom graph does not match the molecule: adjacency has "
            f"{adjacency.shape[0]} atoms and atom_to_fragment "
            f"{atom_to_fragment.shape[0]}, but the molecule has {num_atoms}."
        )

    fragment_features = fragment_result.get("fragment_features")
    if fragment_features is None:
        raise InputError(
            "fragment_result has no fragment_features. Fragment nodes need "
            "per-fragment descriptors, so fragment it with compute_features=True."
        )

    node_features = np.asarray(fragment_features, dtype=FLOAT)
    num_fragments = int(fragment_result["num_fragments"])
    fragment_atom_indices = [list(atoms) for atoms in fragment_result["fragment_atom_indices"]]

    if coords is not None:
        coords = np.asarray(coords, dtype=FLOAT)
    has_3d = coords is not None and coords.shape[0] == num_atoms and bool(np.any(coords))

    # Fragment centroid, i.e. the mean position of its atoms.
    fragment_coords = np.zeros((num_fragments, 3), dtype=FLOAT)
    if has_3d:
        for frag_idx, atoms in enumerate(fragment_atom_indices):
            if atoms and frag_idx < num_fragments:
                fragment_coords[frag_idx] = coords[np.array(atoms, dtype=INT)].mean(0)

    # Each cleaved bond joins the two fragments its endpoints landed in. Bonds
    # whose endpoints ended up in one fragment (small-fragment merging) are
    # internal and contribute no edge, which matches ``fragment_adjacency``.
    cleaved = np.asarray(
        fragment_result.get("cleaved_bond_atoms", np.zeros((0, 2), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1, 2)

    src, dst, cut_atoms = [], [], []
    for begin, end in cleaved:
        frag_begin = int(atom_to_fragment[begin])
        frag_end = int(atom_to_fragment[end])
        if frag_begin == frag_end:
            continue
        src.append(frag_begin)
        dst.append(frag_end)
        cut_atoms.append((int(begin), int(end)))
        src.append(frag_end)
        dst.append(frag_begin)
        cut_atoms.append((int(end), int(begin)))

    num_edges = len(src)
    if num_edges:
        edge_index = np.array([src, dst], dtype=INT)
        edge_cleaved_bond = np.array(cut_atoms, dtype=INT)
        bond_features = adjacency[
            edge_cleaved_bond[:, 0], edge_cleaved_bond[:, 1]
        ][:, channels].astype(FLOAT)
        geometry = _fragment_geometry_features(
            edge_index, edge_cleaved_bond, fragment_coords, coords if has_3d else None
        )
        edge_features = np.concatenate([bond_features, geometry], axis=-1)
    else:
        edge_index = np.zeros((2, 0), dtype=INT)
        edge_cleaved_bond = np.zeros((0, 2), dtype=INT)
        edge_features = np.zeros((0, edge_feature_dim), dtype=FLOAT)

    fragment_adjacency = np.zeros((num_fragments, num_fragments), dtype=bool)
    if num_edges:
        fragment_adjacency[edge_index[0], edge_index[1]] = True

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_cleaved_bond": edge_cleaved_bond,
        "coords": fragment_coords,
        "adjacency": fragment_adjacency,
        "atom_to_fragment": atom_to_fragment.copy(),
        "fragment_atom_indices": fragment_atom_indices,
        "fragment_smiles": list(fragment_result["fragment_smiles"]),
        "num_fragments": num_fragments,
        "num_fragment_edges": int(num_edges),
    }


def _fragment_geometry_features(
    edge_index: np.ndarray,
    edge_cleaved_bond: np.ndarray,
    fragment_coords: np.ndarray,
    coords: Optional[np.ndarray],
) -> np.ndarray:
    """``[centroid distance, cleaved bond length]`` in Angstrom; zeros without 3D."""
    num_edges = edge_index.shape[1]
    if coords is None:
        return np.zeros((num_edges, FRAGMENT_GEOMETRY_FEATURE_DIM), dtype=FLOAT)

    centroid_distance = np.linalg.norm(
        fragment_coords[edge_index[0]] - fragment_coords[edge_index[1]], axis=-1
    )
    bond_length = np.linalg.norm(
        coords[edge_cleaved_bond[:, 0]] - coords[edge_cleaved_bond[:, 1]], axis=-1
    )
    return np.stack([centroid_distance, bond_length], axis=-1)
