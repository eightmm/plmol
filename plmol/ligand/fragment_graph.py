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
import torch
from rdkit import Chem

from ..errors import InputError
from .line_graph import _bond_view_channels

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
    adjacency = torch.as_tensor(adjacency)
    channels = _bond_view_channels(adjacency.shape[-1])
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

    node_features = torch.as_tensor(np.asarray(fragment_features, dtype=np.float32))
    num_fragments = int(fragment_result["num_fragments"])
    fragment_atom_indices = [list(atoms) for atoms in fragment_result["fragment_atom_indices"]]

    if coords is not None:
        coords = torch.as_tensor(coords, dtype=torch.float32)
    has_3d = coords is not None and coords.shape[0] == num_atoms and bool(torch.any(coords))

    # Fragment centroid, i.e. the mean position of its atoms.
    fragment_coords = torch.zeros((num_fragments, 3), dtype=torch.float32)
    if has_3d:
        for frag_idx, atoms in enumerate(fragment_atom_indices):
            if atoms and frag_idx < num_fragments:
                fragment_coords[frag_idx] = coords[torch.tensor(atoms, dtype=torch.long)].mean(0)

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
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_cleaved_bond = torch.tensor(cut_atoms, dtype=torch.long)
        bond_features = adjacency[
            edge_cleaved_bond[:, 0], edge_cleaved_bond[:, 1]
        ][:, channels].to(torch.float32)
        geometry = _fragment_geometry_features(
            edge_index, edge_cleaved_bond, fragment_coords, coords if has_3d else None
        )
        edge_features = torch.cat([bond_features, geometry], dim=-1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_cleaved_bond = torch.zeros((0, 2), dtype=torch.long)
        edge_features = torch.zeros((0, edge_feature_dim), dtype=torch.float32)

    fragment_adjacency = torch.zeros((num_fragments, num_fragments), dtype=torch.bool)
    if num_edges:
        fragment_adjacency[edge_index[0], edge_index[1]] = True

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_cleaved_bond": edge_cleaved_bond,
        "coords": fragment_coords,
        "adjacency": fragment_adjacency,
        "atom_to_fragment": torch.from_numpy(atom_to_fragment.copy()),
        "fragment_atom_indices": fragment_atom_indices,
        "fragment_smiles": list(fragment_result["fragment_smiles"]),
        "num_fragments": num_fragments,
        "num_fragment_edges": int(num_edges),
    }


def _fragment_geometry_features(
    edge_index: torch.Tensor,
    edge_cleaved_bond: torch.Tensor,
    fragment_coords: torch.Tensor,
    coords: Optional[torch.Tensor],
) -> torch.Tensor:
    """``[centroid distance, cleaved bond length]`` in Angstrom; zeros without 3D."""
    num_edges = edge_index.shape[1]
    if coords is None:
        return torch.zeros((num_edges, FRAGMENT_GEOMETRY_FEATURE_DIM), dtype=torch.float32)

    centroid_distance = torch.linalg.norm(
        fragment_coords[edge_index[0]] - fragment_coords[edge_index[1]], dim=-1
    )
    bond_length = torch.linalg.norm(
        coords[edge_cleaved_bond[:, 0]] - coords[edge_cleaved_bond[:, 1]], dim=-1
    )
    return torch.stack([centroid_distance, bond_length], dim=-1)
