"""Bond-wise (line) graph construction for small molecules.

Inverts the atom-wise molecular graph: every bond becomes a node, and two bond
nodes are connected when they share an atom. The shared atom supplies the edge
feature, so the roles of atoms and bonds are exactly swapped relative to
``MoleculeGraphFeaturizer`` output.

Bond node features are read straight out of the atom graph's dense adjacency
rather than recomputed, which keeps the two views numerically consistent.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem

from ..arrays import FLOAT, INT
from ..errors import InputError
from .graph_edge_features import bond_view_channels

# Bond angle channels appended to the shared atom's feature vector.
BOND_ANGLE_FEATURE_DIM = 2


def build_bond_graph(
    mol: Chem.Mol,
    adjacency: Any,
    node_features: Any,
    coords: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the bond-wise graph that corresponds to an atom-wise graph.

    Args:
        mol: RDKit molecule the atom graph was built from. It must be that
            exact molecule: ``graph`` mode canonicalizes atom order, so a
            differently ordered copy of the same structure gives wrong bonds.
        adjacency: Dense atom adjacency ``(N, N, C)`` from ``graph`` mode. Bond
            node features are taken from ``adjacency[begin, end]``.
        node_features: Atom node features ``(N, F)``. The shared atom's row
            becomes the leading part of each bond-graph edge feature.
        coords: Optional atom coordinates ``(N, 3)``. Bond angles are computed
            when 3D coordinates are present and zero otherwise.

    Returns:
        Dict with:
            - ``node_features`` ``(B, C)``: per-bond features, the informative
              channels of ``adjacency[begin, end]``.
            - ``edge_index`` ``(2, E)``: bond pairs sharing an atom, both
              directions.
            - ``edge_features`` ``(E, F + 2)``: shared atom features followed by
              ``[cos(theta), theta / pi]`` for the angle the two bonds subtend.
            - ``edge_shared_atom`` ``(E,)``: atom index bridging each edge.
            - ``bond_index`` ``(B, 2)``: the atom pair each bond node came from.
            - ``atom_to_bonds``: per-atom list of incident bond indices.
            - ``coords`` ``(B, 3)``: bond midpoints.
            - ``adjacency`` ``(B, B)`` bool: bond-node connectivity.
            - ``num_bonds``, ``num_bond_edges``.
    """
    adjacency = np.asarray(adjacency)
    node_features = np.asarray(node_features)
    atom_feature_dim = int(node_features.shape[-1])
    channels = bond_view_channels(adjacency.shape[-1])
    bond_feature_dim = len(channels)
    edge_feature_dim = atom_feature_dim + BOND_ANGLE_FEATURE_DIM

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    if adjacency.shape[0] != num_atoms or node_features.shape[0] != num_atoms:
        raise InputError(
            f"Atom graph does not match the molecule: adjacency has "
            f"{adjacency.shape[0]} atoms and node_features {node_features.shape[0]}, "
            f"but the molecule has {num_atoms}."
        )

    if coords is not None:
        coords = np.asarray(coords, dtype=FLOAT)
    has_3d = coords is not None and coords.shape[0] == num_atoms and bool(np.any(coords))

    if num_bonds == 0:
        return {
            "node_features": np.zeros((0, bond_feature_dim), dtype=adjacency.dtype),
            "edge_index": np.zeros((2, 0), dtype=INT),
            "edge_features": np.zeros((0, edge_feature_dim), dtype=FLOAT),
            "edge_shared_atom": np.zeros((0,), dtype=INT),
            "bond_index": np.zeros((0, 2), dtype=INT),
            "atom_to_bonds": [[] for _ in range(num_atoms)],
            "coords": np.zeros((0, 3), dtype=FLOAT),
            "adjacency": np.zeros((0, 0), dtype=bool),
            "num_bonds": 0,
            "num_bond_edges": 0,
        }

    # Bond nodes follow RDKit bond index order, which is stable for a given mol.
    bond_pairs = np.empty((num_bonds, 2), dtype=np.int64)
    atom_to_bonds: List[List[int]] = [[] for _ in range(num_atoms)]
    for bond in mol.GetBonds():
        b_idx = bond.GetIdx()
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_pairs[b_idx, 0] = begin
        bond_pairs[b_idx, 1] = end
        atom_to_bonds[begin].append(b_idx)
        atom_to_bonds[end].append(b_idx)

    bond_index = bond_pairs
    begin_idx = bond_index[:, 0]
    end_idx = bond_index[:, 1]

    # The atom adjacency is symmetric across every channel, so either direction
    # yields the same bond vector. Only the channels that say something about a
    # bonded pair are kept; see BOND_VIEW_CHANNELS.
    bond_node_features = adjacency[begin_idx, end_idx][:, channels]

    # Two bonds are adjacent when they share an atom; emit both directions to
    # match the atom graph's bidirectional edge convention.
    src: List[int] = []
    dst: List[int] = []
    shared: List[int] = []
    for atom_idx, incident in enumerate(atom_to_bonds):
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                b1, b2 = incident[i], incident[j]
                src.append(b1)
                dst.append(b2)
                src.append(b2)
                dst.append(b1)
                shared.append(atom_idx)
                shared.append(atom_idx)

    num_edges = len(src)
    edge_index = np.array([src, dst], dtype=INT) if num_edges else np.zeros(
        (2, 0), dtype=INT
    )
    edge_shared_atom = np.array(shared, dtype=INT) if num_edges else np.zeros(
        (0,), dtype=INT
    )

    if num_edges:
        shared_atom_features = node_features[edge_shared_atom].astype(FLOAT)
        angle_features = _bond_angle_features(
            bond_index, edge_index, edge_shared_atom, coords if has_3d else None
        )
        edge_features = np.concatenate([shared_atom_features, angle_features], axis=-1)
    else:
        edge_features = np.zeros((0, edge_feature_dim), dtype=FLOAT)

    bond_adjacency = np.zeros((num_bonds, num_bonds), dtype=bool)
    if num_edges:
        bond_adjacency[edge_index[0], edge_index[1]] = True

    if has_3d:
        bond_coords = (coords[begin_idx] + coords[end_idx]) * 0.5
    else:
        bond_coords = np.zeros((num_bonds, 3), dtype=FLOAT)

    return {
        "node_features": bond_node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_shared_atom": edge_shared_atom,
        "bond_index": bond_index,
        "atom_to_bonds": atom_to_bonds,
        "coords": bond_coords,
        "adjacency": bond_adjacency,
        "num_bonds": int(num_bonds),
        "num_bond_edges": int(num_edges),
    }


def _bond_angle_features(
    bond_index: np.ndarray,
    edge_index: np.ndarray,
    edge_shared_atom: np.ndarray,
    coords: Optional[np.ndarray],
) -> np.ndarray:
    """``[cos(theta), theta / pi]`` per bond-graph edge; zeros without 3D coords."""
    num_edges = edge_index.shape[1]
    if coords is None:
        return np.zeros((num_edges, BOND_ANGLE_FEATURE_DIM), dtype=FLOAT)

    # Outer atom of each incident bond, i.e. the end that is not shared.
    src_pairs = bond_index[edge_index[0]]
    dst_pairs = bond_index[edge_index[1]]
    src_outer = np.where(src_pairs[:, 0] == edge_shared_atom, src_pairs[:, 1], src_pairs[:, 0])
    dst_outer = np.where(dst_pairs[:, 0] == edge_shared_atom, dst_pairs[:, 1], dst_pairs[:, 0])

    center = coords[edge_shared_atom]
    v1 = coords[src_outer] - center
    v2 = coords[dst_outer] - center
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    valid = (n1 > 1e-8) & (n2 > 1e-8)
    denom = np.where(valid, n1 * n2, np.ones_like(n1))
    cos_theta = np.clip((v1 * v2).sum(axis=-1) / denom, -1.0, 1.0)
    cos_theta = np.where(valid, cos_theta, np.zeros_like(cos_theta))
    theta = np.where(valid, np.arccos(cos_theta) / np.pi, np.zeros_like(cos_theta))
    return np.stack([cos_theta, theta], axis=-1)
