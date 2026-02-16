"""Backbone features for inverse folding models (ProteinMPNN/ESM-IF/GVP).

Provides chain-boundary-aware dihedral computation, kNN graph construction,
and full backbone feature assembly.
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from .geometry import (
    calculate_dihedral,
    calculate_local_frames,
    calculate_virtual_cb,
    rbf_encode,
)


def compute_backbone_dihedrals(
    coords: torch.Tensor,
    chain_indices: Dict[str, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chain-boundary-aware phi/psi/omega.

    Reuses the vectorized calculate_dihedral() per chain,
    then remaps the output columns to [phi, psi, omega].

    calculate_dihedral(N-CA-C coords) produces (n, 3) per chain where:
      col 0 = phi_i  (0 at first residue)
      col 1 = psi_i  (0 at last residue)
      col 2 = omega_{i+1} (needs shift: omega_i = result[i-1, 2])

    Args:
        coords: (L, MAX_ATOMS, 3) — index 0=N, 1=CA, 2=C
        chain_indices: chain_id -> sorted list of residue indices

    Returns:
        dihedrals: (L, 3) — [phi, psi, omega] in radians
        mask: (L, 3) — bool, True where valid
    """
    L = coords.shape[0]
    dihedrals = torch.zeros(L, 3)
    mask = torch.zeros(L, 3, dtype=torch.bool)

    for chain_id, indices in chain_indices.items():
        idx = sorted(indices)
        n = len(idx)
        if n < 2:
            continue

        idx_t = torch.tensor(idx, dtype=torch.long)
        # Extract N-CA-C for this chain and call vectorized dihedral
        chain_nac = coords[idx_t, :3, :]  # (n, 3, 3)
        raw = calculate_dihedral(chain_nac)  # (n, 3)

        # phi: raw[:, 0] — valid for positions 1..n-1 (first is padding 0)
        dihedrals[idx_t[1:], 0] = raw[1:, 0]
        mask[idx_t[1:], 0] = True

        # psi: raw[:, 1] — valid for positions 0..n-2 (last is padding 0)
        dihedrals[idx_t[:n - 1], 1] = raw[:n - 1, 1]
        mask[idx_t[:n - 1], 1] = True

        # omega: raw[i-1, 2] = omega_i — valid for positions 1..n-1
        dihedrals[idx_t[1:], 2] = raw[:n - 1, 2]
        mask[idx_t[1:], 2] = True

    return dihedrals, mask


def build_backbone_knn_graph(
    coords: torch.Tensor,
    k: int = 30,
    chain_indices: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, torch.Tensor]:
    """kNN graph over CA atoms.

    Args:
        coords: (L, MAX_ATOMS, 3) — index 1=CA
        k: number of neighbors (clamped to L-1)
        chain_indices: for same_chain/seq_sep computation

    Returns:
        edge_index: (2, E) — E = L * k_eff
        edge_dist: (E,) — CA-CA distance
        edge_unit_vec: (E, 3) — unit vector i->j
        edge_seq_sep: (E,) — |i-j| within same chain (0 for cross-chain)
        edge_same_chain: (E,) — bool
    """
    ca = coords[:, 1]  # (L, 3)
    L = ca.shape[0]
    k_eff = min(k, L - 1)

    # CA-CA distance matrix
    dist_matrix = torch.cdist(ca, ca)  # (L, L)
    # Set self-distance to inf
    dist_matrix.fill_diagonal_(float('inf'))

    # kNN: top-k smallest distances per node
    topk_dist, topk_idx = torch.topk(dist_matrix, k_eff, dim=1, largest=False)

    # Build edge_index
    src = torch.arange(L).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    dst = topk_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)  # (2, L*k_eff)

    # Edge distances
    edge_dist = topk_dist.reshape(-1)

    # Unit vectors i->j
    diff = ca[dst] - ca[src]  # (E, 3)
    edge_unit_vec = F.normalize(diff, dim=-1)

    # Sequence separation and same-chain flag
    E = edge_index.shape[1]
    edge_seq_sep = torch.zeros(E, dtype=torch.long)
    edge_same_chain = torch.zeros(E, dtype=torch.bool)

    if chain_indices is not None:
        # Build residue-to-chain mapping
        res_to_chain = torch.full((L,), -1, dtype=torch.long)
        for chain_idx, (chain_id, indices) in enumerate(chain_indices.items()):
            for ri in indices:
                res_to_chain[ri] = chain_idx

        src_chain = res_to_chain[src]
        dst_chain = res_to_chain[dst]
        same_chain = src_chain == dst_chain
        edge_same_chain = same_chain

        # Sequence separation: |i - j| for same chain, 0 for cross-chain
        edge_seq_sep = torch.where(
            same_chain,
            (src - dst).abs(),
            torch.zeros_like(src),
        )

    return {
        'edge_index': edge_index,
        'edge_dist': edge_dist,
        'edge_unit_vec': edge_unit_vec,
        'edge_seq_sep': edge_seq_sep,
        'edge_same_chain': edge_same_chain,
    }


def compute_edge_frame_features(
    ca_coords: torch.Tensor,
    orientation_frames: torch.Tensor,
    edge_index: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute SE(3)-invariant edge features in local reference frames.

    For each edge (i, j):
    - edge_local_pos: position of CA_j in residue i's local frame (R_i^T @ (CA_j - CA_i))
    - edge_rel_orient: relative orientation between frames (R_i^T @ R_j)

    Args:
        ca_coords: (L, 3) CA atom coordinates.
        orientation_frames: (L, 3, 3) local frames (columns = basis vectors).
        edge_index: (2, E) source and destination indices.

    Returns:
        edge_local_pos: (E, 3) neighbor position in source local frame.
        edge_rel_orient: (E, 3, 3) relative rotation matrix.
    """
    src, dst = edge_index[0], edge_index[1]

    # Displacement in world space
    disp = ca_coords[dst] - ca_coords[src]  # (E, 3)

    # Project displacement into source residue's local frame
    # frames[:, :, k] = k-th basis vector → R^T @ disp = [x·disp, y·disp, z·disp]
    frames_src = orientation_frames[src]  # (E, 3, 3)
    edge_local_pos = torch.einsum('edk,ed->ek', frames_src, disp)  # (E, 3)

    # Relative orientation: R_src^T @ R_dst
    frames_dst = orientation_frames[dst]  # (E, 3, 3)
    edge_rel_orient = torch.bmm(
        frames_src.transpose(1, 2), frames_dst
    )  # (E, 3, 3)

    return {
        'edge_local_pos': edge_local_pos,
        'edge_rel_orient': edge_rel_orient,
    }


def compute_backbone_features(
    coords: torch.Tensor,
    residues: list,
    residue_types: torch.Tensor,
    k_neighbors: int = 30,
) -> Dict[str, Any]:
    """Full backbone feature assembly.

    Builds chain_indices, computes virtual CB, dihedrals, local frames,
    and kNN graph in one call.

    Args:
        coords: (L, MAX_ATOMS, 3) full residue coordinates.
        residues: list of (chain_id, res_num, res_type) tuples.
        residue_types: (L,) integer residue type tensor.
        k_neighbors: Number of nearest neighbors for kNN graph.

    Returns:
        Dict with backbone_coords, cb_coords, dihedrals, dihedrals_mask,
        orientation_frames, residue_types, chain_ids, residue_mask,
        edge_index, edge_dist, edge_unit_vec, edge_seq_sep,
        edge_same_chain, edge_rbf, edge_local_pos, edge_rel_orient,
        num_residues, num_chains, k_neighbors.
    """
    num_residues = len(residues)

    # Build chain_indices: chain_id -> list of residue indices
    chain_indices: Dict[str, list] = {}
    for idx, (chain, res_num, res_type) in enumerate(residues):
        if chain not in chain_indices:
            chain_indices[chain] = []
        chain_indices[chain].append(idx)

    # Backbone coords: N(0), CA(1), C(2), O(3)
    backbone_coords = coords[:, :4, :]  # (L, 4, 3)

    # Residue mask: True if all 4 backbone atoms are present (non-zero)
    backbone_norms = torch.norm(backbone_coords, dim=-1)  # (L, 4)
    residue_mask = (backbone_norms > 0).all(dim=1)  # (L,)

    # Virtual CB
    cb_coords = calculate_virtual_cb(coords)

    # Chain-aware backbone dihedrals
    dihedrals, dihedrals_mask = compute_backbone_dihedrals(coords, chain_indices)

    # Sin/cos encoding of dihedrals (L, 6): [sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)]
    dihedrals_sincos = torch.zeros(num_residues, 6)
    dihedrals_sincos[:, 0] = torch.sin(dihedrals[:, 0])
    dihedrals_sincos[:, 1] = torch.cos(dihedrals[:, 0])
    dihedrals_sincos[:, 2] = torch.sin(dihedrals[:, 1])
    dihedrals_sincos[:, 3] = torch.cos(dihedrals[:, 1])
    dihedrals_sincos[:, 4] = torch.sin(dihedrals[:, 2])
    dihedrals_sincos[:, 5] = torch.cos(dihedrals[:, 2])
    # Zero out invalid positions (where mask is False)
    dihedrals_sincos[:, 0:2] *= dihedrals_mask[:, 0:1].float()
    dihedrals_sincos[:, 2:4] *= dihedrals_mask[:, 1:2].float()
    dihedrals_sincos[:, 4:6] *= dihedrals_mask[:, 2:3].float()

    # Local orientation frames
    orientation_frames = calculate_local_frames(coords)

    # Chain IDs as integer tensor
    chain_id_map = {cid: i for i, cid in enumerate(sorted(chain_indices.keys()))}
    chain_ids = torch.tensor(
        [chain_id_map[residues[i][0]] for i in range(num_residues)],
        dtype=torch.long,
    )

    # kNN graph
    graph = build_backbone_knn_graph(coords, k=k_neighbors, chain_indices=chain_indices)

    # RBF encoding of CA-CA distances
    edge_rbf = rbf_encode(graph['edge_dist'])  # (E, 16)

    # SE(3)-invariant edge features in local frames
    ca_coords = coords[:, 1]  # (L, 3)
    frame_feats = compute_edge_frame_features(
        ca_coords, orientation_frames, graph['edge_index']
    )

    return {
        'backbone_coords': backbone_coords,
        'cb_coords': cb_coords,
        'dihedrals': dihedrals,
        'dihedrals_sincos': dihedrals_sincos,
        'dihedrals_mask': dihedrals_mask,
        'orientation_frames': orientation_frames,
        'residue_types': residue_types,
        'chain_ids': chain_ids,
        'residue_mask': residue_mask,
        'edge_index': graph['edge_index'],
        'edge_dist': graph['edge_dist'],
        'edge_unit_vec': graph['edge_unit_vec'],
        'edge_seq_sep': graph['edge_seq_sep'],
        'edge_same_chain': graph['edge_same_chain'],
        'edge_rbf': edge_rbf,
        'edge_local_pos': frame_feats['edge_local_pos'],
        'edge_rel_orient': frame_feats['edge_rel_orient'],
        'num_residues': num_residues,
        'num_chains': len(chain_indices),
        'k_neighbors': k_neighbors,
    }
