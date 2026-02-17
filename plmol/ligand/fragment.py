"""Rotatable-bond fragmentation for small molecules."""

from typing import Any, Dict, List, Tuple

import numpy as np
from rdkit import Chem

from ..constants import ROTATABLE_BOND_SMARTS

_ROTATABLE_PATTERN = Chem.MolFromSmarts(ROTATABLE_BOND_SMARTS)


def fragment_on_rotatable_bonds(
    mol: Chem.Mol,
    min_fragment_size: int = 1,
) -> Dict[str, Any]:
    """Fragment a molecule by cutting at rotatable bonds.

    Args:
        mol: RDKit molecule.
        min_fragment_size: Fragments smaller than this are merged into the
            largest neighbouring fragment.

    Returns:
        Dict with keys: fragment_smiles, atom_to_fragment, fragment_adjacency,
        num_fragments, num_rotatable_bonds.
    """
    matches = mol.GetSubstructMatches(_ROTATABLE_PATTERN)
    bond_indices: List[int] = []
    bond_atom_pairs: List[Tuple[int, int]] = []
    for a, b in matches:
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is not None and bond.GetIdx() not in bond_indices:
            bond_indices.append(bond.GetIdx())
            bond_atom_pairs.append((a, b))

    num_atoms = mol.GetNumAtoms()
    num_rotatable = len(bond_indices)

    if num_rotatable == 0:
        return _single_fragment_result(mol, num_atoms)

    fragmented = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
    atom_map: List[List[int]] = []
    frag_mols = Chem.GetMolFrags(
        fragmented, asMols=True, fragsMolAtomMapping=atom_map
    )

    # Build atom_to_fragment mapping
    atom_to_frag = np.full(num_atoms, -1, dtype=np.int64)
    for frag_idx, atoms in enumerate(atom_map):
        for atom_idx in atoms:
            atom_to_frag[atom_idx] = frag_idx

    num_frags = len(frag_mols)

    # Build fragment adjacency from the rotatable bond pairs
    adj = np.zeros((num_frags, num_frags), dtype=np.int64)
    for a, b in bond_atom_pairs:
        fa, fb = atom_to_frag[a], atom_to_frag[b]
        if fa != fb:
            adj[fa, fb] = 1
            adj[fb, fa] = 1

    # Merge small fragments if requested
    if min_fragment_size > 1:
        frag_mols, atom_map, atom_to_frag, adj = _merge_small_fragments(
            frag_mols, atom_map, atom_to_frag, adj, min_fragment_size
        )
        num_frags = len(frag_mols)

    smiles_list = [Chem.MolToSmiles(m) for m in frag_mols]

    return {
        "fragment_smiles": smiles_list,
        "atom_to_fragment": atom_to_frag,
        "fragment_atom_indices": [list(atoms) for atoms in atom_map],
        "fragment_adjacency": adj,
        "num_fragments": num_frags,
        "num_rotatable_bonds": num_rotatable,
    }


def _single_fragment_result(mol: Chem.Mol, num_atoms: int) -> Dict[str, Any]:
    """Return result when there are no rotatable bonds."""
    return {
        "fragment_smiles": [Chem.MolToSmiles(mol)],
        "atom_to_fragment": np.zeros(num_atoms, dtype=np.int64),
        "fragment_atom_indices": [list(range(num_atoms))],
        "fragment_adjacency": np.zeros((1, 1), dtype=np.int64),
        "num_fragments": 1,
        "num_rotatable_bonds": 0,
    }


def _merge_small_fragments(
    frag_mols: List[Chem.Mol],
    atom_map: List[List[int]],
    atom_to_frag: np.ndarray,
    adj: np.ndarray,
    min_size: int,
) -> Tuple[List[Chem.Mol], List[List[int]], np.ndarray, np.ndarray]:
    """Merge fragments smaller than min_size into their largest neighbour."""
    sizes = [len(atoms) for atoms in atom_map]
    # Map from old index → merged index
    merge_target = list(range(len(sizes)))

    for i, sz in enumerate(sizes):
        if sz >= min_size:
            continue
        # Find the largest neighbour
        neighbours = np.where(adj[i] == 1)[0]
        if len(neighbours) == 0:
            continue
        best = max(neighbours, key=lambda n: sizes[merge_target[n]])
        target = merge_target[best]
        # Merge i into target
        merge_target[i] = target
        sizes[target] += sz

    # Resolve transitive merges
    for i in range(len(merge_target)):
        while merge_target[i] != merge_target[merge_target[i]]:
            merge_target[i] = merge_target[merge_target[i]]

    # Build new fragment list
    unique_targets = sorted(set(merge_target))
    old_to_new = {old: new for new, old in enumerate(unique_targets)}

    new_atom_map: List[List[int]] = [[] for _ in unique_targets]
    for old_idx, atoms in enumerate(atom_map):
        new_idx = old_to_new[merge_target[old_idx]]
        new_atom_map[new_idx].extend(atoms)

    # Update atom_to_frag
    new_atom_to_frag = atom_to_frag.copy()
    for old_idx in range(len(merge_target)):
        new_idx = old_to_new[merge_target[old_idx]]
        new_atom_to_frag[atom_to_frag == old_idx] = new_idx

    # Build new adjacency
    n_new = len(unique_targets)
    new_adj = np.zeros((n_new, n_new), dtype=np.int64)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 1:
                ni, nj = old_to_new[merge_target[i]], old_to_new[merge_target[j]]
                if ni != nj:
                    new_adj[ni, nj] = 1
                    new_adj[nj, ni] = 1

    # Create proper fragment mols by merging the frag_mols lists
    merged_frag_mols = []
    for new_idx in range(n_new):
        # Collect all old frag_mols that map to this new index
        parts = [frag_mols[old] for old, target in enumerate(merge_target)
                 if old_to_new[target] == new_idx]
        if len(parts) == 1:
            merged_frag_mols.append(parts[0])
        else:
            combined = parts[0]
            for p in parts[1:]:
                combined = Chem.CombineMols(combined, p)
            merged_frag_mols.append(combined)

    return merged_frag_mols, new_atom_map, new_atom_to_frag, new_adj
