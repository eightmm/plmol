"""Centralized RDKit utility functions for molecule preparation and conformer generation."""

from typing import Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def canonical_atom_order(mol: Chem.Mol) -> "list[int]":
    """``order[i]`` is the atom of *mol* that becomes atom ``i`` once canonical.

    The permutation :func:`canonicalize_mol` applies, on its own, so a caller
    can line indices from a canonicalised graph back up against the molecule
    it came from.
    """
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    order = [0] * len(ranks)
    for old_idx, new_idx in enumerate(ranks):
        order[new_idx] = old_idx
    return order


def canonicalize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Reorder atoms to canonical order, preserving coordinates if present."""
    new_mol = Chem.RenumberAtoms(mol, canonical_atom_order(mol))
    Chem.FastFindRings(new_mol)
    return new_mol


def prepare_mol(
    mol_or_smiles: Union[str, Chem.Mol],
    add_hs: bool = False,
    canonicalize: bool = True,
    make_copy: bool = True,
) -> Chem.Mol:
    """Prepare molecule from SMILES or RDKit mol with optional canonicalization and hydrogens.

    Args:
        mol_or_smiles: RDKit mol object or SMILES string.
        add_hs: Whether to add explicit hydrogens.
        canonicalize: Whether to reorder atoms to canonical order.
        make_copy: Whether to copy the input mol (ignored for SMILES input).

    Returns:
        Prepared RDKit mol object.

    Raises:
        ValueError: If SMILES string is invalid.
    """
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
    else:
        if make_copy:
            mol = Chem.RWMol(mol_or_smiles).GetMol()
            Chem.FastFindRings(mol)
        else:
            mol = mol_or_smiles

    if canonicalize and mol is not None:
        mol = canonicalize_mol(mol)

    if add_hs and mol is not None:
        if has_3d(mol):
            mol = Chem.AddHs(mol, addCoords=True)
        else:
            mol = Chem.AddHs(mol)

    return mol


def ensure_3d_conformer(
    mol: Chem.Mol,
    random_seed: int = 42,
    optimize: bool = True,
) -> Optional[Chem.Mol]:
    """Return molecule with 3D conformer, generating one if needed.

    Uses ETKDGv3 with MMFF optimization by default, falls back to
    random coordinates if standard embedding fails.

    Returns:
        New mol with conformer, or None on failure.
    """
    if mol.GetNumConformers() > 0 and mol.GetConformer(0).Is3D():
        return mol
    mol_3d = Chem.AddHs(Chem.RWMol(mol).GetMol())
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    status = AllChem.EmbedMolecule(mol_3d, params)
    if status == -1:
        status = AllChem.EmbedMolecule(
            mol_3d, randomSeed=random_seed, useRandomCoords=True
        )
        if status == -1:
            return None
    if optimize:
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
    return mol_3d


def has_3d(mol: Chem.Mol) -> bool:
    """Check whether molecule has at least one 3D conformer."""
    return mol.GetNumConformers() > 0 and mol.GetConformer(0).Is3D()


def get_positions(mol: Chem.Mol) -> np.ndarray:
    """Extract 3D coordinates from first conformer as (N, 3) array.

    Raises:
        ValueError: If molecule has no conformer.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")
    conf = mol.GetConformer(0)
    return np.asarray(conf.GetPositions(), dtype=np.float64).reshape(-1, 3)
