"""Unified input loaders for ligand/protein sources."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..ligand.core import Ligand
from ..protein.core import Protein
from ..errors import DependencyError, InputError

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


def load_protein_input(
    protein_input: Union[str, Protein],
    *,
    standardize: bool = True,
    keep_hydrogens: bool = False,
) -> Protein:
    if isinstance(protein_input, Protein):
        return protein_input
    if not isinstance(protein_input, str):
        raise InputError(f"Unsupported protein input type: {type(protein_input)!r}")
    return Protein.from_pdb(
        protein_input,
        standardize=standardize,
        keep_hydrogens=keep_hydrogens,
    )


def load_ligand_input(
    ligand_input,
    *,
    add_hs: bool = False,
) -> Ligand:
    if isinstance(ligand_input, Ligand):
        return ligand_input
    if Chem is None:
        raise DependencyError("RDKit is required for ligand loading.")

    if hasattr(ligand_input, "GetNumAtoms"):
        return Ligand(ligand_input)

    if not isinstance(ligand_input, str):
        raise InputError(f"Unsupported ligand input type: {type(ligand_input)!r}")

    path = Path(ligand_input)
    if path.exists():
        ext = path.suffix.lower()
        if ext == ".sdf":
            return Ligand.from_sdf(str(path))
        if ext in {".mol", ".mol2", ".pdb"}:
            loader = {
                ".mol": Chem.MolFromMolFile,
                ".mol2": Chem.MolFromMol2File,
                ".pdb": Chem.MolFromPDBFile,
            }[ext]
            mol = loader(str(path), removeHs=not add_hs)
            if mol is None:
                raise InputError(f"Failed to load ligand file: {path}")
            return Ligand(mol)
        raise InputError(f"Unsupported ligand file extension: {ext}")

    return Ligand.from_smiles(ligand_input, add_hs=add_hs)
