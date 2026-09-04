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


#: RDKit stops at 1000 matches and 1000 recursive sub-matches unless told
#: otherwise, and returns what it found without a word. On a whole protein that
#: silently truncates: the hydrophobic pharmacophore matches 1594 atoms in the
#: example structure and the default hands back 1000 of them, whichever 1000
#: come first in the file. Reversing the two chains of that structure changed
#: the detected protein-ligand interactions from 53 to 45.
#:
#: The recursive limit matters on its own. Patterns here exclude with
#: ``!$(...)``, and an exclusion that runs out of budget stops excluding: the
#: h_acceptor pattern accepted two atoms it should not, and the positive one
#: rejected seventeen it should have kept.
_MATCH_PARAMS = Chem.SubstructMatchParameters()
_MATCH_PARAMS.maxMatches = 10 ** 8
_MATCH_PARAMS.maxRecursiveMatches = 10 ** 8
_MATCH_PARAMS.uniquify = True


def substructure_matches(mol: Chem.Mol, pattern: Chem.Mol) -> tuple:
    """Every match of ``pattern`` in ``mol``, with no silent cap.

    Use this rather than ``mol.GetSubstructMatches(pattern)`` anywhere the
    molecule can be larger than a small ligand.
    """
    return mol.GetSubstructMatches(pattern, _MATCH_PARAMS)


_BOND_ORDERS = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
    "QUADRUPLE": Chem.BondType.QUADRUPLE,
    "AROMATIC": Chem.BondType.AROMATIC,
}


def mol_from_pdb_file(path: str, remove_hs: bool = False):
    """A PDB file as an RDKit molecule, tolerating a bond RDKit should not have made.

    ``MolFromPDBFile`` infers connectivity from distance, and on an older or
    lower-resolution structure it sometimes joins two atoms that are merely
    close: in 4HHB it bonds the CG and CE1 of a histidine, 1,3 across the
    imidazole ring, which leaves the carbon with five bonds and makes
    sanitisation refuse the whole file. Reading it strictly returned nothing at
    all, so the interaction featurizer had no protein to work with.

    Strict sanitisation is tried first and the valence check is skipped only if
    it fails. On a structure that passes strictly the two agree exactly -- same
    bonds, and the same matches for every pharmacophore pattern.
    """
    mol = Chem.MolFromPDBFile(path, removeHs=remove_hs)
    if mol is not None:
        return mol
    mol = Chem.MolFromPDBFile(path, removeHs=remove_hs, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    except Exception:
        return None
    return mol


def mol_from_component_bonds(mol: Chem.Mol, bonds: dict) -> "tuple[Optional[Chem.Mol], dict]":
    """Rebuild a PDB-derived molecule's bonds from a component bond table.

    Coordinates say which atoms are close, not which are bonded, and RDKit's
    proximity bonding gets it wrong differently for every copy: the four hemes
    of 4HHB come out with 48, 49, 49 and 51 bonds, some invented and some
    missing. Setting bond orders on that cannot repair it -- an invented bond
    is still there afterwards, and the molecule then fails to sanitize.

    So the bonds are taken from the table instead, which says exactly which
    atom names are joined and how. *mol* is expected to carry the atoms and no
    bonds (read with ``proximityBonding=False``); *bonds* is what
    :meth:`plmol.parsers.mmcif_parser.MMCIFParser.get_component_bonds` returns
    for this component.

    A component whose aromatic ring will not kekulize without knowing where its
    hydrogens sit is left to the caller's fallback rather than repaired. Setting
    each heavy atom's hydrogen count from the table was measured on 56 ligands
    and made things worse, 50 usable to 45: the component's protonation is not
    always the deposited one. Adding a hydrogen to an aromatic nitrogen until it
    kekulizes recovered one more but returned a molecule whose ring was no
    longer aromatic, which is a wrong answer that parses.

    Returns:
        ``(molecule, report)``. The molecule is ``None`` when the result will
        not sanitize, so the caller can fall back. The report counts
        ``bonds_applied``, ``skipped_bonds`` -- table entries naming an atom
        the model does not have, which covers every hydrogen when hydrogens
        were dropped and is how a truncated ligand shows up -- and
        ``atoms_unknown``, model atoms the table does not name.
    """
    report = {"bonds_applied": 0, "skipped_bonds": 0, "atoms_unknown": 0}
    if not bonds or mol is None:
        return None, report

    index_of = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            return None, report
        index_of.setdefault(info.GetName().strip(), atom.GetIdx())

    named = set(index_of)
    report["atoms_unknown"] = sum(
        1 for name in named
        if not any(name in pair for pair in bonds)
    )

    editable = Chem.RWMol(mol)
    for pair, order_name in bonds.items():
        first, second = tuple(pair)
        left, right = index_of.get(first), index_of.get(second)
        if left is None or right is None:
            report["skipped_bonds"] += 1
            continue
        order = _BOND_ORDERS.get(order_name)
        if order is None or editable.GetBondBetweenAtoms(left, right) is not None:
            continue
        editable.AddBond(left, right, order)
        report["bonds_applied"] += 1

    if not report["bonds_applied"]:
        return None, report

    for bond in editable.GetBonds():
        bond.SetIsAromatic(bond.GetBondType() == Chem.BondType.AROMATIC)
    for atom in editable.GetAtoms():
        atom.SetIsAromatic(any(b.GetIsAromatic() for b in atom.GetBonds()))
        atom.SetNoImplicit(False)

    result = editable.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception:
        # A nitrogen with four bonds and no charge is a quaternary ammonium the
        # file did not mark; the component tables here carry no charge column.
        # Charging it is the one repair worth trying, and only that.
        repaired = Chem.RWMol(result)
        charged = False
        for atom in repaired.GetAtoms():
            if (atom.GetSymbol() == "N" and atom.GetFormalCharge() == 0
                    and atom.GetExplicitValence() == 4):
                atom.SetFormalCharge(1)
                charged = True
        if not charged:
            return None, report
        result = repaired.GetMol()
        try:
            Chem.SanitizeMol(result)
        except Exception:
            return None, report
        report["nitrogens_charged"] = True
    # The table carries no stereochemistry and the atoms arrive without bonds,
    # so nothing has assigned any. The coordinates have it: without this the
    # rebuilt ligand came back flat where the deposited SDF has four centres.
    if result.GetNumConformers():
        try:
            Chem.AssignStereochemistryFrom3D(result)
        except Exception:
            pass
    return result, report
