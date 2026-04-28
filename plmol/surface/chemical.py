"""Chemical features mapped from atoms to surface vertices."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Lipinski

from ..constants import (
    AROMATIC_RING_ATOMS,
    ATOMIC_MOLAR_REFRACTIVITY,
    CHARGED_RESIDUES,
    HBOND_ACCEPTOR_ATOMS_BY_RESIDUE,
    HBOND_DONOR_ATOMS_BY_RESIDUE,
    HIS_PARTIAL_CHARGE,
    HIS_POS_IONIZABLE_ATOMS,
    KD_SCALE,
    NEG_IONIZABLE_ATOMS,
    POS_IONIZABLE_ATOMS,
)
from .mapping import (
    SURFACE_KNN_ATOMS,
    _build_knn_weights,
    _knn_map_scalar,
    _normalize_to_range,
)

logger = logging.getLogger(__name__)


def compute_chemical_features(
    verts: np.ndarray,
    atom_positions: np.ndarray,
    mol,
    is_ligand: bool = True,
    charge_method: str = "gasteiger",
    knn_atoms: int = SURFACE_KNN_ATOMS,
    verbose: bool = False,
    _knn_data: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> dict:
    """Compute chemical features mapped from atoms to surface vertices.

    Features include electrostatic potential (Coulomb), hydrophobicity
    (Crippen LogP / Kyte-Doolittle), H-bond donors/acceptors, molar
    refractivity, aromaticity, and ionizability.

    Can be used independently for both ligand and protein surfaces.

    Args:
        verts: Surface points (N, 3)
        atom_positions: Atom positions (M, 3)
        mol: RDKit molecule or _SimpleMol for protein
        is_ligand: Whether this is a ligand (True) or protein (False)
        charge_method: "gasteiger" or "mmff94" (ligand only)
        knn_atoms: Number of nearest atoms per vertex
        verbose: Whether to print progress messages
        _knn_data: Pre-built (knn_idx, knn_weights, knn_dists) to avoid recomputation

    Returns:
        Dict with keys: 'electrostatic', 'hydrophobicity', 'hbd', 'hba',
        'molar_refractivity', 'aromaticity', 'pos_ionizable', 'neg_ionizable'
    """
    if is_ligand and charge_method not in {"gasteiger", "mmff94"}:
        from ..errors import InputError
        raise InputError(
            f"Unsupported charge_method: {charge_method!r}. "
            "Allowed: ['gasteiger', 'mmff94']"
        )

    if _knn_data is not None:
        knn_idx, knn_weights, knn_dists = _knn_data
    else:
        knn_idx, knn_weights, knn_dists = _build_knn_weights(verts, atom_positions, k=knn_atoms)

    knn_dists_clamped = np.maximum(knn_dists, 0.5)

    # --- Electrostatic potential ---
    if verbose:
        logger.debug("Computing electrostatic potential")
    if is_ligand:
        if charge_method == "mmff94":
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                charges = np.array(
                    [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())],
                    dtype=np.float32,
                )
            else:
                logger.warning("MMFF94 failed, falling back to Gasteiger")
                AllChem.ComputeGasteigerCharges(mol)
                charges = np.array(
                    [a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()],
                    dtype=np.float32,
                )
                np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        else:  # "gasteiger"
            AllChem.ComputeGasteigerCharges(mol)
            charges = np.array(
                [a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()],
                dtype=np.float32,
            )
            bad_mask = ~np.isfinite(charges)
            if bad_mask.any():
                logger.warning("%d atoms had NaN/Inf Gasteiger charges (zeroed)", bad_mask.sum())
            np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    else:
        charges = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for atom in mol.GetAtoms():
            res = atom.GetPDBResidueInfo()
            if res:
                res_name = res.GetResidueName().strip()
                atom_name = res.GetName().strip()
                if res_name in CHARGED_RESIDUES:
                    charges[atom.GetIdx()] = CHARGED_RESIDUES[res_name].get(atom_name, 0.0)

    knn_charges = charges[knn_idx]
    electrostatic_raw = np.sum(knn_charges / knn_dists_clamped, axis=1)
    electrostatic = _normalize_to_range(electrostatic_raw)

    # --- Chemical features ---
    if verbose:
        logger.debug("Computing chemical features")
    if is_ligand:
        contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
        logp_contribs = np.array([c[0] for c in contribs], dtype=np.float32)
        mr_contribs = np.array([c[1] for c in contribs], dtype=np.float32)

        hbd_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for match in mol.GetSubstructMatches(Lipinski.HDonorSmarts):
            hbd_atoms[match[0]] = 1.0

        hba_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for match in mol.GetSubstructMatches(Lipinski.HAcceptorSmarts):
            hba_atoms[match[0]] = 1.0

        aromaticity_atoms = np.array(
            [1.0 if a.GetIsAromatic() else 0.0 for a in mol.GetAtoms()],
            dtype=np.float32,
        )

        pos_smarts = Chem.MolFromSmarts("[+1,+2,$([NH2]-C(=N)N),$([NH]=C(N)N),$([nH]1ccnc1)]")
        pos_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for match in mol.GetSubstructMatches(pos_smarts):
            pos_atoms[match[0]] = 1.0

        neg_smarts = Chem.MolFromSmarts("[-1,-2,$([CX3](=O)[OH]),$([CX3](=O)[O-]),$([SX4](=O)(=O)[OH])]")
        neg_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for match in mol.GetSubstructMatches(neg_smarts):
            neg_atoms[match[0]] = 1.0
    else:
        logp_contribs = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        mr_contribs = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        hbd_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        hba_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        aromaticity_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        pos_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        neg_atoms = np.zeros(mol.GetNumAtoms(), dtype=np.float32)

        for atom in mol.GetAtoms():
            res = atom.GetPDBResidueInfo()
            if not res:
                continue
            idx = atom.GetIdx()
            res_name = res.GetResidueName().strip()
            atom_name = res.GetName().strip()
            atomic_num = atom.GetAtomicNum()

            # hydrophobicity (KD scale per residue)
            logp_contribs[idx] = KD_SCALE.get(res_name, 0.0)

            # molar refractivity (element-based)
            mr_contribs[idx] = ATOMIC_MOLAR_REFRACTIVITY.get(atomic_num, 0.0)

            # aromaticity
            if atom_name in AROMATIC_RING_ATOMS.get(res_name, frozenset()):
                aromaticity_atoms[idx] = 1.0

            # pos ionizable
            if atom_name in POS_IONIZABLE_ATOMS.get(res_name, frozenset()):
                pos_atoms[idx] = 1.0
            elif res_name == 'HIS' and atom_name in HIS_POS_IONIZABLE_ATOMS:
                pos_atoms[idx] = HIS_PARTIAL_CHARGE

            # neg ionizable
            if atom_name in NEG_IONIZABLE_ATOMS.get(res_name, frozenset()):
                neg_atoms[idx] = 1.0

            # HBD
            if atom_name == 'N' and res_name != 'PRO':
                hbd_atoms[idx] = 1.0
            elif atom_name in HBOND_DONOR_ATOMS_BY_RESIDUE.get(res_name, frozenset()):
                hbd_atoms[idx] = 1.0

            # HBA
            if atom_name == 'O':
                hba_atoms[idx] = 1.0
            elif atom_name in HBOND_ACCEPTOR_ATOMS_BY_RESIDUE.get(res_name, frozenset()):
                hba_atoms[idx] = 1.0

    hydrophobicity = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, logp_contribs))
    molar_refractivity = _normalize_to_range(_knn_map_scalar(knn_idx, knn_weights, mr_contribs))
    hbd = np.clip(_knn_map_scalar(knn_idx, knn_weights, hbd_atoms), 0, 1)
    hba = np.clip(_knn_map_scalar(knn_idx, knn_weights, hba_atoms), 0, 1)
    aromaticity = np.clip(_knn_map_scalar(knn_idx, knn_weights, aromaticity_atoms), 0, 1)
    pos_ionizable = np.clip(_knn_map_scalar(knn_idx, knn_weights, pos_atoms), 0, 1)
    neg_ionizable = np.clip(_knn_map_scalar(knn_idx, knn_weights, neg_atoms), 0, 1)

    return {
        'electrostatic': electrostatic,
        'hydrophobicity': hydrophobicity,
        'hbd': hbd,
        'hba': hba,
        'molar_refractivity': molar_refractivity,
        'aromaticity': aromaticity,
        'pos_ionizable': pos_ionizable,
        'neg_ionizable': neg_ionizable,
    }
