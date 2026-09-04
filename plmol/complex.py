"""Complex API for protein-ligand workflows."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np

from .base import BaseMolecule, TempFileOwner
from .cache import LRUCache
from .constants import DEFAULT_DISTANCE_CUTOFF
from .errors import DependencyError, InputError
from .interaction import (
    PLInteractionFeaturizer,
    detect_metal_sites,
    encode_metal_features,
    extract_pocket,
)
from .io import load_ligand_input, load_protein_input
from .ligand.core import Ligand
from .rdkit_utils import apply_component_bond_orders
from .protein.core import Protein
from .specs import FEATURE_SPECS, is_all_mode, normalize_modes, normalize_requests

# Li, Be, Na, Mg, K, Ca, Mn, Fe, Co, Ni, Cu, Zn, Sr, Cd, Ba, Hg
_METAL_ATOMIC_NUMBERS = frozenset(
    {3, 4, 11, 12, 19, 20, 25, 26, 27, 28, 29, 30, 38, 48, 56, 80}
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency typing
    Chem = None


def _ligand_graph_to_interaction_index(mol: "Chem.Mol") -> "np.ndarray":
    """Ligand graph node -> the same atom's index in the interaction block.

    The two blocks of a complex result count the ligand's atoms differently.
    ``graph`` mode canonicalises the atom order, which is what lets the bond
    and fragment graphs line up with it; the interaction featurizer indexes
    the molecule as it was given, which for a file is the file's own order.
    Both are self-consistent and neither is wrong, but nothing said they
    disagree, so joining an interaction edge to a ligand node silently paired
    the wrong atoms.

    ``interaction["ligand_coords"][order]`` equals ``ligand["graph"]["coords"]``,
    and the same gather takes any per-atom interaction quantity into the
    graph's numbering. An entry is -1 where the graph node has no counterpart,
    which happens only when the molecule carries explicit hydrogens: the graph
    keeps them and the interaction block is heavy atoms alone.
    """
    from .rdkit_utils import canonical_atom_order

    order = canonical_atom_order(mol)
    heavy = {
        atom: slot
        for slot, atom in enumerate(
            i for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        )
    }
    return np.array([heavy.get(original, -1) for original in order], dtype=np.int64)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    return value


class MolecularComplex(TempFileOwner):
    """User-facing API for arbitrary multi-molecule operations."""

    def __init__(
        self,
        molecules: Optional[Dict[str, BaseMolecule]] = None,
        cache_size: int = 128,
    ) -> None:
        self.molecules: Dict[str, BaseMolecule] = molecules or {}
        self.cache_size = cache_size
        self._cache: LRUCache[Any, Any] = LRUCache(max_size=cache_size)
        self._protein_mol_cache: Optional["Chem.Mol"] = None
        self._ligand_mol_id: Optional[int] = self._current_ligand_mol_id()

    def _current_ligand_mol_id(self) -> Optional[int]:
        lig = self.molecules.get("ligand")
        if isinstance(lig, Ligand) and lig._rdmol is not None:
            return id(lig._rdmol)
        return None

    @classmethod
    def from_inputs(
        cls,
        *,
        protein: Union[str, Protein, None] = None,
        ligand: Union[str, "Chem.Mol", Ligand, None] = None,
        standardize: bool = True,
        keep_hydrogens: bool = False,
        add_hs: bool = False,
        cache_size: int = 128,
        **kwargs: BaseMolecule,
    ) -> "MolecularComplex":
        molecules: Dict[str, BaseMolecule] = {}
        if protein is not None:
            molecules["protein"] = load_protein_input(
                protein,
                standardize=standardize,
                keep_hydrogens=keep_hydrogens,
            )
        if ligand is not None:
            molecules["ligand"] = load_ligand_input(ligand, add_hs=add_hs)
        molecules.update(kwargs)
        return cls(molecules=molecules, cache_size=cache_size)

    @classmethod
    def from_files(
        cls,
        protein_pdb: str,
        ligand_path: str,
        *,
        standardize: bool = True,
        keep_hydrogens: bool = False,
        add_hs: bool = False,
        cache_size: int = 128,
    ) -> "MolecularComplex":
        return cls.from_inputs(
            protein=protein_pdb,
            ligand=ligand_path,
            standardize=standardize,
            keep_hydrogens=keep_hydrogens,
            add_hs=add_hs,
            cache_size=cache_size,
        )

    @classmethod
    def from_mmcif(
        cls,
        path: str,
        *,
        standardize: bool = True,
        keep_hydrogens: bool = False,
        extract_ligands: bool = True,
        ligand_resname: Optional[str] = None,
        ligand_chain: Optional[str] = None,
        add_hs: bool = False,
        cache_size: int = 128,
    ) -> "MolecularComplex":
        """
        Load a MolecularComplex from an mmCIF/PDBx file.

        Auto-detects all entities: protein chains → Protein, nucleic acid
        chains → NucleicAcid (stored as 'nucleic_acid'), and non-water
        ligand HETATM residues → Ligand.

        Requires gemmi (pip install 'plmol[mmcif]').
        """
        from .parsers.mmcif_parser import MMCIFParser
        from .nucleic_acid.core import NucleicAcid
        import tempfile

        parser = MMCIFParser(path, include_nucleic_acids=True)

        # Write a shared temp PDB for protein
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write(parser.to_pdb_string())
            tmp_path = f.name

        molecules: dict = {}

        protein_chains = parser.get_protein_chains()
        if protein_chains:
            protein = Protein(
                pdb_path=tmp_path,
                standardize=standardize,
                keep_hydrogens=keep_hydrogens,
            )
            protein.metadata["source"] = path
            molecules["protein"] = protein

        na_chains = parser.get_nucleic_acid_chains()
        if na_chains:
            na = NucleicAcid(pdb_path=tmp_path)
            na.metadata["source"] = path
            molecules["nucleic_acid"] = na

        if extract_ligands:
            if Chem is None:
                raise DependencyError("RDKit is required for mmCIF ligand extraction.")
            ligand_residues = parser.get_ligand_residues()
            if ligand_resname is not None:
                ligand_residues = [
                    r for r in ligand_residues
                    if r["res_name"].upper() == ligand_resname.upper()
                ]
            if ligand_chain is not None:
                ligand_residues = [
                    r for r in ligand_residues
                    if r["chain_id"] == ligand_chain
                ]
            atom_data = parser.get_atom_data()
            # An HETATM block says which atoms are where, not which bonds are
            # double or aromatic; read from coordinates alone a ligand comes
            # back entirely single-bonded. A PDBx/mmCIF entry carries the table
            # that says, so the ligand is corrected with the file's own answer.
            component_bonds = parser.get_component_bonds()
            loaded_ligands = []
            for ligand_info in ligand_residues:
                pdb_block = cls._ligand_pdb_block_from_atom_data(atom_data, ligand_info)
                if not pdb_block:
                    continue
                mol = Chem.MolFromPDBBlock(
                    pdb_block,
                    removeHs=not add_hs,
                    sanitize=True,
                    proximityBonding=True,
                )
                if mol is None:
                    mol = Chem.MolFromPDBBlock(
                        pdb_block,
                        removeHs=not add_hs,
                        sanitize=False,
                        proximityBonding=True,
                    )
                if mol is None or mol.GetNumAtoms() == 0:
                    continue
                table = component_bonds.get(ligand_info["res_name"], {})
                if table:
                    mol = apply_component_bond_orders(mol, table)
                ligand = Ligand(mol)
                ligand.metadata["source"] = path
                ligand.metadata["mmcif_ligand"] = ligand_info
                ligand.metadata["bond_orders_from_file"] = bool(table)
                loaded_ligands.append(ligand)

            for idx, ligand in enumerate(loaded_ligands):
                key = "ligand" if idx == 0 else f"ligand_{idx + 1}"
                molecules[key] = ligand

        obj = cls(molecules=molecules, cache_size=cache_size)
        obj._owned_temp_paths.append(tmp_path)
        return obj

    @staticmethod
    def _ligand_pdb_block_from_atom_data(
        atom_data: list[dict],
        ligand_info: dict,
    ) -> str:
        lines = []
        serial = 1
        for atom in atom_data:
            if atom.get("record_type") != "HETATM":
                continue
            if atom.get("chain_id") != ligand_info.get("chain_id"):
                continue
            if atom.get("res_name") != ligand_info.get("res_name"):
                continue
            if atom.get("res_num") != ligand_info.get("res_num"):
                continue

            x, y, z = atom["coords"]
            atom_name = str(atom.get("atom_name", ""))[:4]
            res_name = str(atom.get("res_name", "LIG"))[:3]
            chain_id = str(atom.get("chain_id", " "))[:1] or " "
            res_num = int(atom.get("res_num", 1) or 1)
            b_factor = float(atom.get("b_factor", 0.0) or 0.0)
            element = str(atom.get("element", "")).strip()[:2]
            lines.append(
                f"HETATM{serial:5d} {atom_name:<4s} {res_name:>3s} {chain_id:1s}"
                f"{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00{b_factor:6.2f}          {element:>2s}"
            )
            serial += 1

        if not lines:
            return ""
        return "\n".join(lines) + "\nEND\n"

    # ------------------------------------------------------------------
    # Backward-compatible property accessors
    # ------------------------------------------------------------------

    @property
    def ligand_obj(self) -> Optional[Ligand]:
        return self.molecules.get("ligand")  # type: ignore[return-value]

    @ligand_obj.setter
    def ligand_obj(self, value: Optional[Ligand]) -> None:
        if value is None:
            self.molecules.pop("ligand", None)
        else:
            self.molecules["ligand"] = value

    @property
    def protein_obj(self) -> Optional[Protein]:
        return self.molecules.get("protein")  # type: ignore[return-value]

    @protein_obj.setter
    def protein_obj(self, value: Optional[Protein]) -> None:
        if value is None:
            self.molecules.pop("protein", None)
        else:
            self.molecules["protein"] = value

    def cleanup(self) -> None:
        """Remove temporary files owned by this complex and its molecules."""
        for mol in self.molecules.values():
            cleanup = getattr(mol, "cleanup", None)
            if callable(cleanup):
                cleanup()
        super().cleanup()

    # ------------------------------------------------------------------
    # Cache freshness
    # ------------------------------------------------------------------

    def _check_ligand_freshness(self) -> None:
        """Clear cache if the underlying ligand mol object has changed."""
        current_id = self._current_ligand_mol_id()
        if current_id != self._ligand_mol_id:
            self._cache.clear()
            self._ligand_mol_id = current_id

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def set_ligand(self, ligand: Union[str, "Chem.Mol", Ligand], *, add_hs: bool = False) -> None:
        self.molecules["ligand"] = load_ligand_input(ligand, add_hs=add_hs)
        self._cache.clear()
        self._ligand_mol_id = self._current_ligand_mol_id()

    def set_protein(
        self,
        protein: Union[str, Protein],
        *,
        standardize: bool = True,
        keep_hydrogens: bool = False,
    ) -> None:
        self.molecules["protein"] = load_protein_input(
            protein,
            standardize=standardize,
            keep_hydrogens=keep_hydrogens,
        )
        self._protein_mol_cache = None
        self._cache.clear()

    # ------------------------------------------------------------------
    # Feature methods
    # ------------------------------------------------------------------

    def ligand(
        self,
        mode: Union[str, Iterable[str]] = "all",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        surface_kwargs: Optional[Dict[str, Any]] = None,
        voxel_kwargs: Optional[Dict[str, Any]] = None,
        fingerprint_kwargs: Optional[Dict[str, Any]] = None,
        generate_conformer: bool = False,
        add_hs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        ligand_obj = self.molecules.get("ligand")
        if ligand_obj is None:
            raise InputError("Ligand is not set in this complex.")

        self._check_ligand_freshness()
        mode = normalize_modes(FEATURE_SPECS["ligand"], mode)
        key = (
            "ligand",
            _freeze(mode),
            _freeze(graph_kwargs or {}),
            _freeze(surface_kwargs or {}),
            _freeze(voxel_kwargs or {}),
            _freeze(fingerprint_kwargs or {}),
            bool(generate_conformer),
            add_hs,
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = ligand_obj.featurize(
                mode=mode,
                graph_kwargs=graph_kwargs,
                surface_kwargs=surface_kwargs,
                voxel_kwargs=voxel_kwargs,
                fingerprint_kwargs=fingerprint_kwargs,
                generate_conformer=generate_conformer,
                add_hs=add_hs,
            )
            self._cache.set(key, cached)
        return cached

    def protein(
        self,
        mode: Union[str, Iterable[str]] = "all",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        surface_kwargs: Optional[Dict[str, Any]] = None,
        voxel_kwargs: Optional[Dict[str, Any]] = None,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        protein_obj = self.molecules.get("protein")
        if protein_obj is None:
            raise InputError("Protein is not set in this complex.")

        if is_all_mode(mode):
            mode = (
                list(FEATURE_SPECS["protein"].output_keys)
                if getattr(protein_obj, "_pdb_path", None) is not None
                else ["sequence"]
            )
        elif mode is None and getattr(protein_obj, "_pdb_path", None) is None:
            mode = ["sequence"]
        else:
            mode = normalize_modes(FEATURE_SPECS["protein"], mode)
        key = (
            "protein",
            _freeze(mode),
            _freeze(graph_kwargs or {}),
            _freeze(surface_kwargs or {}),
            _freeze(voxel_kwargs or {}),
            _freeze(backbone_kwargs or {}),
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = protein_obj.featurize(
                mode=mode,
                graph_kwargs=graph_kwargs,
                surface_kwargs=surface_kwargs,
                voxel_kwargs=voxel_kwargs,
                backbone_kwargs=backbone_kwargs,
            )
            self._cache.set(key, cached)
        return cached

    def nucleic_acid(
        self,
        mode: Union[str, Iterable[str]] = "all",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        atom_graph_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nucleic_acid_obj = self.molecules.get("nucleic_acid")
        if nucleic_acid_obj is None:
            raise InputError("Nucleic acid is not set in this complex.")

        if is_all_mode(mode):
            mode = (
                list(FEATURE_SPECS["nucleic_acid"].output_keys)
                if getattr(nucleic_acid_obj, "_pdb_path", None) is not None
                else ["sequence"]
            )
        elif mode is None and getattr(nucleic_acid_obj, "_pdb_path", None) is None:
            mode = ["sequence"]
        else:
            mode = normalize_modes(FEATURE_SPECS["nucleic_acid"], mode)
        key = (
            "nucleic_acid",
            _freeze(mode),
            _freeze(graph_kwargs or {}),
            _freeze(atom_graph_kwargs or {}),
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = nucleic_acid_obj.featurize(
                mode=mode,
                graph_kwargs=graph_kwargs,
                atom_graph_kwargs=atom_graph_kwargs,
            )
            self._cache.set(key, cached)
        return cached

    def interaction(
        self,
        distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
        pocket_cutoff: Optional[float] = None,
        knn_cutoff: Optional[int] = None,
        include_contacts: bool = False,
        contact_cutoff: Optional[float] = None,
        include_coords: bool = True,
        include_metal_sites: bool = True,
    ) -> Dict[str, Any]:
        ligand_obj = self.molecules.get("ligand")
        protein_obj = self.molecules.get("protein")
        if ligand_obj is None or protein_obj is None:
            raise InputError("Interaction features require both ligand and protein.")
        if Chem is None:
            raise DependencyError("RDKit is required for interaction featurization.")

        self._check_ligand_freshness()
        key = (
            "interaction",
            float(distance_cutoff),
            pocket_cutoff,
            knn_cutoff,
            bool(include_contacts),
            contact_cutoff,
            bool(include_coords),
            bool(include_metal_sites),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        ligand_mol = ligand_obj._rdmol
        if ligand_mol is None:
            raise InputError("Ligand has no RDKit molecule.")

        if pocket_cutoff is not None:
            if protein_obj._pdb_path is None:
                raise InputError("Protein PDB path is required for pocket interaction features.")
            pocket_list = extract_pocket(protein_obj._pdb_path, ligand_mol, distance_cutoff=float(pocket_cutoff))
            protein_mol = pocket_list[0].pocket_mol
        else:
            if self._protein_mol_cache is None:
                if protein_obj._pdb_path is None:
                    raise InputError("Protein PDB path is required for interaction features.")
                self._protein_mol_cache = Chem.MolFromPDBFile(protein_obj._pdb_path, removeHs=False)
            protein_mol = self._protein_mol_cache

        if protein_mol is None:
            raise InputError("Failed to build protein molecule for interaction featurization.")

        interaction = PLInteractionFeaturizer(
            protein_mol=protein_mol,
            ligand_mol=ligand_mol,
            distance_cutoff=distance_cutoff,
            knn_cutoff=knn_cutoff,
        )
        graph = interaction.get_interaction_graph(
            include_contacts=include_contacts,
            contact_cutoff=contact_cutoff,
            knn_cutoff=knn_cutoff,
        )
        graph["ligand_atom_order"] = _ligand_graph_to_interaction_index(ligand_mol)

        if include_coords:
            protein_coords, ligand_coords = interaction.get_heavy_atom_coords()
            graph["protein_coords"] = protein_coords
            graph["ligand_coords"] = ligand_coords

        if include_metal_sites:
            metal_sites = self._detect_protein_metal_sites(protein_mol)
            graph["metal_sites"] = metal_sites
            graph["metal_features"] = encode_metal_features(
                metal_sites,
                n_residues=self._count_protein_residues(protein_mol),
            )
        self._cache.set(key, graph)
        return graph

    @staticmethod
    def _count_protein_residues(protein_mol: "Chem.Mol") -> int:
        residues = set()
        for _idx in range(protein_mol.GetNumAtoms()):
            atom = protein_mol.GetAtomWithIdx(_idx)
            info = atom.GetPDBResidueInfo()
            if info is None:
                continue
            residues.add((info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode()))
        return len(residues)

    @staticmethod
    def _detect_protein_metal_sites(protein_mol: "Chem.Mol") -> list:
        if protein_mol.GetNumConformers() == 0:
            return []
        conf = protein_mol.GetConformer()
        coords = conf.GetPositions()

        # Look for metals first. Most structures have none, and building a
        # metadata dict per atom before finding that out was the whole cost.
        num_atoms = protein_mol.GetNumAtoms()
        get_atom = protein_mol.GetAtomWithIdx
        metal_indices = [
            i for i in range(num_atoms)
            if get_atom(i).GetAtomicNum() in _METAL_ATOMIC_NUMBERS
        ]
        if not metal_indices:
            return []

        metadata = []
        for i in range(num_atoms):
            atom = get_atom(i)
            info = atom.GetPDBResidueInfo()
            metadata.append(
                {
                    "atom_name": info.GetName().strip() if info is not None else atom.GetSymbol(),
                    "res_name": info.GetResidueName().strip() if info is not None else "",
                    "chain_id": info.GetChainId() if info is not None else "",
                    "element": atom.GetSymbol().upper(),
                }
            )

        import numpy as np

        return detect_metal_sites(
            atom_coords=np.asarray(coords, dtype=np.float32).reshape(-1, 3),
            atom_metadata=metadata,
            metal_indices=metal_indices,
        )

    def featurize(
        self,
        requests: Union[str, Iterable[str]] = "all",
        *,
        ligand_kwargs: Optional[Dict[str, Any]] = None,
        protein_kwargs: Optional[Dict[str, Any]] = None,
        nucleic_acid_kwargs: Optional[Dict[str, Any]] = None,
        interaction_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        is_all = (
            isinstance(requests, str) and requests.lower() == "all"
        ) or (
            not isinstance(requests, str)
            and any(str(r).lower() == "all" for r in requests)
        )
        reqs = normalize_requests(requests)
        out: Dict[str, Any] = {}

        if "ligand" in reqs and (not is_all or "ligand" in self.molecules):
            lk = dict(ligand_kwargs or {})
            lk.setdefault("mode", ["graph", "fingerprint", "smiles", "sequence"])
            out["ligand"] = self.ligand(**lk)
        if "protein" in reqs and (not is_all or "protein" in self.molecules):
            pk = dict(protein_kwargs or {})
            protein_obj = self.molecules.get("protein")
            default_protein_modes = (
                ["graph", "sequence"]
                if getattr(protein_obj, "_pdb_path", None) is not None
                else ["sequence"]
            )
            pk.setdefault("mode", default_protein_modes)
            out["protein"] = self.protein(**pk)
        if "nucleic_acid" in reqs and (not is_all or "nucleic_acid" in self.molecules):
            nk = dict(nucleic_acid_kwargs or {})
            na_obj = self.molecules.get("nucleic_acid")
            default_na_modes = (
                ["sequence", "graph"]
                if getattr(na_obj, "_pdb_path", None) is not None
                else ["sequence"]
            )
            nk.setdefault("mode", default_na_modes)
            out["nucleic_acid"] = self.nucleic_acid(**nk)
        if "interaction" in reqs and (
            not is_all
            or (
                "protein" in self.molecules
                and "ligand" in self.molecules
                and getattr(self.molecules.get("protein"), "_pdb_path", None) is not None
            )
        ):
            out["interaction"] = self.interaction(**(interaction_kwargs or {}))
        return out


# Backward compatibility: Complex is an alias for MolecularComplex
Complex = MolecularComplex
