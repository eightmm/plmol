"""Complex API for protein-ligand workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Union

from .cache import LRUCache
from .constants import DEFAULT_DISTANCE_CUTOFF
from .errors import DependencyError, InputError
from .interaction import PLInteractionFeaturizer, extract_pocket
from .io import load_ligand_input, load_protein_input
from .ligand.core import Ligand
from .protein.core import Protein
from .specs import FEATURE_SPECS, normalize_modes, normalize_requests

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency typing
    Chem = None


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    return value


@dataclass
class Complex:
    """User-facing API for paired protein-ligand operations."""

    ligand_obj: Optional[Ligand] = None
    protein_obj: Optional[Protein] = None
    cache_size: int = 128

    def __post_init__(self) -> None:
        self._cache: LRUCache[Any, Any] = LRUCache(max_size=self.cache_size)
        self._protein_mol_cache: Optional["Chem.Mol"] = None

    @classmethod
    def from_inputs(
        cls,
        protein: Union[str, Protein],
        ligand: Union[str, "Chem.Mol", Ligand],
        *,
        standardize: bool = True,
        keep_hydrogens: bool = False,
        add_hs: bool = False,
        cache_size: int = 128,
    ) -> "Complex":
        protein_obj = load_protein_input(
            protein,
            standardize=standardize,
            keep_hydrogens=keep_hydrogens,
        )
        ligand_obj = load_ligand_input(ligand, add_hs=add_hs)
        return cls(
            ligand_obj=ligand_obj,
            protein_obj=protein_obj,
            cache_size=cache_size,
        )

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
    ) -> "Complex":
        return cls.from_inputs(
            protein=protein_pdb,
            ligand=ligand_path,
            standardize=standardize,
            keep_hydrogens=keep_hydrogens,
            add_hs=add_hs,
            cache_size=cache_size,
        )

    def set_ligand(self, ligand: Union[str, "Chem.Mol", Ligand], *, add_hs: bool = False) -> None:
        self.ligand_obj = load_ligand_input(ligand, add_hs=add_hs)
        self._cache.clear()

    def set_protein(
        self,
        protein: Union[str, Protein],
        *,
        standardize: bool = True,
        keep_hydrogens: bool = False,
    ) -> None:
        self.protein_obj = load_protein_input(
            protein,
            standardize=standardize,
            keep_hydrogens=keep_hydrogens,
        )
        self._protein_mol_cache = None
        self._cache.clear()

    def ligand(
        self,
        mode: Union[str, Iterable[str]] = "all",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        surface_kwargs: Optional[Dict[str, Any]] = None,
        fingerprint_kwargs: Optional[Dict[str, Any]] = None,
        generate_conformer: bool = False,
        add_hs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if self.ligand_obj is None:
            raise InputError("Ligand is not set in this complex.")

        mode = normalize_modes(FEATURE_SPECS["ligand"], mode)
        key = (
            "ligand",
            _freeze(mode),
            _freeze(graph_kwargs or {}),
            _freeze(surface_kwargs or {}),
            _freeze(fingerprint_kwargs or {}),
            bool(generate_conformer),
            add_hs,
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = self.ligand_obj.featurize(
                mode=mode,
                graph_kwargs=graph_kwargs,
                surface_kwargs=surface_kwargs,
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
    ) -> Dict[str, Any]:
        if self.protein_obj is None:
            raise InputError("Protein is not set in this complex.")

        mode = normalize_modes(FEATURE_SPECS["protein"], mode)
        key = ("protein", _freeze(mode), _freeze(graph_kwargs or {}), _freeze(surface_kwargs or {}))
        cached = self._cache.get(key)
        if cached is None:
            cached = self.protein_obj.featurize(
                mode=mode,
                graph_kwargs=graph_kwargs,
                surface_kwargs=surface_kwargs,
            )
            self._cache.set(key, cached)
        return cached

    def interaction(
        self,
        distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
        pocket_cutoff: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.ligand_obj is None or self.protein_obj is None:
            raise InputError("Interaction features require both ligand and protein.")
        if Chem is None:
            raise DependencyError("RDKit is required for interaction featurization.")

        key = ("interaction", float(distance_cutoff), pocket_cutoff)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        ligand_mol = self.ligand_obj._rdmol
        if ligand_mol is None:
            raise InputError("Ligand has no RDKit molecule.")

        if pocket_cutoff is not None:
            if self.protein_obj._pdb_path is None:
                raise InputError("Protein PDB path is required for pocket interaction features.")
            pocket_info = extract_pocket(self.protein_obj._pdb_path, ligand_mol, cutoff=float(pocket_cutoff))
            protein_mol = pocket_info.pocket_mol if not isinstance(pocket_info, list) else pocket_info[0].pocket_mol
        else:
            if self._protein_mol_cache is None:
                if self.protein_obj._pdb_path is None:
                    raise InputError("Protein PDB path is required for interaction features.")
                self._protein_mol_cache = Chem.MolFromPDBFile(self.protein_obj._pdb_path, removeHs=False)
            protein_mol = self._protein_mol_cache

        if protein_mol is None:
            raise InputError("Failed to build protein molecule for interaction featurization.")

        interaction = PLInteractionFeaturizer(
            protein_mol=protein_mol,
            ligand_mol=ligand_mol,
            distance_cutoff=distance_cutoff,
        )
        graph = interaction.get_interaction_graph()
        self._cache.set(key, graph)
        return graph

    def featurize(
        self,
        requests: Union[str, Iterable[str]] = "all",
        *,
        ligand_kwargs: Optional[Dict[str, Any]] = None,
        protein_kwargs: Optional[Dict[str, Any]] = None,
        interaction_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reqs = normalize_requests(requests)
        out: Dict[str, Any] = {}

        if "ligand" in reqs:
            lk = dict(ligand_kwargs or {})
            lk.setdefault("mode", ["graph", "fingerprint", "smiles", "sequence"])
            out["ligand"] = self.ligand(**lk)
        if "protein" in reqs:
            pk = dict(protein_kwargs or {})
            pk.setdefault("mode", ["graph", "sequence"])
            out["protein"] = self.protein(**pk)
        if "interaction" in reqs:
            out["interaction"] = self.interaction(**(interaction_kwargs or {}))
        return out
