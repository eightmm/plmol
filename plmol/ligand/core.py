"""
Ligand Representation
"""
from ..base import BaseMolecule
from typing import Any, Dict, Iterable, Optional, Union
import numpy as np
import torch

try:
    from rdkit import Chem
except ImportError:
    Chem = None

from ..rdkit_utils import has_3d, ensure_3d_conformer
from .featurizer import LigandFeaturizer
from ..errors import InputError, DependencyError
from ..specs import LIGAND_SPEC, normalize_modes

class Ligand(BaseMolecule):
    """
    Represents a small molecule ligand with a consistent featurization API.

    Best practice:
        - Use lazy properties like `graph`, `surface`, and `fingerprint` for
          convenient access with default settings.
        - Use `featurize()` for explicit control over modes/kwargs.
        - `smiles` and `sequence` are always kept in sync.
    """
    
    def __init__(self, rdmol=None):
        super().__init__()
        self._rdmol = rdmol
        self._fingerprint = None
        self._fragment = None
        self._smiles: Optional[str] = None
        self._featurizer: Optional[LigandFeaturizer] = None
        self._featurizer_mol = None
        self._featurizer_variants: Dict[bool, LigandFeaturizer] = {}
        if rdmol:
            if Chem is None:
                raise DependencyError(
                    "RDKit is required to initialize Ligand from an RDKit molecule."
                )
            self._smiles = Chem.MolToSmiles(rdmol)
            self._sequence = self._smiles  # Sequence for ligand is SMILES
            # Extract coords if available
            if has_3d(rdmol):
                conf = rdmol.GetConformer()
                self._coords = conf.GetPositions()
                self._atoms = [a.GetSymbol() for a in rdmol.GetAtoms()]

    @classmethod
    def from_smiles(cls, smiles: str, add_hs: bool = False) -> "Ligand":
        """
        Create Ligand from SMILES string.
        
        Args:
            smiles: SMILES string
            add_hs: Whether to add explicit hydrogens (default: False)
        """
        if not Chem:
            raise DependencyError("RDKit is required to create a Ligand from SMILES.")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise InputError(f"Invalid SMILES string: '{smiles}'")
        if add_hs:
            mol = Chem.AddHs(mol)
        return cls(mol)

    @classmethod
    def from_sdf(cls, path: str) -> "Ligand":
        """Load from SDF file."""
        if not Chem:
            raise DependencyError("RDKit is required to load a Ligand from SDF.")
        suppl = Chem.SDMolSupplier(path)
        mol = next(suppl)
        if mol is None:
            raise InputError(f"Failed to parse molecule from SDF: '{path}'")
        return cls(mol)

    def generate_conformer(self):
        """Generate 3D conformer using canonical method if missing."""
        if self._rdmol:
            orig_num_atoms = self._rdmol.GetNumAtoms()
            mol_3d = ensure_3d_conformer(self._rdmol)
            if mol_3d is not None and has_3d(mol_3d):
                # ensure_3d_conformer adds Hs internally; only remove them
                # when the original mol doesn't have explicit Hs
                if mol_3d.GetNumAtoms() != orig_num_atoms:
                    mol_3d = Chem.RemoveHs(mol_3d)
                self._rdmol.RemoveAllConformers()
                self._rdmol.AddConformer(mol_3d.GetConformer(), assignId=True)
                self._coords = self._rdmol.GetConformer().GetPositions()
                self._invalidate_caches()

    @property
    def smiles(self) -> Optional[str]:
        """Return SMILES string (sequence alias for ligands)."""
        if self._smiles is None and self._rdmol is not None:
            if Chem is None:
                raise DependencyError(
                    "RDKit is required to generate SMILES from an RDKit molecule."
                )
            self._smiles = Chem.MolToSmiles(self._rdmol)
            self._sequence = self._smiles
        return self._smiles or self._sequence

    @smiles.setter
    def smiles(self, value: str) -> None:
        if Chem is None:
            raise DependencyError("RDKit is required to set SMILES.")
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise InputError(f"Invalid SMILES string: '{value}'")
        self._rdmol = mol
        self._smiles = Chem.MolToSmiles(mol)
        self._sequence = self._smiles
        self._coords = None
        self._atoms = []
        self._invalidate_caches()

    @property
    def sequence(self) -> Optional[str]:
        """Return sequence string (alias to SMILES for ligands)."""
        return self.smiles

    @sequence.setter
    def sequence(self, value: str) -> None:
        self.smiles = value

    def get_graph(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Return graph representation with explicit control over parameters.
        (e.g., distance_cutoff for separate bond/dist edges)
        """
        return self.featurize(mode="graph", graph_kwargs=kwargs).get("graph")

    @property
    def graph(self) -> Optional[Dict[str, Any]]:
        """Return graph representation, computing lazily if needed."""
        if self._graph is None:
            if self._rdmol is None:
                raise InputError(
                    "Ligand has no RDKit molecule. Initialize from SMILES/SDF before requesting graph features."
                )
            self.featurize(mode="graph")
        return self._graph

    @property
    def surface(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Return ligand surface representation, computing lazily if needed.

        Note:
            The surface includes ligand-specific atomic/chemical features
            mapped to surface points (e.g., LogP contributions, charge,
            H-bonding, aromaticity).
        """
        if self._surface is None:
            if self._rdmol is None:
                raise InputError(
                    "Ligand has no RDKit molecule. Initialize from SMILES/SDF before requesting surface features."
                )
            self.featurize(mode="surface")
        return self._surface

    @property
    def fingerprint(self):
        """Return fingerprint representation, computing lazily if needed."""
        if self._fingerprint is None:
            if self._rdmol is None:
                raise InputError(
                    "Ligand has no RDKit molecule. Initialize from SMILES/SDF before requesting fingerprints."
                )
            self.featurize(mode="fingerprint")
        return self._fingerprint

    @property
    def fragment(self) -> Optional[Dict[str, Any]]:
        """Return fragment representation, computing lazily if needed."""
        if self._fragment is None:
            if self._rdmol is None:
                raise InputError(
                    "Ligand has no RDKit molecule. Initialize from SMILES/SDF before requesting fragments."
                )
            self.featurize(mode="fragment")
        return self._fragment

    def _invalidate_caches(self) -> None:
        """Clear cached featurizers so they are rebuilt from current _rdmol."""
        self._featurizer = None
        self._featurizer_mol = None
        self._featurizer_variants.clear()
        self._graph = None
        self._fingerprint = None
        self._fragment = None
        self._surface = None

    def _get_featurizer(self, add_hs: Optional[bool] = None) -> LigandFeaturizer:
        if self._rdmol is None:
            raise InputError("Ligand has no RDKit molecule. Initialize from SMILES/SDF first.")
        if add_hs is None:
            if self._featurizer is None or self._featurizer_mol is not self._rdmol:
                self._featurizer = LigandFeaturizer(self._rdmol)
                self._featurizer_mol = self._rdmol
                self._featurizer_variants.clear()
            return self._featurizer

        if Chem is None:
            raise DependencyError("RDKit is required for hydrogen variant featurization.")

        key = bool(add_hs)
        cached = self._featurizer_variants.get(key)
        if cached is not None:
            return cached

        if add_hs:
            variant = Chem.AddHs(
                self._rdmol,
                addCoords=has_3d(self._rdmol),
            )
        else:
            variant = Chem.RemoveHs(self._rdmol)

        featurizer = LigandFeaturizer(variant)
        self._featurizer_variants[key] = featurizer
        return featurizer

    @staticmethod
    def _to_numpy_tree(value: Any) -> Any:
        """Recursively convert torch tensors in nested outputs to numpy arrays."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {k: Ligand._to_numpy_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Ligand._to_numpy_tree(v) for v in value]
        if isinstance(value, tuple):
            return tuple(Ligand._to_numpy_tree(v) for v in value)
        return value

    def featurize(
        self,
        mode: Union[str, Iterable[str]] = "all",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        surface_kwargs: Optional[Dict[str, Any]] = None,
        fingerprint_kwargs: Optional[Dict[str, Any]] = None,
        fragment_kwargs: Optional[Dict[str, Any]] = None,
        voxel_kwargs: Optional[Dict[str, Any]] = None,
        generate_conformer: bool = False,
        add_hs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Generate and cache ligand representations with standardized outputs.

        Args:
            mode: "all" or a single mode or list of modes.
                Supported: graph, bond_graph, fragment_graph, surface, voxel,
                fingerprint, descriptor, smiles, sequence, fragment, morgan
            graph_kwargs: Optional kwargs for graph featurization.
            surface_kwargs: Optional kwargs for surface extraction.
            fingerprint_kwargs: Optional kwargs for fingerprint extraction.
            fragment_kwargs: Optional kwargs for fragmentation.
            voxel_kwargs: Optional kwargs for voxel featurization.
            generate_conformer: Whether to generate a 3D conformer if missing
                for surface/voxel.
            add_hs: Optional override for hydrogen handling.
        """
        if self._rdmol is None:
            raise InputError("Ligand has no RDKit molecule. Initialize from SMILES/SDF first.")

        if (
            (isinstance(mode, str) and mode.lower() == "all")
            or (not isinstance(mode, str) and any(str(m).lower() == "all" for m in mode))
        ):
            modes = list(LIGAND_SPEC.default_modes)
        else:
            modes = normalize_modes(LIGAND_SPEC, mode)
        results: Dict[str, Any] = {}
        
        featurizer = self._get_featurizer(add_hs=add_hs)

        if "smiles" in modes or "sequence" in modes:
            if Chem is None:
                raise DependencyError(
                    "RDKit is required to generate SMILES. Install RDKit to use this feature."
                )
            self._smiles = Chem.MolToSmiles(self._rdmol)
            self._sequence = self._smiles
            results["smiles"] = self._smiles
            results["sequence"] = self._sequence

        if "graph" in modes:
            graph_kwargs = graph_kwargs or {}
            self._graph = self._to_numpy_tree(
                featurizer.get_graph(standardized=True, **graph_kwargs)
            )
            results["graph"] = self._graph

        if "bond_graph" in modes:
            graph_kwargs = graph_kwargs or {}
            results["bond_graph"] = self._to_numpy_tree(
                featurizer.get_bond_graph(**graph_kwargs)
            )

        if "fragment_graph" in modes:
            results["fragment_graph"] = self._to_numpy_tree(
                featurizer.get_fragment_graph(
                    graph_kwargs=graph_kwargs, **dict(fragment_kwargs or {})
                )
            )

        if "fingerprint" in modes:
            fingerprint_kwargs = fingerprint_kwargs or {}
            include_fps = fingerprint_kwargs.get("include_fps")
            # Return descriptor + selected fingerprint dictionary.
            if include_fps is None:
                features = featurizer.get_features()
            else:
                features = featurizer.get_features(include_fps=tuple(include_fps))
            self._fingerprint = self._to_numpy_tree(features)
            results["fingerprint"] = self._fingerprint

        if "descriptor" in modes:
            descriptor = featurizer.get_features(include_fps=())
            descriptor = self._to_numpy_tree(descriptor)
            descriptor["descriptor_names"] = list(featurizer.descriptor_names())
            results["descriptor"] = descriptor

        if "morgan" in modes:
            if "fingerprint" not in results:
                morgan_features = featurizer.get_features(include_fps=("ecfp4",))
                self._fingerprint = self._to_numpy_tree(morgan_features)
                results["fingerprint"] = self._fingerprint
            results["morgan"] = self._to_numpy_tree(featurizer.get_morgan_fingerprint())

        if "surface" in modes:
            surface_kwargs = surface_kwargs or {}
            surface = self._to_numpy_tree(featurizer.get_surface(
                generate_conformer=generate_conformer, **surface_kwargs
            ))
            if surface is not None:
                self._surface = surface
            results["surface"] = surface

        if "voxel" in modes:
            voxel_kwargs = voxel_kwargs or {}
            voxel = self._to_numpy_tree(featurizer.get_voxel(
                generate_conformer=generate_conformer, **voxel_kwargs
            ))
            results["voxel"] = voxel

        if "fragment" in modes:
            fragment_kwargs = dict(fragment_kwargs or {})
            self._fragment = featurizer.get_fragment(**fragment_kwargs)
            results["fragment"] = self._fragment

        return results
