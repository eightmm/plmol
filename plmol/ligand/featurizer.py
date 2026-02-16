"""
Ligand Featurizer

Provides a consistent, "call-friendly" interface for generating ligand
representations (graph, fingerprint, surface, smiles/sequence) from RDKit
ligands or SMILES strings.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None

from .descriptors import MoleculeFeaturizer
from .fragment import fragment_on_rotatable_bonds
from ..surface import build_ligand_surface
from ..voxel import build_ligand_voxel
from ..constants import (
    VDW_RADIUS,
    DEFAULT_VDW_RADIUS,
    SURFACE_DEFAULT_CURVATURE_SCALES,
    SURFACE_DEFAULT_KNN_ATOMS,
    SURFACE_DEFAULT_POINTS_PER_ATOM,
    SURFACE_DEFAULT_PROBE_RADIUS,
    VOXEL_DEFAULT_RESOLUTION,
    VOXEL_DEFAULT_BOX_SIZE,
    VOXEL_DEFAULT_PADDING,
    VOXEL_DEFAULT_SIGMA_SCALE,
    VOXEL_DEFAULT_CUTOFF_SIGMA,
)


class LigandFeaturizer:
    """
    Featurizer focused on ligand representations.

    Supports:
        - Graph representations for GNNs
        - Morgan fingerprints (ECFP)
        - Surface meshes (MaSIF-style point clouds)

    Best practice:
        - Use `featurize()` for a standardized output dictionary.
        - Use `get_graph()`/`get_morgan_fingerprint()`/`get_surface()` for
          individual representations, which also return standardized
          dictionaries.
    """

    def __init__(
        self,
        mol_or_smiles: Optional[Union[str, "Chem.Mol"]] = None,
        add_hs: bool = False,
        canonicalize: bool = True,
        custom_smarts: Optional[Dict[str, str]] = None,
    ):
        if Chem is None:
            raise ImportError("RDKit is required for LigandFeaturizer.")

        self._add_hs = add_hs
        self._canonicalize = canonicalize
        self._ligand_base_featurizer = MoleculeFeaturizer(
            mol_or_smiles,
            add_hs=add_hs,
            canonicalize=canonicalize,
            custom_smarts=custom_smarts,
        )
        self._mol = (
            self._ligand_base_featurizer.get_rdkit_mol()
            if mol_or_smiles is not None
            else None
        )
        self._surface_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}

    def set_molecule(self, mol_or_smiles: Union[str, "Chem.Mol"]) -> None:
        """Reset the ligand used for featurization."""
        self._ligand_base_featurizer = MoleculeFeaturizer(
            mol_or_smiles,
            add_hs=self._add_hs,
            canonicalize=self._canonicalize,
            custom_smarts=self._ligand_base_featurizer.custom_smarts,
        )
        self._mol = self._ligand_base_featurizer.get_rdkit_mol()
        self._surface_cache.clear()

    def featurize(
        self,
        mode: Union[str, Tuple[str, ...], list] = "all",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        surface_kwargs: Optional[Dict[str, Any]] = None,
        fingerprint_kwargs: Optional[Dict[str, Any]] = None,
        voxel_kwargs: Optional[Dict[str, Any]] = None,
        generate_conformer: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate ligand representations in a standardized dictionary.

        Args:
            mode: "all" or a single mode or list/tuple of modes.
                Supported: graph, surface, voxel, fingerprint, smiles, sequence
            graph_kwargs: Optional kwargs for graph featurization.
            surface_kwargs: Optional kwargs for surface extraction.
            fingerprint_kwargs: Optional kwargs for fingerprint extraction.
            voxel_kwargs: Optional kwargs for voxel featurization.
            generate_conformer: Whether to generate a 3D conformer if missing
                (surface/voxel only).

        Returns:
            Dict of requested representations with stable keys. The "graph"
            output includes dense "adjacency" and "bond_mask" tensors,
            plus "node_features", "distance_matrix", "distance_bounds",
            and optional "coords".
        """
        if isinstance(mode, str):
            modes = ["graph", "surface", "voxel", "fingerprint", "smiles", "sequence"] if mode == "all" else [mode]
        else:
            modes = list(mode)

        modes = [m.lower() for m in modes]
        results: Dict[str, Any] = {}

        if "smiles" in modes or "sequence" in modes:
            if Chem is None:
                raise ImportError(
                    "RDKit is required to generate SMILES. Install RDKit to use this feature."
                )
            mol = self._resolve_mol(None)
            if mol is None:
                raise ValueError("No ligand set for SMILES/sequence generation.")
            smiles = Chem.MolToSmiles(mol)
            results["smiles"] = smiles
            results["sequence"] = smiles

        if "graph" in modes:
            graph_kwargs = graph_kwargs or {}
            results["graph"] = self.get_graph(standardized=True, **graph_kwargs)

        if "fingerprint" in modes or "morgan" in modes:
            fingerprint_kwargs = fingerprint_kwargs or {}
            results["fingerprint"] = self.get_morgan_fingerprint(**fingerprint_kwargs)

        if "surface" in modes:
            surface_kwargs = surface_kwargs or {}
            results["surface"] = self.get_surface(
                generate_conformer=generate_conformer, **surface_kwargs
            )

        if "voxel" in modes:
            voxel_kw = dict(voxel_kwargs or {})
            voxel_kw.pop("generate_conformer", None)
            results["voxel"] = self.get_voxel(
                generate_conformer=generate_conformer, **voxel_kw
            )

        if "fragment" in modes:
            results["fragment"] = self.get_fragment()

        return results

    def _get_featurizer(
        self, mol_or_smiles: Optional[Union[str, "Chem.Mol"]]
    ) -> MoleculeFeaturizer:
        if mol_or_smiles is None:
            if self._mol is None:
                raise ValueError("No ligand set for featurization.")
            return self._ligand_base_featurizer
        return MoleculeFeaturizer(
            mol_or_smiles,
            add_hs=self._add_hs,
            canonicalize=self._canonicalize,
            custom_smarts=self._ligand_base_featurizer.custom_smarts,
        )

    def get_graph(
        self,
        mol_or_smiles: Optional[Union[str, "Chem.Mol"]] = None,
        add_hs: bool = False,
        distance_cutoff: Optional[float] = None,
        include_custom_smarts: bool = True,
        standardized: bool = False,
        knn_cutoff: Optional[int] = None,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any], Any]]:
        """
        Return standardized graph representation suitable for GNNs.

        Returns:
            Dict with stable keys:
                - node_features
                - adjacency
                - bond_mask
                - distance_matrix
                - distance_bounds
                - coords (if present)
        """
        featurizer = self._get_featurizer(mol_or_smiles)
        node, edge, adj = featurizer.featurize(
            mol_or_smiles=featurizer.get_rdkit_mol(),
            add_hs=add_hs,
            distance_cutoff=distance_cutoff,
            include_custom_smarts=include_custom_smarts,
            knn_cutoff=knn_cutoff,
        )
        if standardized:
            return self._standardize_graph(node, edge, adj)
        return node, edge, adj

    def get_features(
        self,
        mol_or_smiles: Optional[Union[str, "Chem.Mol"]] = None,
        include_fps: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        """Ligand feature dictionary (descriptors + fingerprints)."""
        if mol_or_smiles is None:
            return self._ligand_base_featurizer.get_features(include_fps=include_fps)
        return self._get_featurizer(mol_or_smiles).get_features(include_fps=include_fps)

    # Backward-compatible alias
    get_feature = get_features

    def get_morgan_fingerprint(
        self,
        mol_or_smiles: Optional[Union[str, "Chem.Mol"]] = None,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> Dict[str, Any]:
        """Return Morgan fingerprint (ECFP) in a standardized dictionary."""
        featurizer = self._get_featurizer(mol_or_smiles)
        fingerprint = featurizer.get_morgan_fingerprint(radius=radius, n_bits=n_bits)
        return {
            "fingerprint": fingerprint,
            "type": "morgan",
            "radius": radius,
            "n_bits": n_bits,
        }

    def get_surface(
        self,
        mol_or_smiles: Optional[Union[str, "Chem.Mol"]] = None,
        grid_density: float = 2.5,
        threshold: float = 0.5,
        sharpness: float = 1.5,
        generate_conformer: bool = False,
        include_features: bool = True,
        include_patches: bool = False,
        patch_radius: float = 6.0,
        max_patch_size: int = 128,
        max_patches: Optional[int] = None,
        patch_center_stride: int = 1,
        max_memory_gb: float = 1.0,
        device: Optional[str] = None,
        curvature_scales: tuple = SURFACE_DEFAULT_CURVATURE_SCALES,
        knn_atoms: int = SURFACE_DEFAULT_KNN_ATOMS,
        mode: str = "mesh",
        charge_method: str = "gasteiger",
        extra_atom_features: Optional[Dict[str, np.ndarray]] = None,
        n_points_per_atom: int = SURFACE_DEFAULT_POINTS_PER_ATOM,
        probe_radius: float = SURFACE_DEFAULT_PROBE_RADIUS,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Create a ligand surface from 3D coordinates.

        Args:
            mode: "mesh" for marching cubes mesh, "point_cloud" for SAS point cloud.
            charge_method: "gasteiger" or "mmff94" for partial charge computation.
            extra_atom_features: Custom per-atom features to map to vertices.
            n_points_per_atom: Points per atom for point cloud mode (default: 100).
            probe_radius: Solvent probe radius for point cloud mode (default: 1.4).

        Returns:
            Dict with "points", "normals" (and legacy "verts"). "faces" is
            included only in mesh mode. Plus ligand-tailored atomic features
            mapped to the surface when include_features is True.
        """
        if Chem is None:
            raise ImportError(
                "RDKit is required for surface extraction. Install RDKit to use this feature."
            )

        mol = self._resolve_mol(mol_or_smiles)
        if mol is None:
            raise ValueError("No ligand provided for surface extraction.")

        if mol.GetNumConformers() == 0:
            if not generate_conformer:
                raise ValueError("No 3D conformer found. Set generate_conformer=True.")
            self._generate_conformer(mol)

        conformer = mol.GetConformer()
        coords = conformer.GetPositions()

        cache_key = self._build_surface_cache_key(
            mol,
            coords,
            grid_density=grid_density,
            threshold=threshold,
            sharpness=sharpness,
            include_features=include_features,
            include_patches=include_patches,
            patch_radius=patch_radius,
            max_patch_size=max_patch_size,
            max_patches=max_patches,
            patch_center_stride=patch_center_stride,
        )
        if mol_or_smiles is None and cache_key in self._surface_cache:
            return self._surface_cache[cache_key]

        radii = np.array(
            [
                VDW_RADIUS.get(atom.GetAtomicNum(), DEFAULT_VDW_RADIUS)
                for atom in mol.GetAtoms()
            ],
            dtype=np.float32,
        )

        try:
            surface = build_ligand_surface(
                coords=coords,
                radii=radii,
                mol=mol,
                grid_density=grid_density,
                threshold=threshold,
                sharpness=sharpness,
                include_features=include_features,
                include_patches=include_patches,
                patch_radius=patch_radius,
                max_patch_size=max_patch_size,
                max_patches=max_patches,
                patch_center_stride=patch_center_stride,
                max_memory_gb=max_memory_gb,
                device=device,
                curvature_scales=curvature_scales,
                knn_atoms=knn_atoms,
                mode=mode,
                charge_method=charge_method,
                extra_atom_features=extra_atom_features,
                n_points_per_atom=n_points_per_atom,
                probe_radius=probe_radius,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Surface featurization requires optional dependencies (open3d, scikit-image, trimesh). "
                "Install them to enable surface features."
            ) from exc
        if surface is None and mol_or_smiles is None:
            self._surface_cache[cache_key] = None
            return None

        if mol_or_smiles is None:
            self._surface_cache[cache_key] = surface
        return surface

    def get_voxel(
        self,
        mol_or_smiles: Optional[Union[str, "Chem.Mol"]] = None,
        generate_conformer: bool = False,
        center: Optional[np.ndarray] = None,
        resolution: float = VOXEL_DEFAULT_RESOLUTION,
        box_size: Optional[int] = VOXEL_DEFAULT_BOX_SIZE,
        padding: float = VOXEL_DEFAULT_PADDING,
        sigma_scale: float = VOXEL_DEFAULT_SIGMA_SCALE,
        cutoff_sigma: float = VOXEL_DEFAULT_CUTOFF_SIGMA,
        charge_method: str = "gasteiger",
    ) -> Optional[Dict[str, Any]]:
        """Create a ligand voxel representation from 3D coordinates.

        Each atom's features are Gaussian-smeared onto a 3D grid,
        producing a multi-channel volume suitable for 3D CNNs.

        Channels (16): occupancy, atom type (6), charge, hydrophobicity,
        HBD, HBA, aromaticity, pos/neg ionizable, hybridization, ring.

        Args:
            mol_or_smiles: RDKit molecule or SMILES. None uses stored mol.
            generate_conformer: Whether to generate 3D conformer if missing.
            center: Grid center (3,). None = ligand centroid.
            resolution: Angstrom per voxel (default: 1.0).
            box_size: Grid dimension per axis (default: 24). None for adaptive.
            padding: Padding in Angstrom when box_size is None.
            sigma_scale: VdW radius multiplier for Gaussian sigma.
            cutoff_sigma: Gaussian cutoff in sigma units.
            charge_method: "gasteiger" or "mmff94".

        Returns:
            Dict with "voxel" (16, D, H, W), "channel_names", "grid_origin",
            "grid_shape", "resolution".
        """
        if Chem is None:
            raise ImportError("RDKit is required for voxel featurization.")

        mol = self._resolve_mol(mol_or_smiles)
        if mol is None:
            raise ValueError("No ligand provided for voxel featurization.")

        if mol.GetNumConformers() == 0:
            if not generate_conformer:
                raise ValueError("No 3D conformer found. Set generate_conformer=True.")
            self._generate_conformer(mol)

        return build_ligand_voxel(
            mol=mol,
            center=center,
            resolution=resolution,
            box_size=box_size,
            padding=padding,
            sigma_scale=sigma_scale,
            cutoff_sigma=cutoff_sigma,
            charge_method=charge_method,
        )

    def get_fragment(
        self,
        mol_or_smiles: Optional[Union[str, "Chem.Mol"]] = None,
        min_fragment_size: int = 1,
    ) -> Dict[str, Any]:
        """Return rotatable-bond fragmentation result."""
        mol = self._resolve_mol(mol_or_smiles)
        if mol is None:
            raise ValueError("No ligand set for fragmentation.")
        return fragment_on_rotatable_bonds(mol, min_fragment_size=min_fragment_size)

    def _resolve_mol(
        self, mol_or_smiles: Optional[Union[str, "Chem.Mol"]]
    ) -> Optional["Chem.Mol"]:
        if mol_or_smiles is None:
            return self._mol
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
            return mol
        return mol_or_smiles

    def _standardize_graph(
        self, node: Dict[str, Any], edge: Dict[str, Any], adj: Any
    ) -> Dict[str, Any]:
        if not isinstance(adj, torch.Tensor):
            adj = torch.as_tensor(adj)
        pair = edge.get("pair_features")
        if pair is not None:
            if not isinstance(pair, torch.Tensor):
                pair = torch.as_tensor(pair)
            # merge bond adjacency and complementary pair channels
            adjacency = torch.cat([adj, pair], dim=-1)
        else:
            adjacency = adj
        # first 4 adjacency channels are bond-type one-hot
        bond_mask = adjacency[..., :4].sum(dim=-1) > 0
        bond_mask.fill_diagonal_(False)
        distance_bounds = edge.get("distance_bounds")
        if distance_bounds is not None and not isinstance(distance_bounds, torch.Tensor):
            distance_bounds = torch.as_tensor(distance_bounds)
        graph = {
            "node_features": node.get("node_feats"),
            "adjacency": adjacency,
            "bond_mask": bond_mask,
            "distance_matrix": edge.get("distance_matrix"),
            "distance_bounds": distance_bounds,
        }
        coords = node.get("coords")
        if coords is None:
            n_atoms = int(graph["node_features"].shape[0]) if graph["node_features"] is not None else 0
            coords = torch.zeros((n_atoms, 3), dtype=torch.float32)
        graph["coords"] = coords
        return graph

    @staticmethod
    def adjacency_to_bond_edges(
        adjacency: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert dense adjacency [N, N, C] to sparse chemical bond edges.

        Bond existence is inferred from the first 4 channels (bond-type one-hot).
        Returns directed edges to keep compatibility with existing conventions.
        """
        if isinstance(adjacency, np.ndarray):
            adjacency = torch.from_numpy(adjacency)

        if adjacency.dim() != 3 or adjacency.size(-1) < 4:
            raise ValueError("adjacency must be [N, N, C] with C >= 4")

        bond_mask = adjacency[..., :4].sum(dim=-1) > 0
        bond_mask.fill_diagonal_(False)
        src, dst = torch.where(bond_mask)
        edge_index = torch.stack([src, dst], dim=0)
        edge_features = adjacency[src, dst]
        return edge_index, edge_features

    def _generate_conformer(self, mol: "Chem.Mol") -> None:
        if AllChem is None:
            raise ImportError("RDKit AllChem is required to generate conformers.")
        mol_3d = MoleculeFeaturizer._ensure_3d_conformer(mol)
        if mol_3d is not None and mol_3d.GetNumConformers() > 0:
            mol.RemoveAllConformers()
            mol.AddConformer(mol_3d.GetConformer(), assignId=True)

    def _build_surface_cache_key(
        self,
        mol: "Chem.Mol",
        coords: np.ndarray,
        grid_density: float,
        threshold: float,
        sharpness: float,
        include_features: bool,
        include_patches: bool,
        patch_radius: float,
        max_patch_size: int,
        max_patches: Optional[int],
        patch_center_stride: int,
    ) -> str:
        coords32 = np.asarray(coords, dtype=np.float32)
        coord_sig = (
            coords32.shape[0],
            round(float(coords32.mean()), 4),
            round(float(coords32.std()), 4),
        )
        return (
            f"{id(mol)}|{coord_sig}|{round(grid_density,3)}|{round(threshold,3)}|"
            f"{round(sharpness,3)}|{int(include_features)}|{int(include_patches)}|"
            f"{round(patch_radius,3)}|{int(max_patch_size)}|"
            f"{-1 if max_patches is None else int(max_patches)}|{int(patch_center_stride)}"
        )

    @property
    def num_atoms(self) -> int:
        return self._ligand_base_featurizer.num_atoms

    @property
    def num_bonds(self) -> int:
        return self._ligand_base_featurizer.num_bonds

    @property
    def num_rings(self) -> int:
        return self._ligand_base_featurizer.num_rings

    @property
    def has_3d(self) -> bool:
        return self._ligand_base_featurizer.has_3d

    @property
    def input_smiles(self) -> Optional[str]:
        return self._ligand_base_featurizer.input_smiles
