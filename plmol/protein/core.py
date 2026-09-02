"""
Protein Representation
"""
import os
import tempfile

from ..base import BaseMolecule
from typing import Any, Dict, Iterable, Optional, Union
import numpy as np
import torch

from .featurizer import ProteinFeaturizer
from ..errors import InputError, DependencyError
from ..specs import PROTEIN_SPEC, is_all_mode, normalize_modes
from ..constants import (
    DEFAULT_ATOM_GRAPH_DISTANCE_CUTOFF,
    DEFAULT_BACKBONE_KNN_NEIGHBORS,
    DEFAULT_RESIDUE_GRAPH_DISTANCE_CUTOFF,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency for pocket extraction
    Chem = None

class Protein(BaseMolecule):
    """
    Represents a protein structure.
    
    Can be initialized from PDB, mmCIF, or Sequence.
    """
    
    def __init__(
        self,
        pdb_id: Optional[str] = None,
        pdb_path: Optional[str] = None,
        standardize: bool = True,
        keep_hydrogens: bool = False,
    ):
        super().__init__()
        self.pdb_id = pdb_id
        self._pdb_path = pdb_path
        self._standardize = standardize
        self._keep_hydrogens = keep_hydrogens
        self._residues = []
        self._chains = []
        self._sequence_by_chain: Optional[Dict[str, str]] = None
        self._featurizer: Optional[ProteinFeaturizer] = None
        self._featurizer_path: Optional[str] = None
        self._graph_level: Optional[str] = None
        self._graph_distance_cutoff: Optional[float] = None

    @classmethod
    def from_pdb(
        cls,
        path: str,
        standardize: bool = True,
        keep_hydrogens: bool = False,
    ) -> "Protein":
        """Load protein from a PDB file."""
        obj = cls(
            pdb_path=path,
            standardize=standardize,
            keep_hydrogens=keep_hydrogens,
        )
        obj.metadata["source"] = path
        return obj

    @classmethod
    def from_mmcif(
        cls,
        path: str,
        chain_id: Optional[str] = None,
        standardize: bool = True,
        keep_hydrogens: bool = False,
    ) -> "Protein":
        """
        Load protein from an mmCIF/PDBx file.

        Requires the gemmi optional dependency (pip install 'plmol[mmcif]').

        Args:
            path: Path to .cif or .mmcif file
            chain_id: If given, restrict to this chain (written to PDB tmp file)
            standardize: Whether to standardize the resulting PDB
            keep_hydrogens: Whether to keep hydrogen atoms
        """
        from ..parsers.mmcif_parser import MMCIFParser
        import tempfile

        parser = MMCIFParser(path, include_nucleic_acids=False)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write(parser.to_pdb_string())
            tmp_path = f.name

        obj = cls(
            pdb_path=tmp_path,
            standardize=standardize,
            keep_hydrogens=keep_hydrogens,
        )
        obj.metadata["source"] = path
        obj.metadata["mmcif_chain_id"] = chain_id
        obj._owned_temp_paths.append(tmp_path)
        return obj

    @classmethod
    def from_structure(
        cls,
        path: str,
        chain_id: Optional[str] = None,
        standardize: bool = True,
        keep_hydrogens: bool = False,
    ) -> "Protein":
        """Load protein from any supported structure file (PDB or mmCIF).

        Auto-detects format from file extension.
        """
        if path.endswith(('.cif', '.mmcif')):
            return cls.from_mmcif(path, chain_id=chain_id, standardize=standardize, keep_hydrogens=keep_hydrogens)
        return cls.from_pdb(path, standardize=standardize, keep_hydrogens=keep_hydrogens)

    @classmethod
    def from_sequence(cls, sequence: str) -> "Protein":
        """Initialize from amino acid sequence (Foldseek/ESM style)."""
        obj = cls()
        obj._sequence = sequence
        return obj

    def cleanup(self) -> None:
        """Release temporary files, including the featurizer's standardized PDB.

        Dropping the featurizer lets it delete the standardized copy it wrote.
        A later featurize call simply rebuilds it, so this is safe to call at
        any point.
        """
        super().cleanup()
        self._featurizer = None
        self._featurizer_path = None

    def get_embedding(
        self,
        model: str = "esmc_600m",
        device: str = "auto",
        by_chain: bool = False,
    ) -> Dict[str, Any]:
        """Per-residue embeddings from a protein language model.

        Args:
            model: A name from ``plmol.list_protein_language_models()``, e.g.
                "esmc_600m", "esm3-open", "ankh-base", "esm2_t33_650m".
            device: "auto" (cuda when available), "cuda" or "cpu".
            by_chain: Embed each chain separately instead of the joined
                sequence. Chains are independent molecules, so this is the
                right choice for multi-chain structures.

        Returns:
            For a single sequence: ``embeddings`` (L, D), ``bos`` (D,),
            ``eos`` (D,), ``model``, ``dim``, ``sequence``. With
            ``by_chain=True``: ``by_chain`` mapping each chain id to that
            dictionary, plus ``model`` and ``dim``.

        Raises:
            InputError: If no sequence is available, or the model is unknown.
            DependencyError: If the model's backend package is not installed.
        """
        from .plm import embed_sequence, embed_sequences, plm_dim

        if by_chain:
            self._load_sequence()
            sequences = self._sequence_by_chain
            if not sequences:
                raise InputError("Protein has no per-chain sequence to embed.")
            return {
                "by_chain": embed_sequences(sequences, model=model, device=device),
                "model": model,
                "dim": plm_dim(model),
            }

        self._load_sequence()
        sequence = self._sequence
        if not sequence and self._sequence_by_chain:
            sequence = "".join(self._sequence_by_chain.values())
        if not sequence:
            raise InputError("Protein has no sequence to embed.")
        return embed_sequence(sequence, model=model, device=device)

    def _get_featurizer(self) -> ProteinFeaturizer:
        if self._pdb_path is None:
            raise InputError("Protein has no PDB path. Initialize from PDB first.")

        if (
            self._featurizer is None
            or self._featurizer_path != self._pdb_path
            or self._featurizer.standardize != self._standardize
            or self._featurizer.keep_hydrogens != self._keep_hydrogens
        ):
            self._featurizer = ProteinFeaturizer(
                self._pdb_path,
                standardize=self._standardize,
                keep_hydrogens=self._keep_hydrogens,
            )
            self._featurizer_path = self._pdb_path

        return self._featurizer

    def _load_sequence(self) -> None:
        if self._sequence is not None or self._sequence_by_chain is not None:
            return
        if self._pdb_path is None:
            raise InputError("Protein has no PDB path. Initialize from PDB first.")

        featurizer = self._get_featurizer()
        sequence_by_chain = featurizer.get_sequence_by_chain()
        self._sequence_by_chain = sequence_by_chain
        if len(sequence_by_chain) == 1:
            self._sequence = next(iter(sequence_by_chain.values()))

    @property
    def sequence(self) -> Optional[Union[str, Dict[str, str]]]:
        """Return sequence string (single chain) or dict of chain -> sequence."""
        self._load_sequence()
        if self._sequence is not None:
            return self._sequence
        return self._sequence_by_chain

    @property
    def graph(self) -> Optional[Dict[str, Any]]:
        """Return residue-level graph representation, computing lazily if needed."""
        if self._graph is None or self._graph_level != "residue":
            if self._pdb_path is None:
                raise InputError(
                    "Protein has no PDB path. Initialize from PDB before requesting graph features."
                )
            self.featurize(mode="graph", graph_kwargs={"level": "residue"})
        return self._graph

    @property
    def surface(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Return protein surface representation, computing lazily if needed.

        Note:
            The surface includes protein-tailored residue/patch and
            MaSIF-style geometric features (hydropathy, electrostatics,
            curvature) mapped to surface points.
        """
        if self._surface is None:
            if self._pdb_path is None:
                raise InputError(
                    "Protein has no PDB path. Initialize from PDB before requesting surface features."
                )
            self.featurize(mode="surface")
        return self._surface

    def _standardize_atom_graph(
        self, node: Dict[str, Any], edge: Dict[str, Any]
    ) -> Dict[str, Any]:
        edge_index = edge.get("edges")
        if isinstance(edge_index, tuple):
            edge_index = torch.stack(edge_index, dim=0)

        graph = {
            # Token features (int64, for nn.Embedding)
            "node_features": node.get("node_features"),
            "atom_tokens": node.get("atom_tokens"),
            "residue_token": node.get("residue_token"),
            "atom_element": node.get("atom_element"),
            # Coordinates
            "coords": node.get("coords"),
            # Node scalar features
            "sasa": node.get("sasa"),
            "relative_sasa": node.get("relative_sasa"),
            "burial_index": node.get("burial_index"),
            "is_polar_sasa": node.get("is_polar_sasa"),
            "is_backbone": node.get("is_backbone"),
            "formal_charge": node.get("formal_charge"),
            "is_hbond_donor": node.get("is_hbond_donor"),
            "is_hbond_acceptor": node.get("is_hbond_acceptor"),
            "secondary_structure": node.get("secondary_structure"),
            # Node index features
            "residue_number": node.get("residue_number"),
            "residue_count": node.get("residue_count"),
            "atom_to_residue": node.get("atom_to_residue"),
            "residue_atom_indices": node.get("residue_atom_indices"),
            # Edge features
            "edge_index": edge_index,
            "edge_distances": edge.get("edge_distances"),
            "same_residue": edge.get("same_residue"),
            "sequence_separation": edge.get("sequence_separation"),
            "unit_vector": edge.get("unit_vector"),
        }

        if "distance_cutoff" in edge:
            graph["distance_cutoff"] = edge["distance_cutoff"]
        if "knn_cutoff" in edge:
            graph["knn_cutoff"] = edge["knn_cutoff"]

        graph["metadata"] = {
            "atom_name": node.get("atom_name"),
            "chain_label": node.get("chain_label"),
        }

        graph["level"] = "atom"

        return graph

    def _standardize_residue_graph(
        self, node: Dict[str, Any], edge: Dict[str, Any], distance_cutoff: float,
        knn_cutoff: Optional[int] = None,
    ) -> Dict[str, Any]:
        edge_index = edge.get("edges")
        if isinstance(edge_index, tuple):
            edge_index = torch.stack(edge_index, dim=0)

        graph = {
            "node_features": node.get("node_scalar_features"),
            "node_vector_features": node.get("node_vector_features"),
            "edge_index": edge_index,
            "edge_features": edge.get("edge_scalar_features"),
            "edge_vector_features": edge.get("edge_vector_features"),
            "coords": node.get("coords"),
            "distance_cutoff": distance_cutoff,
            "knn_cutoff": knn_cutoff,
            "level": "residue",
        }

        return graph

    def get_graph(
        self,
        level: str = "residue",
        distance_cutoff: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return graph representation at residue or atom level."""
        graph_kwargs: Dict[str, Any] = {"level": level}
        if distance_cutoff is not None:
            graph_kwargs["distance_cutoff"] = distance_cutoff
        self.featurize(mode="graph", graph_kwargs=graph_kwargs)
        return self._graph

    def featurize(
        self,
        mode: Union[str, Iterable[str]] = "all",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        surface_kwargs: Optional[Dict[str, Any]] = None,
        voxel_kwargs: Optional[Dict[str, Any]] = None,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate and cache protein representations with standardized outputs.

        Args:
            mode: "all" or a single mode or list of modes.
                Supported: graph, atom_graph, surface, sequence, backbone,
                embedding.
                "atom_graph" is equivalent to graph with
                graph_kwargs={"level": "atom"} and is not part of "all".
            graph_kwargs: Optional kwargs for graph featurization.
                Use {"level": "residue"} (default) or {"level": "atom"}.
            surface_kwargs: Optional kwargs for surface extraction.
            voxel_kwargs: Optional kwargs for voxel featurization.
            backbone_kwargs: Optional kwargs for backbone featurization.
                Supports {"k_neighbors": int} (default: 30).
            embedding_kwargs: Optional kwargs for protein language model
                embeddings. Supports {"model": str, "device": str,
                "by_chain": bool}. See plmol.list_protein_language_models().

        Returns:
            Dict of requested representations with stable keys. The "graph"
            output includes Torch Geometric-friendly keys such as
            "node_features", "edge_index", and "edge_features".
        """
        if is_all_mode(mode):
            modes = list(PROTEIN_SPEC.output_keys) if self._pdb_path is not None else ["sequence"]
        elif mode is None and self._pdb_path is None:
            modes = ["sequence"]
        else:
            modes = normalize_modes(PROTEIN_SPEC, mode)
        results: Dict[str, Any] = {}

        if "sequence" in modes:
            self._load_sequence()
            results["sequence"] = (
                self._sequence if self._sequence is not None else self._sequence_by_chain
            )

        # "atom_graph" is the mode spelling of graph_kwargs={"level": "atom"},
        # matching NucleicAcid. Both spellings stay supported, and asking for
        # both yields a residue graph and an atom graph side by side.
        graph_requests = []
        if "graph" in modes:
            graph_requests.append(("graph", None))
        if "atom_graph" in modes:
            graph_requests.append(("atom_graph", "atom"))

        for result_key, level_override in graph_requests:
            graph_kwargs = dict(graph_kwargs or {})
            level = level_override or graph_kwargs.get("level", "residue")
            knn_cutoff = graph_kwargs.get("knn_cutoff")
            featurizer = self._get_featurizer()

            if level == "atom":
                distance_cutoff = graph_kwargs.get("distance_cutoff", DEFAULT_ATOM_GRAPH_DISTANCE_CUTOFF)
                node, edge = featurizer.get_atom_graph(
                    distance_cutoff=distance_cutoff, knn_cutoff=knn_cutoff
                )
                self._graph = self._standardize_atom_graph(node, edge)
                self._graph_level = "atom"
                self._graph_distance_cutoff = distance_cutoff
            else:
                distance_cutoff = graph_kwargs.get("distance_cutoff", DEFAULT_RESIDUE_GRAPH_DISTANCE_CUTOFF)
                node, edge = featurizer.get_features(
                    distance_cutoff=distance_cutoff, knn_cutoff=knn_cutoff
                )
                self._graph = self._standardize_residue_graph(
                    node, edge, distance_cutoff=distance_cutoff, knn_cutoff=knn_cutoff
                )
                self._graph_level = "residue"
                self._graph_distance_cutoff = distance_cutoff

            results[result_key] = self._graph

        if "embedding" in modes:
            results["embedding"] = self.get_embedding(**(embedding_kwargs or {}))

        if "surface" in modes:
            surface_kwargs = surface_kwargs or {}
            featurizer = self._get_featurizer()
            surface = featurizer.get_surface(**surface_kwargs)
            if surface is not None:
                self._surface = surface
            results["surface"] = surface

        if "voxel" in modes:
            voxel_kwargs = voxel_kwargs or {}
            featurizer = self._get_featurizer()
            results["voxel"] = featurizer.get_voxel(**voxel_kwargs)

        if "backbone" in modes:
            backbone_kwargs = backbone_kwargs or {}
            featurizer = self._get_featurizer()
            k = backbone_kwargs.get("k_neighbors", DEFAULT_BACKBONE_KNN_NEIGHBORS)
            results["backbone"] = featurizer.get_backbone(k_neighbors=k)

        return results

    def featurize_pocket(
        self,
        ligand: Any,
        distance_cutoff: float = 6.0,
        mode: Union[str, Iterable[str]] = "graph",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        surface_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Featurize only the binding pocket around the given ligand.

        Args:
            ligand: RDKit Mol, ligand path, or list supported by `extract_pocket`.
            distance_cutoff: Pocket extraction cutoff in Angstrom.
            mode: Same mode contract as `featurize` (default: "graph").
            graph_kwargs: Optional graph kwargs passed to pocket Protein object.
            surface_kwargs: Optional surface kwargs passed to pocket Protein object.
        """
        if self._pdb_path is None:
            raise InputError("Protein has no PDB path. Initialize from PDB first.")
        if Chem is None:
            raise DependencyError("RDKit is required for pocket featurization.")

        from ..interaction import extract_pocket

        pocket_list = extract_pocket(self._pdb_path, ligand, distance_cutoff=distance_cutoff)
        if not pocket_list:
            raise InputError("Pocket extraction returned no pocket molecules.")
        pocket_info = pocket_list[0]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(Chem.MolToPDBBlock(pocket_info.pocket_mol))
            pocket_pdb = f.name

        try:
            pocket_protein = Protein.from_pdb(
                pocket_pdb,
                standardize=self._standardize,
                keep_hydrogens=self._keep_hydrogens,
            )
            return pocket_protein.featurize(
                mode=mode,
                graph_kwargs=graph_kwargs,
                surface_kwargs=surface_kwargs,
            )
        finally:
            if os.path.exists(pocket_pdb):
                os.unlink(pocket_pdb)
