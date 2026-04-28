"""
NucleicAcid — nucleic acid structure representation.
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Union
import numpy as np

from ..base import BaseMolecule
from ..constants import DNA_RESIDUES, RNA_RESIDUES
from ..specs import NUCLEIC_ACID_SPEC, is_all_mode, normalize_modes
from .featurizer import NucleicFeaturizer
from ..errors import InputError


def _detect_chain_type(res_names: list) -> str:
    """Auto-detect DNA or RNA from residue names. Returns 'DNA', 'RNA', or 'MIXED'."""
    has_dna = any(r in DNA_RESIDUES for r in res_names)
    has_rna = any(r in RNA_RESIDUES for r in res_names)
    if has_dna and not has_rna:
        return "DNA"
    if has_rna and not has_dna:
        return "RNA"
    if has_dna and has_rna:
        return "MIXED"
    return "UNKNOWN"


class NucleicAcid(BaseMolecule):
    """
    Represents a nucleic acid (DNA or RNA) structure.

    Can be initialized from a PDB file or a raw sequence string.
    Featurization is lazy — computed on first access.
    """

    def __init__(
        self,
        pdb_path: Optional[str] = None,
        chain_id: Optional[str] = None,
        na_type: Optional[str] = None,
    ):
        super().__init__()
        self._pdb_path = pdb_path
        self._chain_id = chain_id
        self._na_type = na_type  # "DNA", "RNA", or None (auto-detected)
        self._featurizer: Optional[NucleicFeaturizer] = None
        self._owned_temp_paths: list[str] = []

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pdb(cls, pdb_path: str, chain_id: Optional[str] = None) -> "NucleicAcid":
        """Load nucleic acid from a PDB file."""
        obj = cls(pdb_path=pdb_path, chain_id=chain_id)
        obj.metadata["source"] = pdb_path
        return obj

    @classmethod
    def from_mmcif(cls, path: str, chain_id: Optional[str] = None) -> "NucleicAcid":
        """
        Load nucleic acid from an mmCIF/PDBx file.

        Requires gemmi (pip install 'plmol[mmcif]').
        """
        from ..parsers.mmcif_parser import MMCIFParser
        import tempfile

        parser = MMCIFParser(path, include_nucleic_acids=True)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write(parser.to_pdb_string())
            tmp_path = f.name

        obj = cls(pdb_path=tmp_path, chain_id=chain_id)
        obj.metadata["source"] = path
        obj._owned_temp_paths.append(tmp_path)
        return obj

    def cleanup(self) -> None:
        """Remove temporary files created by constructors such as from_mmcif()."""
        for path in self._owned_temp_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
        self._owned_temp_paths.clear()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    @classmethod
    def from_structure(cls, path: str, chain_id: Optional[str] = None) -> "NucleicAcid":
        """Load nucleic acid from any supported structure file (PDB or mmCIF).

        Auto-detects format from file extension.
        """
        if path.endswith(('.cif', '.mmcif')):
            return cls.from_mmcif(path, chain_id=chain_id)
        return cls.from_pdb(path, chain_id=chain_id)

    @classmethod
    def from_sequence(cls, sequence: str, na_type: str = "DNA") -> "NucleicAcid":
        """
        Initialize from a nucleotide sequence string.

        Args:
            sequence: One-letter nucleotide codes (e.g. "ATGC" for DNA, "AUGC" for RNA)
            na_type: "DNA" or "RNA"
        """
        if na_type not in ("DNA", "RNA"):
            raise InputError(f"na_type must be 'DNA' or 'RNA', got: {na_type!r}")
        obj = cls(na_type=na_type)
        obj._sequence = sequence
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chain_type(self) -> str:
        """Return 'DNA', 'RNA', 'MIXED', or 'UNKNOWN'."""
        if self._na_type is not None:
            return self._na_type
        if self._pdb_path is not None:
            featurizer = self._get_featurizer()
            res_names = [r["res_name"] for r in featurizer._get_na_residues()]
            self._na_type = _detect_chain_type(res_names)
        return self._na_type or "UNKNOWN"

    @property
    def sequence(self) -> Optional[str]:
        """Nucleotide sequence as 1-letter codes."""
        if self._sequence is not None:
            return self._sequence
        if self._pdb_path is not None:
            self._sequence = self._get_featurizer().get_sequence()
        return self._sequence

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_featurizer(self) -> NucleicFeaturizer:
        if self._pdb_path is None:
            raise InputError(
                "NucleicAcid has no PDB path. Initialize from PDB first."
            )
        if self._featurizer is None:
            self._featurizer = NucleicFeaturizer(self._pdb_path, self._chain_id)
        return self._featurizer

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Optional[Dict[str, Any]]:
        """Residue-level graph features. Computed on first access."""
        if self._graph is None:
            result = self.featurize(mode="graph")
            self._graph = result.get("graph")
        return self._graph

    @property
    def coords(self) -> Optional[np.ndarray]:
        return self._coords

    # ------------------------------------------------------------------
    # Featurize
    # ------------------------------------------------------------------

    def featurize(
        self,
        mode: Union[str, Iterable[str]] = "all",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        atom_graph_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute features for the nucleic acid.

        Args:
            mode: One of "sequence", "graph", "backbone", "atom_graph", "all",
                  or a list of these values.
            graph_kwargs: Extra kwargs passed to get_graph().
            atom_graph_kwargs: Extra kwargs passed to get_atom_graph().

        Returns:
            Dict with requested features.
        """
        if is_all_mode(mode):
            modes: List[str] = (
                list(NUCLEIC_ACID_SPEC.output_keys)
                if self._pdb_path is not None
                else ["sequence"]
            )
        elif mode is None and self._pdb_path is None:
            modes = ["sequence"]
        else:
            modes = normalize_modes(NUCLEIC_ACID_SPEC, mode)

        graph_kwargs = graph_kwargs or {}
        atom_graph_kwargs = atom_graph_kwargs or {}

        result: Dict[str, Any] = {}

        if "sequence" in modes:
            if self._pdb_path is not None:
                result["sequence"] = self._get_featurizer().get_sequence_features()
            else:
                result["sequence"] = self._sequence_features_from_string()

        if "graph" in modes:
            result["graph"] = self._get_featurizer().get_graph(**graph_kwargs)

        if "backbone" in modes:
            result["backbone"] = self._get_featurizer().get_backbone()

        if "atom_graph" in modes:
            result["atom_graph"] = self._get_featurizer().get_atom_graph(**atom_graph_kwargs)

        return result

    def _sequence_features_from_string(self) -> Dict[str, Any]:
        """Build minimal sequence features from raw sequence string (no PDB needed)."""
        from ..constants import NUCLEOTIDE_TOKEN, NUCLEOTIDE_3TO1, NUM_NUCLEOTIDE_TYPES, PURINES, PYRIMIDINES

        seq = self._sequence or ""
        na_type = self._na_type or "DNA"

        # Map 1-letter codes back to 3-letter PDB names for DNA/RNA
        _1to3_dna = {"A": "DA", "T": "DT", "G": "DG", "C": "DC"}
        _1to3_rna = {"A": "A", "U": "U", "G": "G", "C": "C"}
        _map = _1to3_dna if na_type == "DNA" else _1to3_rna

        n = len(seq)
        tokens = np.zeros(n, dtype=np.int64)
        is_purine = np.zeros(n, dtype=np.float32)
        is_pyrimidine = np.zeros(n, dtype=np.float32)
        gc_count = 0
        res_names = []

        for i, letter in enumerate(seq.upper()):
            rname = _map.get(letter, "UNK_NT")
            res_names.append(rname)
            tokens[i] = NUCLEOTIDE_TOKEN.get(rname, NUCLEOTIDE_TOKEN["UNK_NT"])
            is_purine[i] = float(rname in PURINES)
            is_pyrimidine[i] = float(rname in PYRIMIDINES)
            if letter in ("G", "C"):
                gc_count += 1

        return {
            "tokens": tokens,
            "is_purine": is_purine,
            "is_pyrimidine": is_pyrimidine,
            "gc_content": float(gc_count / n) if n > 0 else 0.0,
            "res_names": res_names,
        }
