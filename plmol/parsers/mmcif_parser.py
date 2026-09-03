"""
MMCIFParser — adapter for mmCIF/PDBx files via gemmi.

gemmi is an optional dependency. Import fails gracefully with DependencyError.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

from ..errors import DependencyError, InputError
from .base import StructureParser

try:
    import gemmi
    _GEMMI_AVAILABLE = True
except ImportError:  # pragma: no cover
    gemmi = None  # type: ignore
    _GEMMI_AVAILABLE = False


def _require_gemmi() -> None:
    if not _GEMMI_AVAILABLE:
        raise DependencyError(
            "gemmi is required for mmCIF support. "
            "Install it with: pip install 'plmol[mmcif]' or pip install gemmi"
        )


class MMCIFParser(StructureParser):
    """
    Parser for mmCIF/PDBx files using gemmi.

    Provides an interface compatible with the rest of plmol:
    - Entity/chain enumeration
    - Sequence extraction
    - Atom data in a dict format matching PDBParser conventions
    - Conversion to PDB format for downstream featurizers

    Args:
        mmcif_path: Path to the mmCIF file
        include_nucleic_acids: If True, include nucleic acid residues
    """

    def __init__(self, mmcif_path: str, include_nucleic_acids: bool = True):
        _require_gemmi()
        self.mmcif_path = os.path.abspath(mmcif_path)
        self.include_nucleic_acids = include_nucleic_acids

        if not os.path.exists(self.mmcif_path):
            raise FileNotFoundError(f"mmCIF file not found: {self.mmcif_path}")

        self._structure: "gemmi.Structure" = gemmi.read_structure(self.mmcif_path)
        self._atom_cache: Optional[List[Dict[str, Any]]] = None
        self._parsed_atoms_cache: Optional[List["ParsedAtom"]] = None
        self._all_parsed_atoms_cache: Optional[List["ParsedAtom"]] = None

    # ------------------------------------------------------------------
    # StructureParser interface
    # ------------------------------------------------------------------

    @property
    def file_path(self) -> str:
        return self.mmcif_path

    @property
    def protein_atoms(self) -> List["ParsedAtom"]:
        """Filtered atoms (protein/nucleic acid, no water/hydrogen/metal).

        The filter is ``pdb_parser.is_protein_atom``, the one the PDB parser
        applies. It used to be a shorter one written here that dropped only
        water and hydrogen, so the same structure read as mmCIF came back with
        its metals, its ligands and its terminal oxygens included and read as
        PDB did not.
        """
        if self._parsed_atoms_cache is not None:
            return self._parsed_atoms_cache
        from .pdb_parser import is_protein_atom

        atoms = [atom for atom in self.all_atoms
                 if is_protein_atom(atom, self.include_nucleic_acids)]
        self._parsed_atoms_cache = atoms
        return atoms

    @property
    def all_atoms(self) -> List["ParsedAtom"]:
        """All parsed atoms including HETATM, water, etc."""
        if self._all_parsed_atoms_cache is not None:
            return self._all_parsed_atoms_cache
        from ..protein.utils import ParsedAtom

        model = self._structure[0]
        atoms: list = []
        for chain in model:
            for res in chain:
                rname = res.name.strip()
                is_hetatm = res.het_flag == "H"
                record_type = "HETATM" if is_hetatm else "ATOM"
                for atom in res:
                    elem = atom.element.name if atom.element != gemmi.Element("X") else ""
                    try:
                        coords = (atom.pos.x, atom.pos.y, atom.pos.z)
                    except Exception:
                        continue
                    atoms.append(ParsedAtom(
                        record_type=record_type,
                        atom_name=atom.name.strip(),
                        res_name=rname,
                        res_num=res.seqid.num,
                        chain_id=chain.name,
                        coords=coords,
                        element=elem,
                        insertion_code=res.seqid.icode.strip() if res.seqid.icode else "",
                        b_factor=atom.b_iso,
                    ))
        self._all_parsed_atoms_cache = atoms
        return atoms

    def get_sequence_by_chain(self) -> Dict[str, str]:
        """Dict of chain_id -> one-letter sequence."""
        chains: Dict[str, str] = {}
        for chain_id in self.get_chains():
            seq = self.get_sequence(chain_id)
            if seq:
                chains[chain_id] = seq
        return chains

    # ------------------------------------------------------------------
    # Entity / chain info
    # ------------------------------------------------------------------

    def get_entities(self) -> Dict[str, str]:
        """
        Return {entity_id: entity_type} mapping.

        Entity types: 'polymer', 'non-polymer', 'water', 'branched'
        """
        result = {}
        for entity in self._structure.entities:
            etype = str(entity.entity_type).lower().replace("entitytype.", "").replace("_", "-")
            result[entity.name] = etype
        return result

    def get_chains(self) -> List[str]:
        """Return list of chain IDs from the first model."""
        if not self._structure:
            return []
        model = self._structure[0]
        return [chain.name for chain in model]

    def get_sequence(self, chain_id: Optional[str] = None) -> str:
        """
        Return the one-letter sequence for a chain.

        Uses gemmi's built-in one_letter_code for polymers.
        Falls back to 3→1 letter mapping for standard amino acids and nucleotides.
        """
        _require_gemmi()
        from ..constants import AMINO_ACID_3TO1, NUCLEOTIDE_3TO1

        model = self._structure[0]
        if chain_id is None:
            # Concatenate sequences from all chains
            return "".join(
                self.get_sequence(ch.name) for ch in model
            )
        try:
            chain = model[chain_id]
        except (KeyError, ValueError):
            return ""

        seq_chars = []
        for res in chain:
            rname = res.name.strip()
            if rname in AMINO_ACID_3TO1:
                seq_chars.append(AMINO_ACID_3TO1[rname])
            elif rname in NUCLEOTIDE_3TO1:
                seq_chars.append(NUCLEOTIDE_3TO1[rname])
            # skip water, ligands, unknowns
        return "".join(seq_chars)

    # ------------------------------------------------------------------
    # Atom data
    # ------------------------------------------------------------------

    def get_atom_data(self, chain_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return list of atom dicts compatible with PDBParser conventions.

        Each dict has:
            atom_name, res_name, res_num, chain_id, coords (tuple),
            element, insertion_code, b_factor, record_type
        """
        if self._atom_cache is not None:
            atoms = self._atom_cache
            if chain_id is not None:
                return [a for a in atoms if a["chain_id"] == chain_id]
            return atoms

        from ..constants import NUCLEIC_ACID_RESIDUES

        model = self._structure[0]
        atoms: List[Dict[str, Any]] = []

        for chain in model:
            for res in chain:
                rname = res.name.strip()

                # Skip nucleic acids unless opted in
                if not self.include_nucleic_acids and rname in NUCLEIC_ACID_RESIDUES:
                    continue

                # Determine record type: HETATM for non-polymer residues
                is_hetatm = res.het_flag == "H"
                record_type = "HETATM" if is_hetatm else "ATOM"

                for atom in res:
                    try:
                        coords = (atom.pos.x, atom.pos.y, atom.pos.z)
                    except Exception:
                        continue

                    atoms.append({
                        "atom_name": atom.name.strip(),
                        "res_name": rname,
                        "res_num": res.seqid.num,
                        "chain_id": chain.name,
                        "coords": coords,
                        "element": atom.element.name if atom.element != gemmi.Element("X") else "",
                        "insertion_code": res.seqid.icode.strip() if res.seqid.icode else "",
                        "b_factor": atom.b_iso,
                        "record_type": record_type,
                    })

        self._atom_cache = atoms

        if chain_id is not None:
            return [a for a in atoms if a["chain_id"] == chain_id]
        return atoms

    def get_atom_coords(self, chain_id: Optional[str] = None) -> np.ndarray:
        """Return (N, 3) float32 array of atom coordinates."""
        atoms = self.get_atom_data(chain_id=chain_id)
        if not atoms:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([a["coords"] for a in atoms], dtype=np.float32)

    # ------------------------------------------------------------------
    # PDB conversion
    # ------------------------------------------------------------------

    def to_pdb_string(self) -> str:
        """Convert structure to PDB format string via gemmi."""
        _require_gemmi()
        return self._structure.make_pdb_string()

    def to_pdb_file(self, output_path: str) -> str:
        """Write structure to PDB file and return path."""
        pdb_string = self.to_pdb_string()
        with open(output_path, "w") as f:
            f.write(pdb_string)
        return output_path

    def to_pdb_parser(self, include_nucleic_acids: Optional[bool] = None) -> "PDBParser":
        """
        Convert to a PDBParser-compatible object for downstream featurizers.

        Writes a temporary PDB file and returns a PDBParser instance.
        """
        from ..protein.utils import PDBParser

        if include_nucleic_acids is None:
            include_nucleic_acids = self.include_nucleic_acids

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            f.write(self.to_pdb_string())
            tmp_path = f.name

        try:
            parser = PDBParser(tmp_path, include_nucleic_acids=include_nucleic_acids)
        finally:
            os.unlink(tmp_path)

        return parser

    # ------------------------------------------------------------------
    # Convenience: entity classification
    # ------------------------------------------------------------------

    def get_protein_chains(self) -> List[str]:
        """Return chain IDs that contain standard amino acid residues."""
        from ..constants import AMINO_ACID_3TO1
        result = []
        model = self._structure[0]
        for chain in model:
            for res in chain:
                if res.name.strip() in AMINO_ACID_3TO1:
                    result.append(chain.name)
                    break
        return result

    def get_nucleic_acid_chains(self) -> List[str]:
        """Return chain IDs that contain nucleic acid residues."""
        from ..constants import NUCLEIC_ACID_RESIDUES
        result = []
        model = self._structure[0]
        for chain in model:
            for res in chain:
                if res.name.strip() in NUCLEIC_ACID_RESIDUES:
                    result.append(chain.name)
                    break
        return result

    def get_ligand_residues(self) -> List[Dict[str, str]]:
        """
        Return list of non-polymer, non-water HETATM residues.

        Each dict: {chain_id, res_name, res_num}
        """
        from ..constants import AMINO_ACID_3TO1, NUCLEIC_ACID_RESIDUES
        _SKIP = set(AMINO_ACID_3TO1.keys()) | NUCLEIC_ACID_RESIDUES | {"HOH", "WAT"}

        result = []
        model = self._structure[0]
        for chain in model:
            for res in chain:
                rname = res.name.strip()
                if res.het_flag == "H" and rname not in _SKIP:
                    result.append({
                        "chain_id": chain.name,
                        "res_name": rname,
                        "res_num": res.seqid.num,
                    })
        return result
