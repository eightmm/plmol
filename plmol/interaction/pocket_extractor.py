"""
Pocket Extractor Module.

Extracts binding pocket from protein based on ligand proximity.
Uses fast PDB line-based parsing with vectorized distance calculation.
"""

from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem
from ..constants import POCKET_MAX_ATOMS_PER_RESIDUE


# Maximum heavy atoms per residue (covers all standard amino acids)
MAX_ATOMS_PER_RESIDUE = POCKET_MAX_ATOMS_PER_RESIDUE


@dataclass
class PocketInfo:
    """Information about extracted pocket."""
    pocket_mol: Chem.Mol
    pocket_residues: List[Tuple[str, int, str]]  # (chain, resnum, resname)
    num_atoms: int
    num_residues: int
    distance_cutoff: float


def extract_pocket(
    protein_pdb: str,
    ligand: Union[str, Chem.Mol, List[Union[str, Chem.Mol]]],
    distance_cutoff: float = 6.0,
) -> List[PocketInfo]:
    """
    Extract binding pocket from protein based on ligand proximity.

    Simple API for pocket extraction. Protein is parsed once and reused
    for multiple ligands automatically.

    Args:
        protein_pdb: Path to protein PDB file
        ligand: Single ligand (path or Mol), list of ligands, or multi-mol SDF path
        distance_cutoff: Distance cutoff in Angstroms (default: 6.0)

    Returns:
        List of PocketInfo (one per ligand molecule).

    Examples:
        >>> # Single ligand
        >>> pockets = extract_pocket("protein.pdb", "ligand.sdf")
        >>> pocket = pockets[0]
        >>>
        >>> # Multi-molecule SDF
        >>> pockets = extract_pocket("protein.pdb", "multi_ligands.sdf")
        >>>
        >>> # List of ligands
        >>> pockets = extract_pocket("protein.pdb", ["lig1.sdf", "lig2.sdf"])
        >>>
        >>> # With RDKit Mol objects
        >>> pockets = extract_pocket("protein.pdb", ligand_mol, distance_cutoff=8.0)
    """
    extractor = PocketExtractor(protein_pdb)

    # Handle list of ligands
    if isinstance(ligand, list):
        results = []
        for lig in ligand:
            results.extend(_extract_for_ligand_input(extractor, lig, distance_cutoff))
        return results

    # Single input (could be multi-mol SDF)
    return _extract_for_ligand_input(extractor, ligand, distance_cutoff)


def _extract_for_ligand_input(
    extractor: "PocketExtractor",
    ligand: Union[str, Chem.Mol],
    distance_cutoff: float
) -> List[PocketInfo]:
    """Helper to handle single ligand input (may contain multiple mols)."""
    # If it's already a Mol object
    if isinstance(ligand, Chem.Mol):
        return [extractor.extract_for_ligand(ligand, distance_cutoff)]

    # If it's an SDF file, check for multiple molecules
    if ligand.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(ligand, removeHs=True)
        mols = [mol for mol in supplier if mol is not None]
        if not mols:
            raise ValueError(f"No valid molecules in {ligand}")
        return [extractor.extract_for_ligand(mol, distance_cutoff) for mol in mols]

    # Other file formats (single molecule)
    return [extractor.extract_for_ligand(ligand, distance_cutoff)]


@dataclass
class ParsedProtein:
    """Pre-parsed protein data for efficient pocket extraction."""
    residue_coords: np.ndarray  # [num_residue, MAX_ATOMS_PER_RESIDUE, 3]
    residue_lines: List[List[str]]  # PDB lines for each residue
    residue_keys: List[Tuple[str, int, str]]  # (chain, resnum, resname)
    num_residues: int
    pdb_path: str


class PocketExtractor:
    """
    Fast pocket extraction using PDB line-based parsing.

    Instead of parsing the entire protein with RDKit (slow for large proteins),
    this approach:
    1. Reads PDB file line by line, grouping by residue
    2. Builds coordinate tensor [num_residue, max_atoms, 3]
    3. Uses scipy.cdist for fast vectorized distance calculation
    4. Filters residues by distance cutoff
    5. Creates pocket PDB block from selected lines
    6. Calls MolFromPDBBlock only on the small pocket

    This is ~100-700x faster than the RDKit-based approach for large proteins.

    Examples:
        >>> # Single protein, single ligand
        >>> extractor = PocketExtractor("protein.pdb", "ligand.sdf", cutoff=6.0)
        >>> pocket_info = extractor.extract()
        >>>
        >>> # Single protein, multiple ligands (efficient reuse)
        >>> extractor = PocketExtractor.from_protein("protein.pdb")
        >>> for ligand_mol in ligands:
        ...     pocket_info = extractor.extract_for_ligand(ligand_mol, cutoff=6.0)
        >>>
        >>> # Batch processing with list of ligands
        >>> extractor = PocketExtractor.from_protein("protein.pdb")
        >>> pockets = extractor.extract_batch(ligand_mols, cutoff=6.0)
    """

    def __init__(
        self,
        protein_pdb_path: str,
        ligand: Optional[Union[str, Chem.Mol]] = None,
        distance_cutoff: float = 6.0,
    ):
        """
        Initialize PocketExtractor.

        Args:
            protein_pdb_path: Path to protein PDB file
            ligand: Path to ligand file (SDF/MOL2) or RDKit Mol object.
                   If None, use extract_for_ligand() later.
            distance_cutoff: Distance cutoff in Angstroms for pocket extraction
        """
        self.protein_pdb_path = protein_pdb_path
        self.distance_cutoff = distance_cutoff

        # Parse protein PDB (done once)
        self._parse_protein_pdb()

        # Load ligand if provided
        self._ligand_mol = None
        self._ligand_coords = None

        if ligand is not None:
            self.set_ligand(ligand)

    @classmethod
    def from_protein(cls, protein_pdb_path: str) -> "PocketExtractor":
        """
        Create PocketExtractor from protein only (no ligand).

        Use this when you want to extract pockets for multiple ligands
        from the same protein. The protein is parsed once and reused.

        Args:
            protein_pdb_path: Path to protein PDB file

        Returns:
            PocketExtractor instance ready for extract_for_ligand() calls

        Examples:
            >>> extractor = PocketExtractor.from_protein("protein.pdb")
            >>> for ligand in ligands:
            ...     pocket = extractor.extract_for_ligand(ligand, cutoff=6.0)
        """
        return cls(protein_pdb_path, ligand=None)

    @classmethod
    def from_files(
        cls,
        protein_pdb_path: str,
        ligand_path: str,
        distance_cutoff: float = 6.0,
    ) -> "PocketExtractor":
        """
        Create PocketExtractor from file paths.

        Args:
            protein_pdb_path: Path to protein PDB file
            ligand_path: Path to ligand file (SDF, MOL2, etc.)
            distance_cutoff: Distance cutoff in Angstroms
        """
        return cls(protein_pdb_path, ligand_path, distance_cutoff)

    def set_ligand(self, ligand: Union[str, Chem.Mol]) -> None:
        """
        Set or change the ligand molecule.

        Args:
            ligand: Path to ligand file or RDKit Mol object
        """
        if isinstance(ligand, str):
            self._ligand_mol = self._load_ligand(ligand)
        else:
            self._ligand_mol = ligand

        if self._ligand_mol is None:
            raise ValueError("Failed to load ligand molecule")
        if self._ligand_mol.GetNumConformers() == 0:
            raise ValueError("Ligand must have 3D coordinates")

        # Extract ligand coordinates (heavy atoms only)
        self._ligand_coords = self._get_ligand_coords(self._ligand_mol)

    def _load_ligand(self, ligand_path: str) -> Optional[Chem.Mol]:
        """Load ligand from file."""
        if ligand_path.endswith('.sdf'):
            supplier = Chem.SDMolSupplier(ligand_path, removeHs=True)
            return supplier[0] if supplier else None
        elif ligand_path.endswith('.mol2'):
            return Chem.MolFromMol2File(ligand_path, removeHs=True)
        elif ligand_path.endswith('.mol'):
            return Chem.MolFromMolFile(ligand_path, removeHs=True)
        elif ligand_path.endswith('.pdb'):
            return Chem.MolFromPDBFile(ligand_path, removeHs=True)
        else:
            # Try SDF as default
            supplier = Chem.SDMolSupplier(ligand_path, removeHs=True)
            return supplier[0] if supplier else None

    @staticmethod
    def _get_ligand_coords(ligand_mol: Chem.Mol) -> np.ndarray:
        """Extract heavy atom coordinates from ligand."""
        conf = ligand_mol.GetConformer(0)
        coords = []
        for i in range(ligand_mol.GetNumAtoms()):
            atom = ligand_mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() > 1:  # Heavy atoms only
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
        return np.array(coords, dtype=np.float32)

    def _parse_protein_pdb(self):
        """
        Parse protein PDB file into residue-based structure.

        Creates:
            - _residue_coords: [num_residue, MAX_ATOMS_PER_RESIDUE, 3] coordinate tensor
            - _residue_lines: List of PDB lines for each residue
            - _residue_keys: List of (chain, resnum, resname) tuples
        """
        residue_coords: Dict[Tuple, List[List[float]]] = {}
        residue_lines: Dict[Tuple, List[str]] = {}

        with open(self.protein_pdb_path, 'r') as f:
            for line in f:
                # ATOM lines only (standard amino acids, no HETATM)
                if not line.startswith('ATOM'):
                    continue

                # Skip hydrogens
                element = line[76:78].strip() if len(line) > 76 else ''
                if not element:
                    # Fallback: check atom name
                    atom_name = line[12:16].strip()
                    if atom_name.startswith('H') or atom_name in ('1H', '2H', '3H'):
                        continue
                elif element == 'H':
                    continue

                # Extract residue info
                chain = line[21] if len(line) > 21 else ' '
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue
                resname = line[17:20].strip()

                # Extract coordinates
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue

                key = (chain, resnum, resname)

                if key not in residue_coords:
                    residue_coords[key] = []
                    residue_lines[key] = []

                # Limit atoms per residue
                if len(residue_coords[key]) < MAX_ATOMS_PER_RESIDUE:
                    residue_coords[key].append([x, y, z])
                    residue_lines[key].append(line)

        # Sort residue keys for consistent ordering
        self._residue_keys = sorted(residue_coords.keys())
        num_residues = len(self._residue_keys)

        # Build coordinate tensor [num_residue, MAX_ATOMS, 3]
        self._residue_coords = np.full(
            (num_residues, MAX_ATOMS_PER_RESIDUE, 3),
            np.nan,
            dtype=np.float32
        )
        self._residue_lines: List[List[str]] = []

        for i, key in enumerate(self._residue_keys):
            coords = residue_coords[key]
            n_atoms = len(coords)
            self._residue_coords[i, :n_atoms, :] = coords
            self._residue_lines.append(residue_lines[key])

        self._num_residues = num_residues

    def _compute_pocket_mask(
        self,
        ligand_coords: np.ndarray,
        distance_cutoff: float
    ) -> np.ndarray:
        """
        Compute boolean mask for pocket residues using vectorized cdist.

        Args:
            ligand_coords: Ligand coordinates [num_atoms, 3]
            distance_cutoff: Distance cutoff in Angstroms

        Returns:
            Boolean array [num_residue] indicating pocket membership
        """
        num_res = self._num_residues

        # Flatten coordinates: [num_res * MAX_ATOMS, 3]
        coords_flat = self._residue_coords.reshape(-1, 3)

        # Find valid (non-nan) coordinates
        valid_mask = ~np.isnan(coords_flat[:, 0])

        # Compute distances only for valid coordinates
        distances = np.full(
            (coords_flat.shape[0], ligand_coords.shape[0]),
            np.inf,
            dtype=np.float32
        )
        if valid_mask.any():
            distances[valid_mask] = cdist(
                coords_flat[valid_mask],
                ligand_coords
            )

        # Reshape to [num_res, MAX_ATOMS, num_ligand_atoms]
        distances = distances.reshape(num_res, MAX_ATOMS_PER_RESIDUE, -1)

        # Get minimum distance per residue (across all atoms and ligand atoms)
        min_dist_per_res = np.nanmin(distances, axis=(1, 2))

        # Apply cutoff
        return min_dist_per_res < distance_cutoff

    def _extract_pocket_from_mask(
        self,
        pocket_mask: np.ndarray,
        distance_cutoff: float
    ) -> PocketInfo:
        """
        Extract pocket info from residue mask.

        Args:
            pocket_mask: Boolean array [num_residue]
            distance_cutoff: Distance cutoff used

        Returns:
            PocketInfo with extracted pocket
        """
        # Collect PDB lines for pocket residues
        pocket_lines: List[str] = []
        pocket_residues: List[Tuple[str, int, str]] = []

        for i, is_pocket in enumerate(pocket_mask):
            if is_pocket:
                pocket_lines.extend(self._residue_lines[i])
                pocket_residues.append(self._residue_keys[i])

        # Create PDB block
        pdb_block = ''.join(pocket_lines) + 'END\n'

        # Parse pocket with RDKit
        pocket_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True)

        num_atoms = pocket_mol.GetNumAtoms() if pocket_mol else 0

        return PocketInfo(
            pocket_mol=pocket_mol,
            pocket_residues=pocket_residues,
            num_atoms=num_atoms,
            num_residues=len(pocket_residues),
            distance_cutoff=distance_cutoff,
        )

    def extract(self, distance_cutoff: Optional[float] = None) -> PocketInfo:
        """
        Extract pocket based on distance cutoff.

        Args:
            distance_cutoff: Distance cutoff (uses default if not specified)

        Returns:
            PocketInfo with extracted pocket molecule and metadata
        """
        if self._ligand_coords is None:
            raise ValueError(
                "No ligand set. Use set_ligand() first or use extract_for_ligand()."
            )

        distance_cutoff = distance_cutoff if distance_cutoff is not None else self.distance_cutoff
        pocket_mask = self._compute_pocket_mask(self._ligand_coords, distance_cutoff)
        return self._extract_pocket_from_mask(pocket_mask, distance_cutoff)

    def extract_for_ligand(
        self,
        ligand: Union[str, Chem.Mol],
        distance_cutoff: float = 6.0
    ) -> PocketInfo:
        """
        Extract pocket for a specific ligand without modifying internal state.

        This method is ideal for processing multiple ligands with one protein.
        The protein is parsed once during __init__, and this method only
        computes distances and extracts pocket for each ligand.

        Args:
            ligand: Path to ligand file or RDKit Mol object
            distance_cutoff: Distance cutoff in Angstroms

        Returns:
            PocketInfo with extracted pocket

        Examples:
            >>> extractor = PocketExtractor.from_protein("protein.pdb")
            >>> pocket1 = extractor.extract_for_ligand("ligand1.sdf", distance_cutoff=6.0)
            >>> pocket2 = extractor.extract_for_ligand(ligand2_mol, distance_cutoff=8.0)
        """
        # Load ligand
        if isinstance(ligand, str):
            ligand_mol = self._load_ligand(ligand)
        else:
            ligand_mol = ligand

        if ligand_mol is None:
            raise ValueError("Failed to load ligand molecule")
        if ligand_mol.GetNumConformers() == 0:
            raise ValueError("Ligand must have 3D coordinates")

        # Get ligand coordinates
        ligand_coords = self._get_ligand_coords(ligand_mol)

        # Compute pocket mask and extract
        pocket_mask = self._compute_pocket_mask(ligand_coords, distance_cutoff)
        return self._extract_pocket_from_mask(pocket_mask, distance_cutoff)

    def extract_batch(
        self,
        ligands: List[Union[str, Chem.Mol]],
        distance_cutoff: float = 6.0
    ) -> List[PocketInfo]:
        """
        Extract pockets for multiple ligands efficiently.

        Args:
            ligands: List of ligand file paths or RDKit Mol objects
            distance_cutoff: Distance cutoff in Angstroms

        Returns:
            List of PocketInfo for each ligand

        Examples:
            >>> extractor = PocketExtractor.from_protein("protein.pdb")
            >>> ligand_mols = [mol1, mol2, mol3]
            >>> pockets = extractor.extract_batch(ligand_mols, distance_cutoff=6.0)
        """
        results = []
        for ligand in ligands:
            try:
                pocket_info = self.extract_for_ligand(ligand, distance_cutoff)
                results.append(pocket_info)
            except ValueError as e:
                # Skip invalid ligands but could also raise or log
                results.append(None)
        return results

    def get_pocket_pdb_block(self, distance_cutoff: Optional[float] = None) -> str:
        """
        Get PDB block string for the pocket.

        Args:
            distance_cutoff: Distance cutoff (uses default if not specified)

        Returns:
            PDB format string containing only pocket residues
        """
        if self._ligand_coords is None:
            raise ValueError("No ligand set. Use set_ligand() first.")

        distance_cutoff = distance_cutoff if distance_cutoff is not None else self.distance_cutoff
        pocket_mask = self._compute_pocket_mask(self._ligand_coords, distance_cutoff)
        pocket_lines: List[str] = []

        for i, is_pocket in enumerate(pocket_mask):
            if is_pocket:
                pocket_lines.extend(self._residue_lines[i])

        return ''.join(pocket_lines) + 'END\n'

    def save_pocket_pdb(self, output_path: str, distance_cutoff: Optional[float] = None) -> None:
        """
        Save pocket to PDB file.

        Args:
            output_path: Path to save pocket PDB
            distance_cutoff: Distance cutoff (uses default if not specified)
        """
        pdb_block = self.get_pocket_pdb_block(distance_cutoff)
        with open(output_path, 'w') as f:
            f.write(pdb_block)

    def get_pocket_residue_mask(self, distance_cutoff: Optional[float] = None) -> np.ndarray:
        """
        Get boolean mask indicating which residues are in the pocket.

        Args:
            distance_cutoff: Distance cutoff (uses default if not specified)

        Returns:
            Boolean array [num_residue]
        """
        if self._ligand_coords is None:
            raise ValueError("No ligand set. Use set_ligand() first.")

        distance_cutoff = distance_cutoff if distance_cutoff is not None else self.distance_cutoff
        return self._compute_pocket_mask(self._ligand_coords, distance_cutoff)

    def get_residue_distances(self, ligand: Optional[Union[str, Chem.Mol]] = None) -> np.ndarray:
        """
        Get minimum distance from each residue to any ligand atom.

        Args:
            ligand: Optional ligand (uses internal ligand if not specified)

        Returns:
            Array [num_residue] of minimum distances in Angstroms
        """
        if ligand is not None:
            if isinstance(ligand, str):
                ligand_mol = self._load_ligand(ligand)
            else:
                ligand_mol = ligand
            ligand_coords = self._get_ligand_coords(ligand_mol)
        elif self._ligand_coords is not None:
            ligand_coords = self._ligand_coords
        else:
            raise ValueError("No ligand provided or set.")

        num_res = self._num_residues
        coords_flat = self._residue_coords.reshape(-1, 3)
        valid_mask = ~np.isnan(coords_flat[:, 0])

        distances = np.full(
            (coords_flat.shape[0], ligand_coords.shape[0]),
            np.inf,
            dtype=np.float32
        )
        if valid_mask.any():
            distances[valid_mask] = cdist(
                coords_flat[valid_mask],
                ligand_coords
            )

        distances = distances.reshape(num_res, MAX_ATOMS_PER_RESIDUE, -1)
        return np.nanmin(distances, axis=(1, 2))

    def get_parsed_protein(self) -> ParsedProtein:
        """
        Get pre-parsed protein data for external use.

        Returns:
            ParsedProtein dataclass with all parsed data
        """
        return ParsedProtein(
            residue_coords=self._residue_coords.copy(),
            residue_lines=[lines.copy() for lines in self._residue_lines],
            residue_keys=self._residue_keys.copy(),
            num_residues=self._num_residues,
            pdb_path=self.protein_pdb_path,
        )

    @property
    def num_residues(self) -> int:
        """Total number of residues in protein."""
        return self._num_residues

    @property
    def residue_keys(self) -> List[Tuple[str, int, str]]:
        """List of (chain, resnum, resname) for all residues."""
        return self._residue_keys.copy()

    @property
    def ligand_mol(self) -> Optional[Chem.Mol]:
        """Ligand molecule (None if not set)."""
        return self._ligand_mol

    def __repr__(self) -> str:
        ligand_info = (
            f"ligand_atoms={len(self._ligand_coords)}"
            if self._ligand_coords is not None
            else "no_ligand"
        )
        return (
            f"PocketExtractor("
            f"residues={self._num_residues}, "
            f"{ligand_info}, "
            f"cutoff={self.distance_cutoff}Å)"
        )
