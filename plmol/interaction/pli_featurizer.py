"""
Protein-Ligand Interaction (PLI) Edge Featurizer.

This module provides edge-level feature extraction for protein-ligand interactions,
detecting various interaction types and generating features for GNN models.

Uses Heavy Atom Only approach:
- Nodes: Only heavy atoms (C, N, O, S, P, halogens, etc.)
- Edges: Interactions between heavy atoms
- H information: Encoded in edge features (angles calculated with implicit/explicit H)
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import math
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ..constants import (
    PHARMACOPHORE_SMARTS,
    INTERACTION_TYPES,
    INTERACTION_TYPE_IDX,
    NUM_INTERACTION_TYPES,
    IDEAL_DISTANCES,
    PHARMACOPHORE_IDX,
    NUM_PHARMACOPHORE_TYPES,
    # Element types (heavy atoms only)
    HEAVY_ELEMENT_TYPES,
    NUM_HEAVY_ELEMENT_TYPES,
    # Hybridization types
    HYBRIDIZATION_TYPES,
    NUM_HYBRIDIZATION_TYPES,
    # Residue types
    RESIDUE_TYPES,
    NUM_RESIDUE_TYPES,
)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Element types alias for this module
ELEMENT_TYPES = HEAVY_ELEMENT_TYPES
NUM_ELEMENT_TYPES = NUM_HEAVY_ELEMENT_TYPES


@dataclass
class Interaction:
    """Data class representing a single protein-ligand interaction."""
    protein_atom_idx: int      # Heavy atom index in protein
    ligand_atom_idx: int       # Heavy atom index in ligand
    interaction_type: str
    distance: float
    angle: Optional[float] = None       # Ring angle for pi-stacking
    dha_angle: Optional[float] = None   # D-H-A angle for H-bonds
    cxa_angle: Optional[float] = None   # C-X-A angle for halogen bonds
    has_valid_angle: bool = False       # Whether angle was successfully calculated
    strength: float = 1.0
    metadata: Optional[Dict] = field(default_factory=dict)


class PLInteractionFeaturizer:
    """
    Protein-Ligand Interaction Edge Featurizer (Heavy Atom Only).

    Uses heavy atoms only for graph nodes, with hydrogen information
    encoded in edge features through angle calculations.

    Uses chemically accurate PHARMACOPHORE_SMARTS patterns:
        - h_acceptor: H-bond acceptors
        - h_donor: H-bond donors
        - hydrophobic: Hydrophobic atoms
        - positive: Positively charged/ionizable
        - negative: Negatively charged/ionizable
        - aromatic: Aromatic atoms
        - halogen: Halogen bond donors

    Workflow:
        1. Input molecules (with or without H)
        2. Add hydrogens with 3D coordinates if missing
        3. Detect interactions between heavy atoms
        4. Calculate angles using H positions
        5. Store angles in edge features
        6. Output graph uses heavy atom indices only

    Features include (Total: 73 dims):
        - Interaction type one-hot (7 dims)
        - Distance and angle (4 dims): distance, angle, has_valid_angle, angle_type
        - Element types one-hot (20 dims): protein (10) + ligand (10)
        - Hybridization one-hot (12 dims): protein (6) + ligand (6)
        - Formal charges (2 dims)
        - Aromatic flags (2 dims)
        - Ring membership + degree (4 dims)
        - Residue type one-hot (21 dims)
        - Is backbone flag (1 dim)

    Examples:
        >>> featurizer = PLInteractionFeaturizer(protein_mol, ligand_mol)
        >>> edges, edge_features = featurizer.get_interaction_edges()
        >>> # edges: heavy atom indices only
        >>> # edge_features: includes H-based angle information
    """

    def __init__(
        self,
        protein_mol: Chem.Mol,
        ligand_mol: Chem.Mol,
        distance_cutoff: float = 4.5,
    ):
        """
        Initialize the PLI featurizer.

        Args:
            protein_mol: RDKit mol object for protein (must have 3D coordinates)
            ligand_mol: RDKit mol object for ligand (must have 3D coordinates)
            distance_cutoff: Maximum distance for interaction detection (Angstrom)

        Raises:
            ValueError: If molecules lack 3D coordinates
        """
        self.distance_cutoff = distance_cutoff
        self._cache: Dict[str, Any] = {}

        # Store original molecules and prepare with hydrogens
        self._protein_with_h = self._prepare_mol_with_hydrogens(protein_mol)
        self._ligand_with_h = self._prepare_mol_with_hydrogens(ligand_mol)

        # Validate 3D coordinates
        if self._protein_with_h.GetNumConformers() == 0:
            raise ValueError("Protein molecule must have 3D coordinates")
        if self._ligand_with_h.GetNumConformers() == 0:
            raise ValueError("Ligand molecule must have 3D coordinates")

        # Build heavy atom index mappings
        self._build_heavy_atom_mappings()

        # Get coordinates for molecules with H (for angle calculations)
        self._protein_coords_with_h = self._get_coords(self._protein_with_h)
        self._ligand_coords_with_h = self._get_coords(self._ligand_with_h)

        # Get heavy atom only coordinates
        self._protein_coords = self._protein_coords_with_h[self._protein_heavy_indices]
        self._ligand_coords = self._ligand_coords_with_h[self._ligand_heavy_indices]

        # Compute distance matrix (heavy atoms only)
        self._distance_matrix = self._compute_distance_matrix()

        # Compile SMARTS patterns
        self._compile_patterns()

        # Detect pharmacophore features (on molecules with H, then map to heavy)
        self._detect_pharmacophores()

        # Extract atom chemical features (heavy atoms only)
        self._extract_atom_features()

        # Extract residue information for protein
        self._extract_residue_info()

    def _prepare_mol_with_hydrogens(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Prepare molecule with explicit hydrogens and 3D coordinates.

        If molecule lacks hydrogens, adds them with computed coordinates.
        """
        if mol is None:
            raise ValueError("Molecule cannot be None")

        mol = Chem.Mol(mol)  # Copy to avoid modifying original

        # Check if already has hydrogens
        has_hydrogens = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())

        if not has_hydrogens:
            # Add hydrogens
            has_3d = mol.GetNumConformers() > 0
            if has_3d:
                mol = Chem.AddHs(mol, addCoords=True)
            else:
                mol = Chem.AddHs(mol)
                # Generate 3D coordinates if none exist
                AllChem.EmbedMolecule(mol, randomSeed=42)

        return mol

    def _build_heavy_atom_mappings(self):
        """Build mappings between heavy atom indices and full molecule indices."""
        # Protein
        self._protein_heavy_indices = []  # heavy_idx -> full_idx
        self._protein_heavy_to_full = {}  # heavy_idx -> full_idx
        self._protein_full_to_heavy = {}  # full_idx -> heavy_idx

        heavy_idx = 0
        for atom in self._protein_with_h.GetAtoms():
            full_idx = atom.GetIdx()
            if atom.GetAtomicNum() > 1:  # Not hydrogen
                self._protein_heavy_indices.append(full_idx)
                self._protein_heavy_to_full[heavy_idx] = full_idx
                self._protein_full_to_heavy[full_idx] = heavy_idx
                heavy_idx += 1

        self._protein_heavy_indices = np.array(self._protein_heavy_indices)
        self.num_protein_atoms = len(self._protein_heavy_indices)

        # Ligand
        self._ligand_heavy_indices = []
        self._ligand_heavy_to_full = {}
        self._ligand_full_to_heavy = {}

        heavy_idx = 0
        for atom in self._ligand_with_h.GetAtoms():
            full_idx = atom.GetIdx()
            if atom.GetAtomicNum() > 1:
                self._ligand_heavy_indices.append(full_idx)
                self._ligand_heavy_to_full[heavy_idx] = full_idx
                self._ligand_full_to_heavy[full_idx] = heavy_idx
                heavy_idx += 1

        self._ligand_heavy_indices = np.array(self._ligand_heavy_indices)
        self.num_ligand_atoms = len(self._ligand_heavy_indices)

        # Build H neighbors for each heavy atom (for angle calculations)
        self._protein_h_neighbors = self._get_h_neighbors(self._protein_with_h)
        self._ligand_h_neighbors = self._get_h_neighbors(self._ligand_with_h)

    def _get_h_neighbors(self, mol: Chem.Mol) -> Dict[int, List[int]]:
        """Get hydrogen neighbor indices for each heavy atom (full indices)."""
        h_neighbors = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:  # Heavy atom
                h_list = []
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        h_list.append(neighbor.GetIdx())
                h_neighbors[atom.GetIdx()] = h_list
        return h_neighbors

    def _get_coords(self, mol: Chem.Mol) -> np.ndarray:
        """Extract 3D coordinates from molecule."""
        conf = mol.GetConformer(0)
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        return np.array(coords)

    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute distance matrix between heavy atoms only."""
        diff = self._protein_coords[:, np.newaxis, :] - self._ligand_coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _compile_patterns(self):
        """Compile SMARTS patterns for pharmacophore detection."""
        self._patterns = {}
        for name, smarts in PHARMACOPHORE_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                self._patterns[name] = pattern

    def _get_matched_atoms_heavy(self, mol: Chem.Mol, pattern_name: str, full_to_heavy: Dict) -> Set[int]:
        """Get heavy atom indices matching a pattern (converted to heavy atom indices)."""
        matched = set()
        if pattern_name in self._patterns:
            pattern = self._patterns[pattern_name]
            for match in mol.GetSubstructMatches(pattern):
                for full_idx in match:
                    if full_idx in full_to_heavy:  # Only heavy atoms
                        matched.add(full_to_heavy[full_idx])
        return matched

    def _detect_pharmacophores(self):
        """Detect pharmacophore features and map to heavy atom indices."""
        # Map PHARMACOPHORE_SMARTS keys to internal category names
        category_mapping = {
            'h_donor': 'hbond_donor',
            'h_acceptor': 'hbond_acceptor',
            'positive': 'positive_charge',
            'negative': 'negative_charge',
            'aromatic': 'aromatic',
            'hydrophobic': 'hydrophobic',
            'halogen': 'halogen_bond',
            'metal_coord': 'metal_coord',
        }

        self._protein_pharmacophores = {}
        self._ligand_pharmacophores = {}

        for smarts_name, internal_name in category_mapping.items():
            self._protein_pharmacophores[internal_name] = self._get_matched_atoms_heavy(
                self._protein_with_h, smarts_name, self._protein_full_to_heavy
            )
            self._ligand_pharmacophores[internal_name] = self._get_matched_atoms_heavy(
                self._ligand_with_h, smarts_name, self._ligand_full_to_heavy
            )

        self._protein_atom_types = self._create_atom_type_mapping(self._protein_pharmacophores)
        self._ligand_atom_types = self._create_atom_type_mapping(self._ligand_pharmacophores)

    def _create_atom_type_mapping(self, pharmacophores: Dict[str, Set[int]]) -> Dict[int, List[str]]:
        """Create mapping from heavy atom index to list of pharmacophore types."""
        mapping = {}
        for category, atoms in pharmacophores.items():
            for atom_idx in atoms:
                if atom_idx not in mapping:
                    mapping[atom_idx] = []
                mapping[atom_idx].append(category)
        return mapping

    # =========================================================================
    # Atom Chemical Feature Extraction (Heavy Atoms Only)
    # =========================================================================

    def _extract_atom_features(self):
        """Extract chemical features for heavy atoms only."""
        self._protein_atom_features = self._get_atom_chemical_features(
            self._protein_with_h, self._protein_full_to_heavy, self._protein_h_neighbors
        )
        self._ligand_atom_features = self._get_atom_chemical_features(
            self._ligand_with_h, self._ligand_full_to_heavy, self._ligand_h_neighbors
        )

    def _get_atom_chemical_features(
        self, mol: Chem.Mol, full_to_heavy: Dict, h_neighbors: Dict
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract chemical features for each heavy atom.

        Includes hydrogen-related information stored on the heavy atom.
        """
        features = {}
        for atom in mol.GetAtoms():
            full_idx = atom.GetIdx()
            if full_idx not in full_to_heavy:
                continue  # Skip hydrogens

            heavy_idx = full_to_heavy[full_idx]
            symbol = atom.GetSymbol()

            # Element one-hot index
            if symbol in ELEMENT_TYPES:
                element_idx = ELEMENT_TYPES.index(symbol)
            else:
                element_idx = ELEMENT_TYPES.index('Other')

            # Hybridization one-hot index
            hyb = atom.GetHybridization()
            if hyb in HYBRIDIZATION_TYPES:
                hyb_idx = HYBRIDIZATION_TYPES.index(hyb)
            else:
                hyb_idx = NUM_HYBRIDIZATION_TYPES - 1

            # Count attached hydrogens
            num_hs = len(h_neighbors.get(full_idx, []))

            features[heavy_idx] = {
                'element': symbol,
                'element_idx': element_idx,
                'hybridization': hyb,
                'hybridization_idx': hyb_idx,
                'formal_charge': atom.GetFormalCharge(),
                'is_aromatic': atom.GetIsAromatic(),
                'is_in_ring': atom.IsInRing(),
                'num_hs': num_hs,  # Number of attached H
                'degree': atom.GetDegree() - num_hs,  # Degree to heavy atoms only
                'full_idx': full_idx,  # Keep reference to full index
            }

        return features

    # =========================================================================
    # Residue Information Extraction (Protein-specific)
    # =========================================================================

    def _extract_residue_info(self):
        """Extract residue information for protein heavy atoms."""
        self._protein_residue_info = {}

        for atom in self._protein_with_h.GetAtoms():
            full_idx = atom.GetIdx()
            if full_idx not in self._protein_full_to_heavy:
                continue

            heavy_idx = self._protein_full_to_heavy[full_idx]
            res_info = atom.GetPDBResidueInfo()

            if res_info is not None:
                res_name = res_info.GetResidueName().strip()
                res_num = res_info.GetResidueNumber()
                atom_name = res_info.GetName().strip()
                chain = res_info.GetChainId()

                if res_name in RESIDUE_TYPES:
                    res_idx = RESIDUE_TYPES.index(res_name)
                else:
                    res_idx = RESIDUE_TYPES.index('Other')

                is_backbone = atom_name in ['N', 'CA', 'C', 'O']

                self._protein_residue_info[heavy_idx] = {
                    'residue_name': res_name,
                    'residue_idx': res_idx,
                    'residue_num': res_num,
                    'atom_name': atom_name,
                    'chain': chain,
                    'is_backbone': is_backbone,
                }
            else:
                self._protein_residue_info[heavy_idx] = {
                    'residue_name': 'UNK',
                    'residue_idx': RESIDUE_TYPES.index('Other'),
                    'residue_num': -1,
                    'atom_name': '',
                    'chain': '',
                    'is_backbone': False,
                }

    # =========================================================================
    # Geometric Angle Calculations (Using H positions)
    # =========================================================================

    def _calculate_angle(
        self, coord1: np.ndarray, coord2: np.ndarray, coord3: np.ndarray
    ) -> float:
        """Calculate angle at coord2 (in degrees)."""
        v1 = coord1 - coord2
        v2 = coord3 - coord2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle

    def _calculate_dha_angle_heavy(
        self,
        donor_heavy_idx: int,
        acceptor_coord: np.ndarray,
        is_protein_donor: bool
    ) -> Tuple[Optional[float], bool]:
        """
        Calculate D-H...A angle for hydrogen bond.

        Args:
            donor_heavy_idx: Heavy atom index of donor
            acceptor_coord: 3D coordinate of acceptor
            is_protein_donor: True if donor is from protein, False if from ligand

        Returns:
            Tuple of (angle, has_valid_angle)
        """
        if is_protein_donor:
            full_idx = self._protein_heavy_to_full[donor_heavy_idx]
            h_list = self._protein_h_neighbors.get(full_idx, [])
            coords = self._protein_coords_with_h
        else:
            full_idx = self._ligand_heavy_to_full[donor_heavy_idx]
            h_list = self._ligand_h_neighbors.get(full_idx, [])
            coords = self._ligand_coords_with_h

        if not h_list:
            return None, False

        # Calculate angle with first hydrogen (or best one)
        d_coord = coords[full_idx]
        best_angle = None

        for h_idx in h_list:
            h_coord = coords[h_idx]
            angle = self._calculate_angle(d_coord, h_coord, acceptor_coord)
            if best_angle is None or angle > best_angle:
                best_angle = angle

        return best_angle, True

    def _calculate_halogen_angle_heavy(
        self,
        halogen_heavy_idx: int,
        acceptor_coord: np.ndarray,
        is_protein_halogen: bool
    ) -> Tuple[Optional[float], bool]:
        """
        Calculate C-X...A angle for halogen bond.

        Returns:
            Tuple of (angle, has_valid_angle)
        """
        if is_protein_halogen:
            mol = self._protein_with_h
            full_idx = self._protein_heavy_to_full[halogen_heavy_idx]
            coords = self._protein_coords_with_h
        else:
            mol = self._ligand_with_h
            full_idx = self._ligand_heavy_to_full[halogen_heavy_idx]
            coords = self._ligand_coords_with_h

        # Find attached heavy atom (usually carbon)
        atom = mol.GetAtomWithIdx(full_idx)
        c_idx = None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() > 1:
                c_idx = neighbor.GetIdx()
                break

        if c_idx is None:
            return None, False

        c_coord = coords[c_idx]
        x_coord = coords[full_idx]

        angle = self._calculate_angle(c_coord, x_coord, acceptor_coord)
        return angle, True

    # =========================================================================
    # Interaction Detection Methods
    # =========================================================================

    def _get_close_pairs(self, cutoff: Optional[float] = None) -> List[Tuple[int, int, float]]:
        """Get heavy atom pairs within distance cutoff."""
        cutoff = cutoff or self.distance_cutoff
        pairs = []
        protein_idxs, ligand_idxs = np.where(self._distance_matrix < cutoff)
        for p_idx, l_idx in zip(protein_idxs, ligand_idxs):
            dist = self._distance_matrix[p_idx, l_idx]
            pairs.append((int(p_idx), int(l_idx), float(dist)))
        return pairs

    def detect_hydrogen_bonds(self) -> List[Interaction]:
        """Detect hydrogen bonds with D-H-A angle calculation."""
        if 'hydrogen_bonds' in self._cache:
            return self._cache['hydrogen_bonds']

        interactions = []
        cutoff = INTERACTION_TYPES['hydrogen_bond']['distance_cutoff']
        angle_cutoff = INTERACTION_TYPES['hydrogen_bond'].get('angle_cutoff', 120.0)

        # Protein donor - Ligand acceptor
        for p_idx in self._protein_pharmacophores['hbond_donor']:
            for l_idx in self._ligand_pharmacophores['hbond_acceptor']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        dha_angle, has_valid = self._calculate_dha_angle_heavy(
                            p_idx, self._ligand_coords[l_idx], is_protein_donor=True
                        )

                        # Filter by angle if valid
                        if has_valid and dha_angle is not None and dha_angle < angle_cutoff:
                            continue

                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='hydrogen_bond',
                            distance=dist,
                            dha_angle=dha_angle,
                            has_valid_angle=has_valid,
                            metadata={'donor': 'protein', 'acceptor': 'ligand'}
                        ))

        # Protein acceptor - Ligand donor
        for p_idx in self._protein_pharmacophores['hbond_acceptor']:
            for l_idx in self._ligand_pharmacophores['hbond_donor']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        dha_angle, has_valid = self._calculate_dha_angle_heavy(
                            l_idx, self._protein_coords[p_idx], is_protein_donor=False
                        )

                        if has_valid and dha_angle is not None and dha_angle < angle_cutoff:
                            continue

                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='hydrogen_bond',
                            distance=dist,
                            dha_angle=dha_angle,
                            has_valid_angle=has_valid,
                            metadata={'donor': 'ligand', 'acceptor': 'protein'}
                        ))

        self._cache['hydrogen_bonds'] = interactions
        return interactions

    def detect_salt_bridges(self) -> List[Interaction]:
        """Detect salt bridges (ionic interactions)."""
        if 'salt_bridges' in self._cache:
            return self._cache['salt_bridges']

        interactions = []
        cutoff = INTERACTION_TYPES['salt_bridge']['distance_cutoff']

        for p_idx in self._protein_pharmacophores['positive_charge']:
            for l_idx in self._ligand_pharmacophores['negative_charge']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='salt_bridge',
                            distance=dist,
                            has_valid_angle=True,  # No angle needed for salt bridge
                            metadata={'protein_charge': 'positive', 'ligand_charge': 'negative'}
                        ))

        for p_idx in self._protein_pharmacophores['negative_charge']:
            for l_idx in self._ligand_pharmacophores['positive_charge']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='salt_bridge',
                            distance=dist,
                            has_valid_angle=True,
                            metadata={'protein_charge': 'negative', 'ligand_charge': 'positive'}
                        ))

        self._cache['salt_bridges'] = interactions
        return interactions

    def detect_pi_stacking(self) -> List[Interaction]:
        """Detect pi-stacking with ring angle calculation."""
        if 'pi_stacking' in self._cache:
            return self._cache['pi_stacking']

        interactions = []
        cutoff = INTERACTION_TYPES['pi_stacking']['distance_cutoff']

        protein_rings = self._get_aromatic_rings(
            self._protein_with_h, self._protein_coords_with_h, self._protein_full_to_heavy
        )
        ligand_rings = self._get_aromatic_rings(
            self._ligand_with_h, self._ligand_coords_with_h, self._ligand_full_to_heavy
        )

        for p_ring in protein_rings:
            for l_ring in ligand_rings:
                dist = np.linalg.norm(p_ring['center'] - l_ring['center'])
                if dist < cutoff:
                    cos_angle = abs(np.dot(p_ring['normal'], l_ring['normal']))
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                    is_parallel = angle < 30 or angle > 150
                    is_perpendicular = 60 < angle < 120

                    if is_parallel or is_perpendicular:
                        interactions.append(Interaction(
                            protein_atom_idx=p_ring['heavy_atoms'][0],
                            ligand_atom_idx=l_ring['heavy_atoms'][0],
                            interaction_type='pi_stacking',
                            distance=dist,
                            angle=angle,
                            has_valid_angle=True,
                            metadata={
                                'stacking_type': 'parallel' if is_parallel else 'T-shaped',
                                'protein_ring_atoms': p_ring['heavy_atoms'],
                                'ligand_ring_atoms': l_ring['heavy_atoms'],
                            }
                        ))

        self._cache['pi_stacking'] = interactions
        return interactions

    def detect_cation_pi(self) -> List[Interaction]:
        """Detect cation-pi interactions."""
        if 'cation_pi' in self._cache:
            return self._cache['cation_pi']

        interactions = []
        cutoff = INTERACTION_TYPES['cation_pi']['distance_cutoff']

        protein_rings = self._get_aromatic_rings(
            self._protein_with_h, self._protein_coords_with_h, self._protein_full_to_heavy
        )
        ligand_rings = self._get_aromatic_rings(
            self._ligand_with_h, self._ligand_coords_with_h, self._ligand_full_to_heavy
        )

        # Protein cation - Ligand aromatic
        for p_idx in self._protein_pharmacophores['positive_charge']:
            if p_idx >= self.num_protein_atoms:
                continue
            p_coord = self._protein_coords[p_idx]
            for l_ring in ligand_rings:
                dist = np.linalg.norm(p_coord - l_ring['center'])
                if dist < cutoff:
                    vec_to_center = l_ring['center'] - p_coord
                    vec_to_center = vec_to_center / (np.linalg.norm(vec_to_center) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(
                        abs(np.dot(vec_to_center, l_ring['normal'])), -1, 1
                    )))

                    interactions.append(Interaction(
                        protein_atom_idx=p_idx,
                        ligand_atom_idx=l_ring['heavy_atoms'][0],
                        interaction_type='cation_pi',
                        distance=dist,
                        angle=angle,
                        has_valid_angle=True,
                        metadata={'cation': 'protein', 'pi': 'ligand'}
                    ))

        # Ligand cation - Protein aromatic
        for l_idx in self._ligand_pharmacophores['positive_charge']:
            if l_idx >= self.num_ligand_atoms:
                continue
            l_coord = self._ligand_coords[l_idx]
            for p_ring in protein_rings:
                dist = np.linalg.norm(l_coord - p_ring['center'])
                if dist < cutoff:
                    vec_to_center = p_ring['center'] - l_coord
                    vec_to_center = vec_to_center / (np.linalg.norm(vec_to_center) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(
                        abs(np.dot(vec_to_center, p_ring['normal'])), -1, 1
                    )))

                    interactions.append(Interaction(
                        protein_atom_idx=p_ring['heavy_atoms'][0],
                        ligand_atom_idx=l_idx,
                        interaction_type='cation_pi',
                        distance=dist,
                        angle=angle,
                        has_valid_angle=True,
                        metadata={'cation': 'ligand', 'pi': 'protein'}
                    ))

        self._cache['cation_pi'] = interactions
        return interactions

    def detect_hydrophobic(self) -> List[Interaction]:
        """Detect hydrophobic contacts."""
        if 'hydrophobic' in self._cache:
            return self._cache['hydrophobic']

        interactions = []
        cutoff = INTERACTION_TYPES['hydrophobic']['distance_cutoff']

        for p_idx in self._protein_pharmacophores['hydrophobic']:
            for l_idx in self._ligand_pharmacophores['hydrophobic']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='hydrophobic',
                            distance=dist,
                            has_valid_angle=True,
                        ))

        self._cache['hydrophobic'] = interactions
        return interactions

    def detect_halogen_bonds(self) -> List[Interaction]:
        """Detect halogen bonds with C-X-A angle calculation."""
        if 'halogen_bonds' in self._cache:
            return self._cache['halogen_bonds']

        interactions = []
        cutoff = INTERACTION_TYPES['halogen_bond']['distance_cutoff']
        angle_cutoff = INTERACTION_TYPES['halogen_bond'].get('angle_cutoff', 140.0)

        # Ligand halogen - Protein acceptor
        for l_idx in self._ligand_pharmacophores['halogen_bond']:
            for p_idx in self._protein_pharmacophores['hbond_acceptor']:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        cxa_angle, has_valid = self._calculate_halogen_angle_heavy(
                            l_idx, self._protein_coords[p_idx], is_protein_halogen=False
                        )

                        if has_valid and cxa_angle is not None and cxa_angle < angle_cutoff:
                            continue

                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='halogen_bond',
                            distance=dist,
                            cxa_angle=cxa_angle,
                            has_valid_angle=has_valid,
                            metadata={'halogen_source': 'ligand'}
                        ))

        self._cache['halogen_bonds'] = interactions
        return interactions

    def detect_metal_coordination(self) -> List[Interaction]:
        """Detect metal coordination interactions.

        Uses element-based detection for protein metal ions (Zn, Fe, Mg, etc.)
        and SMARTS-based detection for ligand coordinating atoms (N, O, S lone pairs).
        """
        if 'metal_coordination' in self._cache:
            return self._cache['metal_coordination']

        interactions = []
        cutoff = INTERACTION_TYPES['metal_coordination']['distance_cutoff']

        # Find protein metal atoms by element type
        metal_symbols = {'Zn', 'Fe', 'Mg', 'Mn', 'Cu', 'Co', 'Ni', 'Ca', 'Na', 'K'}
        protein_metals = set()
        for atom in self._protein_with_h.GetAtoms():
            full_idx = atom.GetIdx()
            if full_idx in self._protein_full_to_heavy:
                if atom.GetSymbol() in metal_symbols:
                    protein_metals.add(self._protein_full_to_heavy[full_idx])

        # Ligand coordinating atoms: metal_coord + h_acceptor + negative_charge
        ligand_coordinators = set()
        ligand_coordinators.update(self._ligand_pharmacophores.get('metal_coord', set()))
        ligand_coordinators.update(self._ligand_pharmacophores.get('hbond_acceptor', set()))
        ligand_coordinators.update(self._ligand_pharmacophores.get('negative_charge', set()))

        for p_idx in protein_metals:
            for l_idx in ligand_coordinators:
                if p_idx < self.num_protein_atoms and l_idx < self.num_ligand_atoms:
                    dist = self._distance_matrix[p_idx, l_idx]
                    if dist < cutoff:
                        interactions.append(Interaction(
                            protein_atom_idx=p_idx,
                            ligand_atom_idx=l_idx,
                            interaction_type='metal_coordination',
                            distance=dist,
                            has_valid_angle=True,
                        ))

        self._cache['metal_coordination'] = interactions
        return interactions

    def _get_aromatic_rings(
        self, mol: Chem.Mol, coords: np.ndarray, full_to_heavy: Dict
    ) -> List[Dict[str, Any]]:
        """Get aromatic ring info with heavy atom indices."""
        ring_info = mol.GetRingInfo()
        rings = []

        for ring_atoms in ring_info.AtomRings():
            # Check if all ring atoms are aromatic and heavy
            is_aromatic = True
            heavy_atoms = []
            for idx in ring_atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() == 1:  # Skip if H in ring (shouldn't happen)
                    continue
                if not atom.GetIsAromatic():
                    is_aromatic = False
                    break
                if idx in full_to_heavy:
                    heavy_atoms.append(full_to_heavy[idx])

            if not is_aromatic or not heavy_atoms:
                continue

            # Get ring coordinates (heavy atoms only in full coords)
            ring_coords = coords[list(ring_atoms)]
            center = np.mean(ring_coords, axis=0)

            if len(ring_coords) >= 3:
                v1 = ring_coords[1] - ring_coords[0]
                v2 = ring_coords[2] - ring_coords[0]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 1e-8:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
            else:
                normal = np.array([0, 0, 1])

            rings.append({
                'heavy_atoms': heavy_atoms,  # Heavy atom indices
                'center': center,
                'normal': normal,
            })

        return rings

    # =========================================================================
    # Main Feature Extraction Methods
    # =========================================================================

    def detect_all_interactions(self) -> List[Interaction]:
        """Detect all types of interactions."""
        if 'all_interactions' in self._cache:
            return self._cache['all_interactions']

        all_interactions = []
        all_interactions.extend(self.detect_hydrogen_bonds())
        all_interactions.extend(self.detect_salt_bridges())
        all_interactions.extend(self.detect_pi_stacking())
        all_interactions.extend(self.detect_cation_pi())
        all_interactions.extend(self.detect_hydrophobic())
        all_interactions.extend(self.detect_halogen_bonds())
        all_interactions.extend(self.detect_metal_coordination())

        self._cache['all_interactions'] = all_interactions
        return all_interactions

    def get_interaction_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interaction edges as tensors.

        Returns:
            Tuple of (edges, edge_features):
                - edges: [2, num_interactions] tensor of heavy atom indices
                - edge_features: [num_interactions, feature_dim] tensor
        """
        interactions = self.detect_all_interactions()

        if not interactions:
            return (
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, self._get_edge_feature_dim(), dtype=torch.float32)
            )

        edges = torch.tensor([
            [inter.protein_atom_idx for inter in interactions],
            [inter.ligand_atom_idx for inter in interactions]
        ], dtype=torch.long)

        edge_features = self._build_edge_features(interactions)

        return edges, edge_features

    def _get_edge_feature_dim(self) -> int:
        """Get total edge feature dimension."""
        return (
            NUM_INTERACTION_TYPES +            # interaction type one-hot (7)
            4 +                                # distance + angles + has_valid (4)
            NUM_ELEMENT_TYPES * 2 +            # element types (20)
            NUM_HYBRIDIZATION_TYPES * 2 +      # hybridization (12)
            2 +                                # formal charges (2)
            2 +                                # aromatic (2)
            4 +                                # in_ring (2) + degree (2)
            NUM_RESIDUE_TYPES +                # residue type (21)
            1 +                                # is_backbone (1)
            1                                  # interaction_strength (1)
        )  # Total: 74

    def _build_edge_features(self, interactions: List[Interaction]) -> torch.Tensor:
        """Build comprehensive feature tensor for interactions."""
        num_interactions = len(interactions)
        feature_dim = self._get_edge_feature_dim()
        features = torch.zeros(num_interactions, feature_dim)

        for i, inter in enumerate(interactions):
            offset = 0

            # 1. Interaction type one-hot (7 dims)
            type_idx = INTERACTION_TYPE_IDX.get(inter.interaction_type, 0)
            features[i, offset + type_idx] = 1.0
            offset += NUM_INTERACTION_TYPES

            # 2. Distance and geometric features (4 dims)
            features[i, offset] = inter.distance / self.distance_cutoff
            offset += 1

            # Combined angle (ring/DHA/CXA - use whichever is available)
            angle_val = inter.angle or inter.dha_angle or inter.cxa_angle
            if angle_val is not None:
                features[i, offset] = angle_val / 180.0
            offset += 1

            # Has valid angle flag
            features[i, offset] = float(inter.has_valid_angle)
            offset += 1

            # Angle type indicator (0=none, 0.33=ring, 0.67=dha, 1.0=cxa)
            if inter.angle is not None:
                features[i, offset] = 0.33
            elif inter.dha_angle is not None:
                features[i, offset] = 0.67
            elif inter.cxa_angle is not None:
                features[i, offset] = 1.0
            offset += 1

            # 3. Element types (20 dims)
            p_feats = self._protein_atom_features.get(inter.protein_atom_idx, {})
            l_feats = self._ligand_atom_features.get(inter.ligand_atom_idx, {})

            p_elem_idx = p_feats.get('element_idx', NUM_ELEMENT_TYPES - 1)
            features[i, offset + p_elem_idx] = 1.0
            offset += NUM_ELEMENT_TYPES

            l_elem_idx = l_feats.get('element_idx', NUM_ELEMENT_TYPES - 1)
            features[i, offset + l_elem_idx] = 1.0
            offset += NUM_ELEMENT_TYPES

            # 4. Hybridization (12 dims)
            p_hyb_idx = p_feats.get('hybridization_idx', NUM_HYBRIDIZATION_TYPES - 1)
            features[i, offset + p_hyb_idx] = 1.0
            offset += NUM_HYBRIDIZATION_TYPES

            l_hyb_idx = l_feats.get('hybridization_idx', NUM_HYBRIDIZATION_TYPES - 1)
            features[i, offset + l_hyb_idx] = 1.0
            offset += NUM_HYBRIDIZATION_TYPES

            # 5. Formal charges (2 dims)
            p_charge = p_feats.get('formal_charge', 0)
            l_charge = l_feats.get('formal_charge', 0)
            features[i, offset] = (p_charge + 2) / 4.0
            features[i, offset + 1] = (l_charge + 2) / 4.0
            offset += 2

            # 6. Aromatic (2 dims)
            features[i, offset] = float(p_feats.get('is_aromatic', False))
            features[i, offset + 1] = float(l_feats.get('is_aromatic', False))
            offset += 2

            # 7. Ring membership + degree (4 dims)
            features[i, offset] = float(p_feats.get('is_in_ring', False))
            features[i, offset + 1] = float(l_feats.get('is_in_ring', False))
            features[i, offset + 2] = p_feats.get('degree', 0) / 4.0
            features[i, offset + 3] = l_feats.get('degree', 0) / 4.0
            offset += 4

            # 8. Residue type for protein (21 dims)
            res_info = self._protein_residue_info.get(inter.protein_atom_idx, {})
            res_idx = res_info.get('residue_idx', NUM_RESIDUE_TYPES - 1)
            features[i, offset + res_idx] = 1.0
            offset += NUM_RESIDUE_TYPES

            # 9. Is backbone (1 dim)
            is_backbone = res_info.get('is_backbone', False)
            features[i, offset] = float(is_backbone)
            offset += 1

            # 10. Interaction strength: Gaussian decay from ideal distance (1 dim)
            ideal = IDEAL_DISTANCES.get(inter.interaction_type, 3.0)
            sigma = 0.5
            strength = math.exp(-0.5 * ((inter.distance - ideal) / sigma) ** 2)
            features[i, offset] = strength
            offset += 1

        return features

    def _get_contact_edges(
        self, cutoff: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all heavy atom pairs within cutoff as generic contact edges.

        Args:
            cutoff: Distance cutoff in Angstrom.

        Returns:
            contact_edges: (2, E_contact) protein/ligand heavy atom indices.
            contact_distances: (E_contact,) pairwise distances.
        """
        mask = self._distance_matrix < cutoff
        p_idx, l_idx = np.where(mask)
        if len(p_idx) == 0:
            return (
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
            )
        edges = torch.tensor(np.stack([p_idx, l_idx]), dtype=torch.long)
        distances = torch.tensor(
            self._distance_matrix[mask], dtype=torch.float32
        )
        return edges, distances

    def get_interaction_graph(
        self,
        include_contacts: bool = False,
        contact_cutoff: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get complete interaction graph data.

        Args:
            include_contacts: If True, also include all heavy atom pairs within
                cutoff as generic contact edges (separate from pharmacophore edges).
            contact_cutoff: Distance cutoff for contacts (default: self.distance_cutoff).
        """
        interactions = self.detect_all_interactions()
        edges, edge_features = self.get_interaction_edges()

        type_counts = {}
        for inter in interactions:
            type_counts[inter.interaction_type] = type_counts.get(inter.interaction_type, 0) + 1

        graph = {
            'edges': edges,
            'edge_features': edge_features,
            'interactions': interactions,
            'num_interactions': len(interactions),
            'interaction_counts': type_counts,
            'num_protein_atoms': self.num_protein_atoms,  # Heavy atoms only
            'num_ligand_atoms': self.num_ligand_atoms,    # Heavy atoms only
            'distance_cutoff': self.distance_cutoff,
            'feature_dim': self._get_edge_feature_dim(),
            'metadata': {
                'interaction_type_indices': INTERACTION_TYPE_IDX,
                'pharmacophore_indices': PHARMACOPHORE_IDX,
                'element_types': ELEMENT_TYPES,
                'residue_types': RESIDUE_TYPES,
                'heavy_atom_only': True,
            }
        }

        if include_contacts:
            c_cutoff = contact_cutoff or self.distance_cutoff
            contact_edges, contact_distances = self._get_contact_edges(c_cutoff)
            graph['contact_edges'] = contact_edges
            graph['contact_distances'] = contact_distances
            graph['num_contacts'] = contact_edges.shape[1]

        return graph

    def get_atom_pharmacophore_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get pharmacophore features for heavy atoms."""
        protein_feats = torch.zeros(self.num_protein_atoms, NUM_PHARMACOPHORE_TYPES)
        ligand_feats = torch.zeros(self.num_ligand_atoms, NUM_PHARMACOPHORE_TYPES)

        for atom_idx, types in self._protein_atom_types.items():
            if atom_idx < self.num_protein_atoms:
                for ptype in types:
                    if ptype in PHARMACOPHORE_IDX:
                        protein_feats[atom_idx, PHARMACOPHORE_IDX[ptype]] = 1.0

        for atom_idx, types in self._ligand_atom_types.items():
            if atom_idx < self.num_ligand_atoms:
                for ptype in types:
                    if ptype in PHARMACOPHORE_IDX:
                        ligand_feats[atom_idx, PHARMACOPHORE_IDX[ptype]] = 1.0

        return protein_feats, ligand_feats

    def get_atom_chemical_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get chemical features for heavy atoms.

        Returns:
            Tuple of (protein_features, ligand_features)
            Features: element (10) + hybridization (6) + charge (1) +
                     num_hs (1) + aromatic (1) + in_ring (1) + degree (1) = 21
        """
        atom_feat_dim = NUM_ELEMENT_TYPES + NUM_HYBRIDIZATION_TYPES + 5

        protein_feats = torch.zeros(self.num_protein_atoms, atom_feat_dim)
        ligand_feats = torch.zeros(self.num_ligand_atoms, atom_feat_dim)

        for idx, feats in self._protein_atom_features.items():
            if idx >= self.num_protein_atoms:
                continue
            offset = 0
            protein_feats[idx, offset + feats['element_idx']] = 1.0
            offset += NUM_ELEMENT_TYPES
            protein_feats[idx, offset + feats['hybridization_idx']] = 1.0
            offset += NUM_HYBRIDIZATION_TYPES
            protein_feats[idx, offset] = (feats['formal_charge'] + 2) / 4.0
            protein_feats[idx, offset + 1] = feats['num_hs'] / 4.0
            protein_feats[idx, offset + 2] = float(feats['is_aromatic'])
            protein_feats[idx, offset + 3] = float(feats['is_in_ring'])
            protein_feats[idx, offset + 4] = feats['degree'] / 4.0

        for idx, feats in self._ligand_atom_features.items():
            if idx >= self.num_ligand_atoms:
                continue
            offset = 0
            ligand_feats[idx, offset + feats['element_idx']] = 1.0
            offset += NUM_ELEMENT_TYPES
            ligand_feats[idx, offset + feats['hybridization_idx']] = 1.0
            offset += NUM_HYBRIDIZATION_TYPES
            ligand_feats[idx, offset] = (feats['formal_charge'] + 2) / 4.0
            ligand_feats[idx, offset + 1] = feats['num_hs'] / 4.0
            ligand_feats[idx, offset + 2] = float(feats['is_aromatic'])
            ligand_feats[idx, offset + 3] = float(feats['is_in_ring'])
            ligand_feats[idx, offset + 4] = feats['degree'] / 4.0

        return protein_feats, ligand_feats

    def get_residue_features(self) -> torch.Tensor:
        """Get residue-level features for protein heavy atoms."""
        feat_dim = NUM_RESIDUE_TYPES + 2
        features = torch.zeros(self.num_protein_atoms, feat_dim)

        for idx, res_info in self._protein_residue_info.items():
            if idx >= self.num_protein_atoms:
                continue
            features[idx, res_info['residue_idx']] = 1.0
            features[idx, NUM_RESIDUE_TYPES] = float(res_info['is_backbone'])
            features[idx, NUM_RESIDUE_TYPES + 1] = float(not res_info['is_backbone'])

        return features

    def get_distance_based_edges(
        self, cutoff: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all heavy atom pairs within distance cutoff."""
        pairs = self._get_close_pairs(cutoff)

        if not pairs:
            return (
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32)
            )

        edges = torch.tensor([
            [p[0] for p in pairs],
            [p[1] for p in pairs]
        ], dtype=torch.long)

        distances = torch.tensor([p[2] for p in pairs], dtype=torch.float32)

        return edges, distances

    def get_heavy_atom_coords(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get 3D coordinates of heavy atoms only."""
        return (
            torch.tensor(self._protein_coords, dtype=torch.float32),
            torch.tensor(self._ligand_coords, dtype=torch.float32)
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_interaction_summary(self) -> str:
        """Get text summary of detected interactions."""
        interactions = self.detect_all_interactions()
        type_counts = {}
        angles_available = 0
        for inter in interactions:
            type_counts[inter.interaction_type] = type_counts.get(inter.interaction_type, 0) + 1
            if inter.has_valid_angle:
                angles_available += 1

        lines = [
            f"PLI Summary (Heavy Atom Only)",
            f"  Protein heavy atoms: {self.num_protein_atoms}",
            f"  Ligand heavy atoms: {self.num_ligand_atoms}",
            f"  Distance cutoff: {self.distance_cutoff} Å",
            f"  Total interactions: {len(interactions)}",
            f"  Valid angle calculations: {angles_available}/{len(interactions)}",
            f"  Feature dimension: {self._get_edge_feature_dim()}",
            "",
            "Interaction counts:"
        ]

        for itype in INTERACTION_TYPE_IDX:
            count = type_counts.get(itype, 0)
            lines.append(f"  {itype}: {count}")

        return '\n'.join(lines)

    def get_feature_description(self) -> Dict[str, Any]:
        """Get description of feature dimensions."""
        offset = 0
        breakdown = {}

        breakdown['interaction_type'] = (offset, NUM_INTERACTION_TYPES)
        offset += NUM_INTERACTION_TYPES

        breakdown['distance'] = (offset, 1)
        offset += 1
        breakdown['angle'] = (offset, 1)
        offset += 1
        breakdown['has_valid_angle'] = (offset, 1)
        offset += 1
        breakdown['angle_type'] = (offset, 1)
        offset += 1

        breakdown['protein_element'] = (offset, NUM_ELEMENT_TYPES)
        offset += NUM_ELEMENT_TYPES
        breakdown['ligand_element'] = (offset, NUM_ELEMENT_TYPES)
        offset += NUM_ELEMENT_TYPES

        breakdown['protein_hybridization'] = (offset, NUM_HYBRIDIZATION_TYPES)
        offset += NUM_HYBRIDIZATION_TYPES
        breakdown['ligand_hybridization'] = (offset, NUM_HYBRIDIZATION_TYPES)
        offset += NUM_HYBRIDIZATION_TYPES

        breakdown['formal_charges'] = (offset, 2)
        offset += 2

        breakdown['aromatic'] = (offset, 2)
        offset += 2

        breakdown['ring_and_degree'] = (offset, 4)
        offset += 4

        breakdown['residue_type'] = (offset, NUM_RESIDUE_TYPES)
        offset += NUM_RESIDUE_TYPES

        breakdown['is_backbone'] = (offset, 1)
        offset += 1

        return {
            'total_dim': offset,
            'breakdown': breakdown
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PLInteractionFeaturizer("
            f"protein_heavy={self.num_protein_atoms}, "
            f"ligand_heavy={self.num_ligand_atoms}, "
            f"cutoff={self.distance_cutoff}Å, "
            f"feature_dim={self._get_edge_feature_dim()})"
        )
