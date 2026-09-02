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
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ..rdkit_utils import has_3d

from ..constants import (
    PHARMACOPHORE_SMARTS,
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
    # Feature normalization
    FORMAL_CHARGE_OFFSET,
    FORMAL_CHARGE_SCALE,
    DEGREE_SCALE,
    NUM_HS_SCALE,
)
from ..rdkit_utils import has_3d, get_positions
from ..errors import InputError


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

    Features include (Total: 74 dims):
        - Interaction type one-hot (7 dims)
        - Distance and angle (4 dims): distance, angle, has_valid_angle, angle_type
        - Element types one-hot (20 dims): protein (10) + ligand (10)
        - Hybridization one-hot (12 dims): protein (6) + ligand (6)
        - Formal charges (2 dims)
        - Aromatic flags (2 dims)
        - Ring membership + degree (4 dims)
        - Residue type one-hot (21 dims)
        - Is backbone flag (1 dim)
        - Interaction strength (1 dim): Gaussian decay from ideal distance

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
        knn_cutoff: Optional[int] = None,
    ):
        """
        Initialize the PLI featurizer.

        Args:
            protein_mol: RDKit mol object for protein (must have 3D coordinates)
            ligand_mol: RDKit mol object for ligand (must have 3D coordinates)
            distance_cutoff: Maximum distance for interaction detection (Angstrom)
            knn_cutoff: Optional k-nearest neighbors cutoff for bipartite edges

        Raises:
            ValueError: If molecules lack 3D coordinates
        """
        self.distance_cutoff = distance_cutoff
        self.knn_cutoff = knn_cutoff
        self._cache: Dict[str, Any] = {}

        # Store original molecules and prepare with hydrogens
        self._protein_with_h = self._prepare_mol_with_hydrogens(protein_mol)
        self._ligand_with_h = self._prepare_mol_with_hydrogens(ligand_mol)

        # Validate 3D coordinates
        if not has_3d(self._protein_with_h):
            raise InputError("Protein molecule must have a 3D conformer required for interaction detection")
        if not has_3d(self._ligand_with_h):
            raise InputError("Ligand molecule must have a 3D conformer required for interaction detection")

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

        # Build detector and encoder delegates
        self._build_delegates()

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _prepare_mol_with_hydrogens(self, mol: Chem.Mol) -> Chem.Mol:
        """Prepare molecule with explicit hydrogens and 3D coordinates."""
        if mol is None:
            raise InputError("Molecule cannot be None")

        mol = Chem.Mol(mol)  # Copy to avoid modifying original

        # Check if already has hydrogens
        has_hydrogens = any(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in range(mol.GetNumAtoms())
        )

        if not has_hydrogens:
            if has_3d(mol):
                mol = Chem.AddHs(mol, addCoords=True)
            else:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)

        return mol

    def _build_heavy_atom_mappings(self):
        """Build mappings between heavy atom indices and full molecule indices."""
        # Protein
        self._protein_heavy_indices = []  # heavy_idx -> full_idx
        self._protein_heavy_to_full = {}  # heavy_idx -> full_idx
        self._protein_full_to_heavy = {}  # full_idx -> heavy_idx

        heavy_idx = 0
        _protein = self._protein_with_h
        for _idx in range(_protein.GetNumAtoms()):
            atom = _protein.GetAtomWithIdx(_idx)
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
        _ligand = self._ligand_with_h
        for _idx in range(_ligand.GetNumAtoms()):
            atom = _ligand.GetAtomWithIdx(_idx)
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
        for _idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(_idx)
            if atom.GetAtomicNum() > 1:  # Heavy atom
                h_list = []
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        h_list.append(neighbor.GetIdx())
                h_neighbors[atom.GetIdx()] = h_list
        return h_neighbors

    def _get_coords(self, mol: Chem.Mol) -> np.ndarray:
        """Extract 3D coordinates from molecule."""
        return get_positions(mol)

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
        for _idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(_idx)
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

        _protein = self._protein_with_h
        for _idx in range(_protein.GetNumAtoms()):
            atom = _protein.GetAtomWithIdx(_idx)
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
    # Delegate Construction
    # =========================================================================

    def _build_delegates(self):
        """Build InteractionDetector and InteractionGraphBuilder delegates."""
        from .pli_detectors import InteractionDetector
        from .pli_encoding import InteractionGraphBuilder

        self._detector = InteractionDetector(
            protein_coords=self._protein_coords,
            ligand_coords=self._ligand_coords,
            distance_matrix=self._distance_matrix,
            protein_pharmacophores=self._protein_pharmacophores,
            ligand_pharmacophores=self._ligand_pharmacophores,
            num_protein_atoms=self.num_protein_atoms,
            num_ligand_atoms=self.num_ligand_atoms,
            protein_with_h=self._protein_with_h,
            ligand_with_h=self._ligand_with_h,
            protein_coords_with_h=self._protein_coords_with_h,
            ligand_coords_with_h=self._ligand_coords_with_h,
            protein_heavy_to_full=self._protein_heavy_to_full,
            ligand_heavy_to_full=self._ligand_heavy_to_full,
            protein_full_to_heavy=self._protein_full_to_heavy,
            ligand_full_to_heavy=self._ligand_full_to_heavy,
            protein_h_neighbors=self._protein_h_neighbors,
            ligand_h_neighbors=self._ligand_h_neighbors,
            distance_cutoff=self.distance_cutoff,
            knn_cutoff=self.knn_cutoff,
        )

        self._encoder = InteractionGraphBuilder(
            distance_cutoff=self.distance_cutoff,
            knn_cutoff=self.knn_cutoff,
            distance_matrix=self._distance_matrix,
            num_protein_atoms=self.num_protein_atoms,
            num_ligand_atoms=self.num_ligand_atoms,
            protein_atom_features=self._protein_atom_features,
            ligand_atom_features=self._ligand_atom_features,
            protein_residue_info=self._protein_residue_info,
        )

    # =========================================================================
    # Interaction Detection (delegated)
    # =========================================================================

    def detect_hydrogen_bonds(self) -> List[Interaction]:
        """Detect hydrogen bonds with D-H-A angle calculation."""
        return self._detector.detect_hydrogen_bonds()

    def detect_salt_bridges(self) -> List[Interaction]:
        """Detect salt bridges (ionic interactions)."""
        return self._detector.detect_salt_bridges()

    def detect_pi_stacking(self) -> List[Interaction]:
        """Detect pi-stacking with ring angle calculation."""
        return self._detector.detect_pi_stacking()

    def detect_cation_pi(self) -> List[Interaction]:
        """Detect cation-pi interactions."""
        return self._detector.detect_cation_pi()

    def detect_hydrophobic(self) -> List[Interaction]:
        """Detect hydrophobic contacts."""
        return self._detector.detect_hydrophobic()

    def detect_halogen_bonds(self) -> List[Interaction]:
        """Detect halogen bonds with C-X-A angle calculation."""
        return self._detector.detect_halogen_bonds()

    def detect_metal_coordination(self) -> List[Interaction]:
        """Detect metal coordination interactions."""
        return self._detector.detect_metal_coordination()

    def detect_all_interactions(self) -> List[Interaction]:
        """Detect all types of interactions."""
        return self._detector.detect_all_interactions()

    # =========================================================================
    # Graph Building (delegated)
    # =========================================================================

    def get_interaction_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interaction edges as tensors.

        Returns:
            Tuple of (edges, edge_features):
                - edges: [2, num_interactions] tensor of heavy atom indices
                - edge_features: [num_interactions, feature_dim] tensor
        """
        interactions = self.detect_all_interactions()
        return self._encoder.get_interaction_edges(interactions)

    def _get_edge_feature_dim(self) -> int:
        """Get total edge feature dimension."""
        return self._encoder._get_edge_feature_dim()

    def get_interaction_graph(
        self,
        include_contacts: bool = False,
        contact_cutoff: Optional[float] = None,
        knn_cutoff: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get complete interaction graph data.

        Args:
            include_contacts: If True, also include all heavy atom pairs within
                cutoff as generic contact edges (separate from pharmacophore edges).
            contact_cutoff: Distance cutoff for contacts (default: self.distance_cutoff).
            knn_cutoff: Optional k-nearest neighbors cutoff for contact edges.
        """
        interactions = self.detect_all_interactions()
        edges, edge_features = self.get_interaction_edges()
        return self._encoder.get_interaction_graph(
            interactions, edges, edge_features,
            include_contacts=include_contacts,
            contact_cutoff=contact_cutoff,
            knn_cutoff=knn_cutoff,
        )

    def get_interaction_summary(self) -> str:
        """Get text summary of detected interactions."""
        interactions = self.detect_all_interactions()
        return self._encoder.get_interaction_summary(interactions)

    def get_feature_description(self) -> Dict[str, Any]:
        """Get description of feature dimensions."""
        return self._encoder.get_feature_description()

    # =========================================================================
    # Atom Feature Methods (kept here - not part of detection or encoding)
    # =========================================================================

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
            protein_feats[idx, offset] = (feats['formal_charge'] + FORMAL_CHARGE_OFFSET) / FORMAL_CHARGE_SCALE
            protein_feats[idx, offset + 1] = feats['num_hs'] / NUM_HS_SCALE
            protein_feats[idx, offset + 2] = float(feats['is_aromatic'])
            protein_feats[idx, offset + 3] = float(feats['is_in_ring'])
            protein_feats[idx, offset + 4] = feats['degree'] / DEGREE_SCALE

        for idx, feats in self._ligand_atom_features.items():
            if idx >= self.num_ligand_atoms:
                continue
            offset = 0
            ligand_feats[idx, offset + feats['element_idx']] = 1.0
            offset += NUM_ELEMENT_TYPES
            ligand_feats[idx, offset + feats['hybridization_idx']] = 1.0
            offset += NUM_HYBRIDIZATION_TYPES
            ligand_feats[idx, offset] = (feats['formal_charge'] + FORMAL_CHARGE_OFFSET) / FORMAL_CHARGE_SCALE
            ligand_feats[idx, offset + 1] = feats['num_hs'] / NUM_HS_SCALE
            ligand_feats[idx, offset + 2] = float(feats['is_aromatic'])
            ligand_feats[idx, offset + 3] = float(feats['is_in_ring'])
            ligand_feats[idx, offset + 4] = feats['degree'] / DEGREE_SCALE

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
        self, distance_cutoff: Optional[float] = None,
        knn_cutoff: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all heavy atom pairs within distance cutoff."""
        pairs = self._detector._get_close_pairs(distance_cutoff, knn_cutoff=knn_cutoff)

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

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PLInteractionFeaturizer("
            f"protein_heavy={self.num_protein_atoms}, "
            f"ligand_heavy={self.num_ligand_atoms}, "
            f"cutoff={self.distance_cutoff}\u00c5, "
            f"feature_dim={self._get_edge_feature_dim()})"
        )
