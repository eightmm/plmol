"""
Unified molecule featurizer for molecular feature extraction.

This module provides molecular-level features (descriptors and fingerprints)
and graph-level features (node and edge features) from RDKit mol objects or SMILES.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from .graph_featurizer import MoleculeGraphFeaturizer


class MoleculeFeaturizer:
    """
    Unified molecule featurizer with efficient caching.

    Supports two usage patterns:

    1. Object-oriented (recommended for repeated access):
        >>> featurizer = MoleculeFeaturizer("CCO")
        >>> features = featurizer.get_feature()
        >>> node, edge, adj = featurizer.get_graph()

    2. Functional (for one-off extraction):
        >>> featurizer = MoleculeFeaturizer()
        >>> features = featurizer.get_feature("CCO")
        >>> node, edge, adj = featurizer.get_graph("CCO")

    Examples:
        >>> # From SMILES
        >>> featurizer = MoleculeFeaturizer("CCO")
        >>> features = featurizer.get_feature()
        >>> descriptors = featurizer.get_descriptors()
        >>>
        >>> # From RDKit mol
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> featurizer = MoleculeFeaturizer(mol)
        >>>
        >>> # From SDF file
        >>> suppl = Chem.SDMolSupplier('molecules.sdf')
        >>> for mol in suppl:
        >>>     featurizer = MoleculeFeaturizer(mol)
        >>>     features = featurizer.get_feature()
    """

    def __init__(
        self,
        mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
        hydrogen: bool = False,
        canonicalize: bool = True,
        custom_smarts: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the molecule featurizer.

        Args:
            mol_or_smiles: Optional molecule (RDKit mol or SMILES string).
                          If provided, enables object-oriented usage with caching.
                          If None, use functional API by passing molecule to methods.
            hydrogen: Whether to add explicit hydrogens to molecules (default: False)
                     Heavy atoms only is recommended for GNN models since H count
                     is already encoded in node features (total_hs).
            canonicalize: Whether to reorder atoms to canonical order (default: True)
                         Ensures same molecule always produces same features regardless
                         of input atom ordering. Recommended for ML consistency.
            custom_smarts: Optional dictionary of custom SMARTS patterns
                          e.g., {'aromatic_nitrogen': 'n', 'carboxyl': 'C(=O)O'}

        Raises:
            ValueError: If molecule cannot be parsed
        """
        self._graph_featurizer = MoleculeGraphFeaturizer()
        self.hydrogen = hydrogen
        self.canonicalize = canonicalize
        self.custom_smarts = custom_smarts or {}
        self._cache: Dict[str, Any] = {}

        # Object-oriented mode: initialize with molecule
        self._mol: Optional[Chem.Mol] = None
        self.input_smiles: Optional[str] = None
        self.input_mol: Optional[Chem.Mol] = None
        self.num_atoms: int = 0
        self.num_bonds: int = 0
        self.num_rings: int = 0
        self.has_3d: bool = False

        if mol_or_smiles is not None:
            self._init_molecule(mol_or_smiles)

    def _init_molecule(self, mol_or_smiles: Union[str, Chem.Mol]) -> None:
        """Initialize with a specific molecule for object-oriented usage."""
        # Store input for reference
        if isinstance(mol_or_smiles, str):
            self.input_smiles = mol_or_smiles
            self.input_mol = Chem.MolFromSmiles(mol_or_smiles)
            if self.input_mol is None:
                raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
        else:
            self.input_mol = mol_or_smiles
            self.input_smiles = Chem.MolToSmiles(mol_or_smiles) if mol_or_smiles else None

        # Prepare molecule (canonicalize and add hydrogens if requested)
        self._mol = self._prepare_mol(self.input_mol, self.hydrogen, self.canonicalize)
        if self._mol is None:
            raise ValueError(f"Failed to prepare molecule: {mol_or_smiles}")

        # Cache basic info
        self.num_atoms = self._mol.GetNumAtoms()
        self.num_bonds = self._mol.GetNumBonds()
        self.num_rings = self._mol.GetRingInfo().NumRings()
        self.has_3d = self._mol.GetNumConformers() > 0

    # =========================================================================
    # Molecule Preparation
    # =========================================================================

    @staticmethod
    def _canonicalize_mol(mol: Chem.Mol) -> Chem.Mol:
        """
        Reorder atoms to canonical order.

        This ensures the same molecule always produces the same atom ordering
        regardless of the input order. Coordinates are also reordered.

        Args:
            mol: RDKit mol object

        Returns:
            RDKit mol object with atoms in canonical order
        """
        # Get canonical ranking for each atom
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
        # Convert ranks to new atom order: newOrder[new_idx] = old_idx
        # ranks[old_idx] = new_idx, so we need inverse
        new_order = [0] * len(ranks)
        for old_idx, new_idx in enumerate(ranks):
            new_order[new_idx] = old_idx
        # Renumber atoms (this also reorders coordinates if present)
        return Chem.RenumberAtoms(mol, new_order)

    @staticmethod
    def _prepare_mol(
        mol_or_smiles: Union[str, Chem.Mol],
        add_hs: bool = True,
        canonicalize: bool = True
    ) -> Chem.Mol:
        """
        Prepare molecule from SMILES string or RDKit mol object.

        Preserves 3D coordinates when adding hydrogens if the molecule has a conformer.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string
            add_hs: Whether to add hydrogens
            canonicalize: Whether to reorder atoms to canonical order

        Returns:
            RDKit mol object with optional hydrogens in canonical order

        Raises:
            ValueError: If SMILES string is invalid
        """
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
        else:
            mol = mol_or_smiles

        # Canonicalize BEFORE adding hydrogens for consistent ordering
        if canonicalize and mol is not None:
            mol = MoleculeFeaturizer._canonicalize_mol(mol)

        if add_hs and mol is not None:
            has_3d_coords = mol.GetNumConformers() > 0
            if has_3d_coords:
                mol = Chem.AddHs(mol, addCoords=True)
            else:
                mol = Chem.AddHs(mol)

        return mol

    # =========================================================================
    # Molecular Descriptor Features
    # =========================================================================

    def get_physicochemical_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract physicochemical features from molecule.

        Returns:
            Dictionary of normalized physicochemical descriptors
        """
        features = {}

        # Basic properties
        features['mw'] = min(Descriptors.MolWt(mol) / 1000.0, 1.0)
        features['logp'] = (Descriptors.MolLogP(mol) + 5) / 10.0
        features['tpsa'] = min(Descriptors.TPSA(mol) / 200.0, 1.0)

        # Flexibility
        n_bonds = mol.GetNumBonds()
        features['n_rotatable_bonds'] = min(rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0, 1.0)
        features['flexibility'] = min(
            rdMolDescriptors.CalcNumRotatableBonds(mol) / n_bonds if n_bonds > 0 else 0, 1.0
        )

        # H-bonding
        features['hbd'] = min(rdMolDescriptors.CalcNumHBD(mol) / 10.0, 1.0)
        features['hba'] = min(rdMolDescriptors.CalcNumHBA(mol) / 15.0, 1.0)

        # Atom/bond counts
        features['n_atoms'] = min(mol.GetNumAtoms() / 100.0, 1.0)
        features['n_bonds'] = min(n_bonds / 120.0, 1.0)
        features['n_rings'] = min(rdMolDescriptors.CalcNumRings(mol) / 10.0, 1.0)
        features['n_aromatic_rings'] = min(rdMolDescriptors.CalcNumAromaticRings(mol) / 8.0, 1.0)

        # Heteroatom ratio
        n_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        features['heteroatom_ratio'] = n_heteroatoms / mol.GetNumAtoms()

        # Topological indices
        features['balaban_j'] = min(Descriptors.BalabanJ(mol) / 5.0, 1.0)
        features['bertz_ct'] = min(Descriptors.BertzCT(mol) / 2000.0, 1.0)
        features['chi0'] = min(Descriptors.Chi0(mol) / 50.0, 1.0)
        features['chi1'] = min(Descriptors.Chi1(mol) / 30.0, 1.0)
        features['chi0n'] = min(Descriptors.Chi0n(mol) / 50.0, 1.0)

        hka = Descriptors.HallKierAlpha(mol)
        features['hall_kier_alpha'] = min(abs(hka) / 5.0, 1.0) if hka != -1 else 0.0

        features['kappa1'] = min(Descriptors.Kappa1(mol) / 50.0, 1.0)
        features['kappa2'] = min(Descriptors.Kappa2(mol) / 20.0, 1.0)
        features['kappa3'] = min(Descriptors.Kappa3(mol) / 10.0, 1.0)

        # Electronic properties
        features['mol_mr'] = min(Descriptors.MolMR(mol) / 200.0, 1.0)
        features['labute_asa'] = min(Descriptors.LabuteASA(mol) / 500.0, 1.0)
        features['num_radical_electrons'] = min(Descriptors.NumRadicalElectrons(mol) / 5.0, 1.0)
        features['num_valence_electrons'] = min(Descriptors.NumValenceElectrons(mol) / 500.0, 1.0)

        # Ring complexity
        features['num_saturated_rings'] = min(rdMolDescriptors.CalcNumSaturatedRings(mol) / 10.0, 1.0)
        features['num_aliphatic_rings'] = min(rdMolDescriptors.CalcNumAliphaticRings(mol) / 10.0, 1.0)
        features['num_saturated_heterocycles'] = min(
            rdMolDescriptors.CalcNumSaturatedHeterocycles(mol) / 8.0, 1.0
        )
        features['num_aliphatic_heterocycles'] = min(
            rdMolDescriptors.CalcNumAliphaticHeterocycles(mol) / 8.0, 1.0
        )
        features['num_aromatic_heterocycles'] = min(
            rdMolDescriptors.CalcNumAromaticHeterocycles(mol) / 8.0, 1.0
        )

        # Atom counts
        features['num_heteroatoms'] = min(rdMolDescriptors.CalcNumHeteroatoms(mol) / 30.0, 1.0)
        total_formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        features['formal_charge'] = (total_formal_charge + 5) / 10.0

        return features

    def get_druglike_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract drug-likeness features from molecule.

        Returns:
            Dictionary of drug-likeness descriptors
        """
        features = {}

        # Lipinski's Rule of Five
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        features['lipinski_violations'] = violations / 4.0
        features['passes_lipinski'] = 1.0 if violations == 0 else 0.0

        # QED score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features['qed'] = QED.qed(mol)

        # Other drug-like properties
        features['num_heavy_atoms'] = min(mol.GetNumHeavyAtoms() / 50.0, 1.0)

        csp3_count = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetHybridization() == Chem.HybridizationType.SP3 and atom.GetAtomicNum() == 6
        )
        total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        features['frac_csp3'] = csp3_count / total_carbons if total_carbons > 0 else 0.0

        return features

    def get_structural_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract structural features from molecule.

        Returns:
            Dictionary of structural descriptors
        """
        features = {}

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        features['n_ring_systems'] = min(len(atom_rings) / 8.0, 1.0)

        ring_sizes = [len(ring) for ring in atom_rings]
        if ring_sizes:
            features['max_ring_size'] = min(max(ring_sizes) / 12.0, 1.0)
            features['avg_ring_size'] = min(np.mean(ring_sizes) / 8.0, 1.0)
        else:
            features['max_ring_size'] = 0.0
            features['avg_ring_size'] = 0.0

        return features

    # =========================================================================
    # Fingerprint Features
    # =========================================================================

    def get_fingerprints(self, mol: Chem.Mol) -> Dict[str, torch.Tensor]:
        """
        Extract various molecular fingerprints.

        Returns:
            Dictionary of fingerprint tensors
        """
        fingerprints = {}

        # MACCS keys
        fingerprints['maccs'] = torch.tensor(
            MACCSkeys.GenMACCSKeys(mol).ToList(), dtype=torch.float32
        )

        # Morgan fingerprints
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, countSimulation=True, includeChirality=True
        )
        fingerprints['morgan'] = torch.from_numpy(morgan_gen.GetFingerprintAsNumPy(mol)).float()
        fingerprints['morgan_count'] = torch.from_numpy(
            morgan_gen.GetCountFingerprintAsNumPy(mol)
        ).float()

        # Feature Morgan
        feature_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
            countSimulation=True,
        )
        fingerprints['feature_morgan'] = torch.from_numpy(
            feature_morgan_gen.GetFingerprintAsNumPy(mol)
        ).float()

        # RDKit fingerprint
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=1, maxPath=7, fpSize=2048, countSimulation=True,
            branchedPaths=True, useBondOrder=True,
        )
        fingerprints['rdkit'] = torch.from_numpy(rdkit_gen.GetFingerprintAsNumPy(mol)).float()

        # Atom pair fingerprint
        ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=1, maxDistance=8, fpSize=2048, countSimulation=True
        )
        fingerprints['atom_pair'] = torch.from_numpy(ap_gen.GetFingerprintAsNumPy(mol)).float()

        # Topological torsion
        tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            torsionAtomCount=4, fpSize=2048, countSimulation=True
        )
        fingerprints['topological_torsion'] = torch.from_numpy(
            tt_gen.GetFingerprintAsNumPy(mol)
        ).float()

        # Pharmacophore 2D
        pharm_fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        bit_vector = torch.zeros(1024)
        for bit_id in pharm_fp.GetOnBits():
            if bit_id < 1024:
                bit_vector[bit_id] = 1.0
        fingerprints['pharmacophore2d'] = bit_vector.float()

        return fingerprints

    # =========================================================================
    # Custom SMARTS Features
    # =========================================================================

    def _get_custom_smarts_features(self, mol: Chem.Mol) -> Optional[torch.Tensor]:
        """
        Compute custom SMARTS pattern matches for each atom.

        Returns:
            torch.Tensor or None: Binary features [n_atoms, n_patterns] if patterns provided
        """
        if not self.custom_smarts:
            return None

        num_atoms = mol.GetNumAtoms()
        num_patterns = len(self.custom_smarts)
        features = torch.zeros(num_atoms, num_patterns)

        for idx, (name, pattern) in enumerate(self.custom_smarts.items()):
            try:
                # Parse SMARTS pattern
                smart_mol = Chem.MolFromSmarts(pattern)
                if smart_mol is None:
                    print(f"Warning: Invalid SMARTS pattern for '{name}': {pattern}")
                    continue

                # Find matches
                matches = mol.GetSubstructMatches(smart_mol)

                # Mark matched atoms
                for match in matches:
                    for atom_idx in match:
                        if atom_idx < num_atoms:
                            features[atom_idx, idx] = 1.0

            except Exception as e:
                print(f"Warning: Error processing SMARTS pattern '{name}': {e}")

        return features

    # =========================================================================
    # Main Feature Extraction Methods
    # =========================================================================

    def get_feature(
        self, mol_or_smiles: Optional[Union[str, Chem.Mol]] = None, add_hs: bool = False
    ) -> Dict:
        """
        Extract all molecular-level features including descriptors and fingerprints.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string.
                          If None, uses the molecule set during initialization.
            add_hs: Whether to add hydrogens (ignored if using cached molecule, default: False)

        Returns:
            Dictionary containing:
                - descriptor: Tensor of molecular descriptors [40]
                - maccs: MACCS fingerprint [167]
                - morgan: Morgan fingerprint [2048]
                - morgan_count: Morgan count fingerprint [2048]
                - feature_morgan: Feature Morgan fingerprint [2048]
                - rdkit: RDKit fingerprint [2048]
                - atom_pair: Atom pair fingerprint [2048]
                - topological_torsion: Topological torsion fingerprint [2048]
                - pharmacophore2d: 2D pharmacophore fingerprint [1024]
        """
        # Object-oriented mode with caching
        if mol_or_smiles is None:
            if self._mol is None:
                raise ValueError("No molecule provided. Either initialize with a molecule or pass one to this method.")
            if 'features' not in self._cache:
                self._cache['features'] = self._compute_features(self._mol)
            return self._cache['features']

        # Functional mode (no caching)
        mol = self._prepare_mol(mol_or_smiles, add_hs, self.canonicalize)
        return self._compute_features(mol)

    def _compute_features(self, mol: Chem.Mol) -> Dict:
        """Compute all features for a prepared molecule."""
        physicochemical = self.get_physicochemical_features(mol)
        druglike = self.get_druglike_features(mol)
        structural = self.get_structural_features(mol)

        # Build descriptor tensor
        descriptor_keys = [
            'mw', 'logp', 'tpsa', 'n_rotatable_bonds', 'flexibility',
            'hbd', 'hba', 'n_atoms', 'n_bonds', 'n_rings', 'n_aromatic_rings',
            'heteroatom_ratio', 'balaban_j', 'bertz_ct', 'chi0', 'chi1',
            'hall_kier_alpha', 'kappa1', 'kappa2', 'kappa3', 'mol_mr',
            'labute_asa', 'num_radical_electrons', 'num_valence_electrons',
            'num_saturated_rings', 'num_aliphatic_rings', 'num_saturated_heterocycles',
            'num_aliphatic_heterocycles', 'num_aromatic_heterocycles',
            'num_heteroatoms', 'formal_charge', 'chi0n',
        ]
        druglike_keys = [
            'lipinski_violations', 'passes_lipinski', 'qed', 'num_heavy_atoms', 'frac_csp3'
        ]
        structural_keys = ['n_ring_systems', 'max_ring_size', 'avg_ring_size']

        descriptors = []
        for key in descriptor_keys:
            descriptors.append(float(physicochemical[key]))
        for key in druglike_keys:
            descriptors.append(float(druglike[key]))
        for key in structural_keys:
            descriptors.append(float(structural[key]))

        fingerprints = self.get_fingerprints(mol)

        return {
            'descriptor': torch.tensor(descriptors, dtype=torch.float32),
            'maccs': fingerprints['maccs'],
            'morgan': fingerprints['morgan'],
            'morgan_count': fingerprints['morgan_count'],
            'feature_morgan': fingerprints['feature_morgan'],
            'rdkit': fingerprints['rdkit'],
            'atom_pair': fingerprints['atom_pair'],
            'topological_torsion': fingerprints['topological_torsion'],
            'pharmacophore2d': fingerprints['pharmacophore2d'],
        }

    def get_graph(
        self,
        mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
        add_hs: bool = False,
        distance_cutoff: Optional[float] = None,
        include_custom_smarts: bool = True,
    ) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        Create molecular graph with node and edge features.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string.
                          If None, uses the molecule set during initialization.
            add_hs: Whether to add hydrogens (ignored if using cached molecule, default: False)
            distance_cutoff: Optional distance cutoff for edges (if 3D available)
                           If None, uses bond connectivity
            include_custom_smarts: Whether to include custom SMARTS features in node features

        Returns:
            Tuple of (node_dict, edge_dict, adjacency_matrix):
                - node_dict: {'node_feats': [N, 157], 'coords': [N, 3]}
                - edge_dict: {'edges': [2, E], 'edge_feats': [E, 66]}
                - adjacency_matrix: [N, N, 66]
        """
        # Object-oriented mode with caching
        if mol_or_smiles is None:
            if self._mol is None:
                raise ValueError("No molecule provided. Either initialize with a molecule or pass one to this method.")

            cache_key = f'graph_{distance_cutoff}_{include_custom_smarts}'
            if cache_key not in self._cache:
                self._cache[cache_key] = self._compute_graph(
                    self._mol, distance_cutoff, include_custom_smarts
                )
            return self._cache[cache_key]

        # Functional mode (no caching)
        mol = self._prepare_mol(mol_or_smiles, add_hs, self.canonicalize)
        return self._compute_graph(mol, distance_cutoff, include_custom_smarts)

    def _compute_graph(
        self,
        mol: Chem.Mol,
        distance_cutoff: Optional[float] = None,
        include_custom_smarts: bool = True,
    ) -> Tuple[Dict, Dict, torch.Tensor]:
        """Compute graph representation for a prepared molecule."""
        node, edge, adj = self._graph_featurizer.featurize(mol)

        # Add custom SMARTS features if requested
        if include_custom_smarts and self.custom_smarts:
            custom_features = self._get_custom_smarts_features(mol)
            if custom_features is not None:
                original_feats = node['node_feats']
                node['node_feats'] = torch.cat([original_feats, custom_features], dim=1)

        # If distance cutoff specified and 3D coords available, filter edges
        has_3d = mol.GetNumConformers() > 0
        if distance_cutoff is not None and has_3d and 'coords' in node:
            from scipy.spatial import distance_matrix

            coords = node['coords'].numpy()
            dist_matrix = distance_matrix(coords, coords)

            # Create edges based on distance cutoff
            edges_array = np.where((dist_matrix < distance_cutoff) & (dist_matrix > 0))

            # Update edge information
            edge['edges'] = torch.tensor([edges_array[0], edges_array[1]])
            edge['distance_cutoff'] = distance_cutoff

        return node, edge, adj

    # =========================================================================
    # Convenience Methods (Object-Oriented API)
    # =========================================================================

    def get_descriptors(self) -> torch.Tensor:
        """
        Get only molecular descriptors (requires molecule initialization).

        Returns:
            torch.Tensor: 40 normalized molecular descriptors
        """
        features = self.get_feature()
        return features['descriptor']

    def get_morgan_fingerprint(self, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
        """
        Get Morgan fingerprint (requires molecule initialization).

        Args:
            radius: Radius for Morgan fingerprint (currently uses default 2)
            n_bits: Number of bits (currently uses default 2048)

        Returns:
            torch.Tensor: Morgan fingerprint
        """
        features = self.get_feature()
        return features['morgan']

    def get_custom_smarts_features(self) -> Optional[Dict[str, Any]]:
        """
        Get custom SMARTS pattern matches for each atom.

        Returns:
            Dictionary with 'features' tensor and 'names' list, or None if no patterns
        """
        if not self.custom_smarts:
            return None

        if self._mol is None:
            raise ValueError("No molecule set. Initialize with a molecule first.")

        cache_key = 'custom_smarts_result'
        if cache_key not in self._cache:
            features = self._get_custom_smarts_features(self._mol)
            if features is None:
                self._cache[cache_key] = None
            else:
                self._cache[cache_key] = {
                    'features': features,
                    'names': list(self.custom_smarts.keys()),
                    'patterns': self.custom_smarts
                }
        return self._cache[cache_key]

    def get_3d_coordinates(self) -> Optional[torch.Tensor]:
        """
        Get 3D coordinates if available (requires molecule initialization).

        Returns:
            torch.Tensor or None: 3D coordinates [n_atoms, 3]
        """
        if self._mol is None:
            raise ValueError("No molecule set. Initialize with a molecule first.")

        if not self.has_3d:
            return None

        node, _, _ = self.get_graph()
        return node.get('coords', None)

    def get_all_features(self, save_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all features at once (requires molecule initialization).

        Args:
            save_to: Optional path to save features

        Returns:
            Dictionary containing all features and metadata
        """
        if self._mol is None:
            raise ValueError("No molecule set. Initialize with a molecule first.")

        features = self.get_feature()
        node, edge, adj = self.get_graph()

        all_features = {
            'descriptors': features['descriptor'],
            'fingerprints': {k: v for k, v in features.items() if k != 'descriptor'},
            'graph': {'node': node, 'edge': edge, 'adj': adj},
            'metadata': {
                'input_smiles': self.input_smiles,
                'num_atoms': self.num_atoms,
                'num_bonds': self.num_bonds,
                'num_rings': self.num_rings,
                'has_3d': self.has_3d,
                'hydrogens_added': self.hydrogen
            }
        }

        if save_to:
            torch.save(all_features, save_to)

        return all_features

    # Aliases for consistency
    extract = get_all_features
    get_features = get_feature

    def __repr__(self) -> str:
        """String representation."""
        if self._mol is not None:
            return (f"MoleculeFeaturizer(smiles='{self.input_smiles}', "
                    f"atoms={self.num_atoms}, bonds={self.num_bonds})")
        return "MoleculeFeaturizer(no molecule set)"
