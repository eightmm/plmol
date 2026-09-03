"""
Unified ligand featurizer for small-molecule feature extraction.

This module provides ligand-level features (descriptors and fingerprints)
and graph-level features (node and edge features) from RDKit mol objects or SMILES.
"""

import logging
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import numpy as np

from ..arrays import FLOAT
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem import FilterCatalog as _FilterCatalogModule

from ..errors import InputError
from .fingerprint_generator import FingerprintGenerator
from .graph import MoleculeGraphFeaturizer
from ..rdkit_utils import (
    canonicalize_mol,
    prepare_mol,
    ensure_3d_conformer,
    has_3d,
)

# Compiled once; SMARTS parsing is not free at descriptor-call rates.
_AMIDE_PATTERN = Chem.MolFromSmarts('[C](=[O])[N]')


class MoleculeFeaturizer:
    """
    Unified ligand featurizer with efficient caching.

    Supports two usage patterns:

    1. Object-oriented (recommended for repeated access):
        >>> featurizer = MoleculeFeaturizer("CCO")
        >>> features = featurizer.get_features()
        >>> node, edge, adj = featurizer.get_graph()

    2. Functional (for one-off extraction):
        >>> featurizer = MoleculeFeaturizer()
        >>> features = featurizer.get_features("CCO")
        >>> node, edge, adj = featurizer.get_graph("CCO")

    Examples:
        >>> # From SMILES
        >>> featurizer = MoleculeFeaturizer("CCO")
        >>> features = featurizer.get_features()
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
        >>>     features = featurizer.get_features()
    """

    _DESCRIPTOR_NAMES = (
        'mw', 'logp', 'tpsa', 'n_rotatable_bonds', 'flexibility',
        'hbd', 'hba', 'n_atoms', 'n_bonds', 'n_rings', 'n_aromatic_rings',
        'heteroatom_ratio', 'balaban_j', 'bertz_ct', 'chi0', 'chi1',
        'hall_kier_alpha', 'kappa1', 'kappa2', 'kappa3', 'mol_mr',
        'labute_asa', 'num_radical_electrons', 'num_valence_electrons',
        'num_saturated_rings', 'num_aliphatic_rings', 'num_saturated_heterocycles',
        'num_aliphatic_heterocycles', 'num_aromatic_heterocycles',
        'num_heteroatoms', 'formal_charge', 'chi0n',
        'lipinski_violations', 'passes_lipinski', 'qed', 'num_heavy_atoms', 'frac_csp3',
        'n_atom_rings', 'max_ring_size', 'avg_ring_size',
        'max_partial_charge', 'min_partial_charge',
        'max_abs_partial_charge', 'min_abs_partial_charge',
        'veber_violations', 'ghose_violations', 'egan_violations',
        'muegge_violations', 'pfizer_375_alert', 'gsk_4400_pass',
        'pains_alert_count', 'brenk_alert_count',
        'num_amide_bonds', 'num_stereocenters',
        'num_spiro_atoms', 'num_bridgehead_atoms',
        'esol_logs',
        'npr1', 'npr2', 'asphericity', 'eccentricity', 'radius_of_gyration',
    )

    @classmethod
    def descriptor_names(cls) -> Tuple[str, ...]:
        """Return descriptor names in the same order as the descriptor vector."""
        return cls._DESCRIPTOR_NAMES

    def __init__(
        self,
        mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
        add_hs: bool = False,
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
        self.add_hs = add_hs
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
                raise InputError(f"Invalid SMILES: {mol_or_smiles}")
        else:
            self.input_mol = mol_or_smiles
            self.input_smiles = Chem.MolToSmiles(mol_or_smiles) if mol_or_smiles else None

        # Prepare molecule (canonicalize and add hydrogens if requested)
        self._mol = self._prepare_mol(self.input_mol, self.add_hs, self.canonicalize)
        if self._mol is None:
            raise InputError(f"Failed to prepare molecule: {mol_or_smiles}")

        # Cache basic info
        self.num_atoms = self._mol.GetNumAtoms()
        self.num_bonds = self._mol.GetNumBonds()
        self.num_rings = self._mol.GetRingInfo().NumRings()
        self.has_3d = has_3d(self._mol)

    # =========================================================================
    # Molecule Preparation
    # =========================================================================

    @staticmethod
    def _canonicalize_mol(mol: Chem.Mol) -> Chem.Mol:
        """Reorder atoms to canonical order."""
        return canonicalize_mol(mol)

    @staticmethod
    def _prepare_mol(
        mol_or_smiles: Union[str, Chem.Mol],
        add_hs: bool = False,
        canonicalize: bool = True
    ) -> Chem.Mol:
        """Prepare molecule from SMILES string or RDKit mol object."""
        return prepare_mol(mol_or_smiles, add_hs=add_hs, canonicalize=canonicalize)

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

        features['n_atom_rings'] = min(len(atom_rings) / 8.0, 1.0)

        ring_sizes = [len(ring) for ring in atom_rings]
        if ring_sizes:
            features['max_ring_size'] = min(max(ring_sizes) / 12.0, 1.0)
            features['avg_ring_size'] = min(np.mean(ring_sizes) / 8.0, 1.0)
        else:
            features['max_ring_size'] = 0.0
            features['avg_ring_size'] = 0.0

        return features

    # =========================================================================
    # ADMET & Pharmacokinetic Features
    # =========================================================================

    @classmethod
    def _get_filter_catalogs(cls):
        """Lazily initialize and cache PAINS/Brenk FilterCatalogs."""
        if cls._PAINS_CATALOG is None:
            params = _FilterCatalogModule.FilterCatalogParams()
            params.AddCatalog(
                _FilterCatalogModule.FilterCatalogParams.FilterCatalogs.PAINS
            )
            cls._PAINS_CATALOG = _FilterCatalogModule.FilterCatalog(params)
        if cls._BRENK_CATALOG is None:
            params = _FilterCatalogModule.FilterCatalogParams()
            params.AddCatalog(
                _FilterCatalogModule.FilterCatalogParams.FilterCatalogs.BRENK
            )
            cls._BRENK_CATALOG = _FilterCatalogModule.FilterCatalog(params)
        return cls._PAINS_CATALOG, cls._BRENK_CATALOG

    @staticmethod
    def _ensure_3d_conformer(
        mol: Chem.Mol,
        random_seed: int = 42,
        optimize: bool = True,
    ) -> Optional[Chem.Mol]:
        """Return molecule with 3D conformer, generating one if needed."""
        return ensure_3d_conformer(mol, random_seed=random_seed, optimize=optimize)

    def get_admet_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract ADMET-relevant features from molecule.

        Returns 22 normalized descriptors covering charge distribution,
        ADMET filter scores, structural alerts, structural complexity,
        estimated solubility, and 3D shape.

        Returns:
            Dictionary of 22 ADMET descriptors (all normalized to ~[0, 1])
        """
        features: Dict[str, float] = {}

        # --- Charge distribution (4) ---
        mol_copy = Chem.RWMol(mol).GetMol()
        AllChem.ComputeGasteigerCharges(mol_copy)
        charges = []
        for atom in mol_copy.GetAtoms():
            try:
                charge = atom.GetDoubleProp('_GasteigerCharge')
            except KeyError:
                continue
            if not (np.isnan(charge) or np.isinf(charge)):
                charges.append(charge)
        if charges:
            features['max_partial_charge'] = (max(charges) + 1.0) / 2.0
            features['min_partial_charge'] = (min(charges) + 1.0) / 2.0
            abs_charges = [abs(c) for c in charges]
            features['max_abs_partial_charge'] = min(max(abs_charges), 1.0)
            features['min_abs_partial_charge'] = min(min(abs_charges), 1.0)
        else:
            features['max_partial_charge'] = 0.5
            features['min_partial_charge'] = 0.5
            features['max_abs_partial_charge'] = 0.0
            features['min_abs_partial_charge'] = 0.0

        # --- ADMET filter scores (6) ---
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        mr = Descriptors.MolMR(mol)
        n_atoms = mol.GetNumAtoms()
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        n_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)

        # Veber: TPSA <= 140, RotBonds <= 10
        features['veber_violations'] = sum([tpsa > 140, n_rot > 10]) / 2.0
        # Ghose: 160<=MW<=480, -0.4<=LogP<=5.6, 40<=MR<=130, 20<=atoms<=70
        features['ghose_violations'] = sum([
            mw < 160 or mw > 480, logp < -0.4 or logp > 5.6,
            mr < 40 or mr > 130, n_atoms < 20 or n_atoms > 70,
        ]) / 4.0
        # Egan: LogP <= 5.88, TPSA <= 131.6
        features['egan_violations'] = sum([logp > 5.88, tpsa > 131.6]) / 2.0
        # Muegge: 200<=MW<=600, -2<=LogP<=5, TPSA<=150, rings<=7, C>4, het>1, rot<=15, HBA<=10, HBD<=5
        features['muegge_violations'] = sum([
            mw < 200 or mw > 600, logp < -2 or logp > 5,
            tpsa > 150, n_rings > 7, n_carbons <= 4,
            n_hetero <= 1, n_rot > 15, hba > 10, hbd > 5,
        ]) / 9.0
        # Pfizer 3/75: LogP > 3 and TPSA < 75 (toxic risk)
        features['pfizer_375_alert'] = 1.0 if (logp > 3 and tpsa < 75) else 0.0
        # GSK 4/400: MW <= 400 and LogP <= 4 (favorable)
        features['gsk_4400_pass'] = 1.0 if (mw <= 400 and logp <= 4) else 0.0

        # --- Structural alerts (2) ---
        pains_cat, brenk_cat = self._get_filter_catalogs()
        features['pains_alert_count'] = min(len(pains_cat.GetMatches(mol)) / 5.0, 1.0)
        features['brenk_alert_count'] = min(len(brenk_cat.GetMatches(mol)) / 5.0, 1.0)

        # --- Structural complexity (4) ---
        n_amides = len(mol.GetSubstructMatches(_AMIDE_PATTERN)) if _AMIDE_PATTERN else 0
        features['num_amide_bonds'] = min(n_amides / 10.0, 1.0)
        features['num_stereocenters'] = min(
            rdMolDescriptors.CalcNumAtomStereoCenters(mol) / 10.0, 1.0
        )
        features['num_spiro_atoms'] = min(
            rdMolDescriptors.CalcNumSpiroAtoms(mol) / 5.0, 1.0
        )
        features['num_bridgehead_atoms'] = min(
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol) / 5.0, 1.0
        )

        # --- Solubility estimate (1) ---
        # ESOL (Delaney) equation
        aromatic_count = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        ap = aromatic_count / n_atoms if n_atoms > 0 else 0.0
        esol_logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * n_rot - 0.74 * ap
        features['esol_logs'] = np.clip((esol_logs + 10.0) / 20.0, 0.0, 1.0)

        # --- 3D shape descriptors (5) ---
        mol_3d = self._ensure_3d_conformer(mol)
        if mol_3d is not None and mol_3d.GetNumConformers() > 0:
            try:
                features['npr1'] = rdMolDescriptors.CalcNPR1(mol_3d)
                features['npr2'] = rdMolDescriptors.CalcNPR2(mol_3d)
                features['asphericity'] = min(
                    rdMolDescriptors.CalcAsphericity(mol_3d), 1.0
                )
                features['eccentricity'] = min(
                    rdMolDescriptors.CalcEccentricity(mol_3d), 1.0
                )
                features['radius_of_gyration'] = min(
                    rdMolDescriptors.CalcRadiusOfGyration(mol_3d) / 10.0, 1.0
                )
            except (RuntimeError, ValueError):
                for k in ('npr1', 'npr2', 'asphericity', 'eccentricity', 'radius_of_gyration'):
                    features[k] = 0.0
        else:
            for k in ('npr1', 'npr2', 'asphericity', 'eccentricity', 'radius_of_gyration'):
                features[k] = 0.0

        return features

    # =========================================================================
    # Fingerprint Features (delegated to FingerprintGenerator)
    # =========================================================================

    _fp_generator = FingerprintGenerator()

    @classmethod
    def _get_fp_generators(cls) -> Dict[str, Any]:
        """Lazily initialize and cache RDKit fingerprint generators."""
        return FingerprintGenerator._get_fp_generators()

    @classmethod
    def _normalize_include_fps(
        cls, include_fps: Optional[Iterable[str]]
    ) -> Tuple[str, ...]:
        """Normalize/validate fingerprint selection."""
        return FingerprintGenerator._normalize_include_fps(include_fps)

    def get_fingerprints(
        self,
        mol: Chem.Mol,
        include_fps: Optional[Iterable[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract various molecular fingerprints.

        Returns:
            Dictionary of fingerprint tensors
        """
        return self._fp_generator.get_fingerprints(mol, include_fps=include_fps)

    # =========================================================================
    # Custom SMARTS Features
    # =========================================================================

    def _get_custom_smarts_features(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """
        Compute custom SMARTS pattern matches for each atom.

        Returns:
            np.ndarray or None: Binary features [n_atoms, n_patterns] if patterns provided
        """
        if not self.custom_smarts:
            return None

        num_atoms = mol.GetNumAtoms()
        num_patterns = len(self.custom_smarts)
        features = np.zeros((num_atoms, num_patterns), dtype=FLOAT)

        for idx, (name, pattern) in enumerate(self.custom_smarts.items()):
            try:
                # Parse SMARTS pattern
                smart_mol = Chem.MolFromSmarts(pattern)
                if smart_mol is None:
                    logger.warning("Invalid SMARTS pattern for '%s': %s", name, pattern)
                    continue

                # Find matches
                matches = mol.GetSubstructMatches(smart_mol)

                # Mark matched atoms
                for match in matches:
                    for atom_idx in match:
                        if atom_idx < num_atoms:
                            features[atom_idx, idx] = 1.0

            except Exception as e:
                logger.warning("Error processing SMARTS pattern '%s': %s", name, e)

        return features

    # =========================================================================
    # Main Feature Extraction Methods
    # =========================================================================

    def get_features(
        self,
        mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
        add_hs: bool = False,
        include_fps: Optional[Iterable[str]] = None,
    ) -> Dict:
        """
        Extract all molecular-level features including descriptors and fingerprints.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string.
                          If None, uses the molecule set during initialization.
            add_hs: Whether to add hydrogens (ignored if using cached molecule, default: False)

        Returns:
            Dictionary containing:
                - descriptors: Tensor of molecular descriptors [62]
                - maccs: MACCS fingerprint [167]
                - ecfp4: ECFP4 fingerprint [2048]
                - ecfp4_count: ECFP4 count fingerprint [2048]
                - ecfp4_feature: ECFP4 feature-invariant fingerprint [2048]
                - ecfp6: ECFP6 fingerprint [2048]
                - ecfp6_feature: ECFP6 feature-invariant fingerprint [2048]
                - rdkit: RDKit fingerprint [2048]
                - atom_pair: Atom pair fingerprint [2048]
                - topological_torsion: Topological torsion fingerprint [2048]
                - pharmacophore2d: 2D pharmacophore fingerprint [1024]
                - avalon: Avalon fingerprint [2048]
                - erg: ErG pharmacophore fingerprint [315]
        """
        # Object-oriented mode with caching
        if mol_or_smiles is None:
            if self._mol is None:
                raise InputError("No molecule provided. Either initialize with a molecule or pass one to this method.")
            include_key = self._normalize_include_fps(include_fps)
            cache_key = ('features', include_key)
            if cache_key not in self._cache:
                self._cache[cache_key] = self._compute_features(self._mol, include_fps=include_key)
            return self._cache[cache_key]

        # Functional mode (no caching)
        mol = self._prepare_mol(mol_or_smiles, add_hs, self.canonicalize)
        include_key = self._normalize_include_fps(include_fps)
        return self._compute_features(mol, include_fps=include_key)

    def _compute_descriptor_tensor(self, mol: Chem.Mol) -> np.ndarray:
        """Build the 62-dim descriptor tensor for a prepared molecule."""
        physicochemical = self.get_physicochemical_features(mol)
        druglike = self.get_druglike_features(mol)
        structural = self.get_structural_features(mol)
        admet = self.get_admet_features(mol)

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
        structural_keys = ['n_atom_rings', 'max_ring_size', 'avg_ring_size']
        admet_keys = [
            'max_partial_charge', 'min_partial_charge',
            'max_abs_partial_charge', 'min_abs_partial_charge',
            'veber_violations', 'ghose_violations', 'egan_violations',
            'muegge_violations', 'pfizer_375_alert', 'gsk_4400_pass',
            'pains_alert_count', 'brenk_alert_count',
            'num_amide_bonds', 'num_stereocenters',
            'num_spiro_atoms', 'num_bridgehead_atoms',
            'esol_logs',
            'npr1', 'npr2', 'asphericity', 'eccentricity', 'radius_of_gyration',
        ]

        descriptors = np.asarray(
            [float(physicochemical[key]) for key in descriptor_keys] +
            [float(druglike[key]) for key in druglike_keys] +
            [float(structural[key]) for key in structural_keys] +
            [float(admet[key]) for key in admet_keys],
            dtype=np.float32,
        )
        return descriptors

    def _cached_descriptor_tensor(self) -> np.ndarray:
        """Descriptor tensor for the initialized molecule, computed once."""
        if 'descriptors' not in self._cache:
            self._cache['descriptors'] = self._compute_descriptor_tensor(self._mol)
        return self._cache['descriptors']

    def _compute_features(
        self,
        mol: Chem.Mol,
        include_fps: Optional[Iterable[str]] = None,
    ) -> Dict:
        """Compute all features for a prepared molecule."""
        if mol is self._mol:
            descriptors = self._cached_descriptor_tensor()
        else:
            descriptors = self._compute_descriptor_tensor(mol)

        include_order = self._normalize_include_fps(include_fps)
        include_set = set(include_order)
        fingerprints = self.get_fingerprints(mol, include_fps=include_set)

        out = {'descriptors': descriptors}
        for fp_name in include_order:
            if fp_name in fingerprints:
                out[fp_name] = fingerprints[fp_name]
        return out

    def featurize(
        self,
        mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
        add_hs: bool = False,
        distance_cutoff: Optional[float] = None,
        include_custom_smarts: bool = True,
        knn_cutoff: Optional[int] = None,
        generate_conformer: bool = True,
    ) -> Tuple[Dict, Dict, np.ndarray]:
        """
        New interface for featurization.
        """
        # Object-oriented mode with caching
        if mol_or_smiles is None:
            if self._mol is None:
                raise InputError("No molecule provided.")
            return self._compute_graph(self._mol, distance_cutoff, include_custom_smarts,
                                       knn_cutoff=knn_cutoff,
                                       generate_conformer=generate_conformer)

        # Functional mode
        mol = self._prepare_mol(mol_or_smiles, add_hs, self.canonicalize)
        return self._compute_graph(mol, distance_cutoff, include_custom_smarts,
                                   knn_cutoff=knn_cutoff,
                                   generate_conformer=generate_conformer)

    def get_graph(
        self,
        mol_or_smiles: Optional[Union[str, Chem.Mol]] = None,
        add_hs: bool = False,
        distance_cutoff: Optional[float] = None,
        include_custom_smarts: bool = True,
        knn_cutoff: Optional[int] = None,
        generate_conformer: bool = True,
    ) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Create molecular graph with node and edge features.
        """
        return self.featurize(mol_or_smiles, add_hs, distance_cutoff, include_custom_smarts,
                              knn_cutoff=knn_cutoff,
                              generate_conformer=generate_conformer)

    def _compute_graph(
        self,
        mol: Chem.Mol,
        distance_cutoff: Optional[float] = None,
        include_custom_smarts: bool = True,
        knn_cutoff: Optional[int] = None,
        generate_conformer: bool = True,
    ) -> Tuple[Dict, Dict, np.ndarray]:
        """Compute graph representation for a prepared molecule."""
        node, edge, adj = self._graph_featurizer.featurize(
            mol,
            distance_cutoff=distance_cutoff,
            knn_cutoff=knn_cutoff,
            generate_conformer=generate_conformer,
        )

        # Add custom SMARTS features if requested
        if include_custom_smarts and self.custom_smarts:
            custom_features = self._get_custom_smarts_features(mol)
            if custom_features is not None:
                original_feats = node['node_feats']
                node['node_feats'] = np.concatenate([original_feats, custom_features], axis=1)

        return node, edge, adj

    # =========================================================================
    # Convenience Methods (Object-Oriented API)
    # =========================================================================

    def get_descriptors(self) -> np.ndarray:
        """
        Get only molecular descriptors (requires molecule initialization).

        Returns:
            np.ndarray: 62 normalized molecular descriptors
        """
        if self._mol is None:
            raise InputError("No molecule provided. Either initialize with a molecule or pass one to this method.")
        return self._cached_descriptor_tensor()

    def get_morgan_fingerprint(self, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """
        Get Morgan fingerprint (requires molecule initialization).

        Args:
            radius: Radius for Morgan fingerprint (default: 2)
            n_bits: Number of bits (default: 2048)

        Returns:
            np.ndarray: Morgan fingerprint of shape [n_bits]
        """
        if self._mol is None:
            raise InputError("No molecule set. Initialize with a molecule first.")

        # Use cached value for default parameters
        if radius == 2 and n_bits == 2048:
            features = self.get_features()
            return features['ecfp4']

        # Generate custom fingerprint
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, countSimulation=True, includeChirality=True
        )
        return morgan_gen.GetFingerprintAsNumPy(self._mol).astype(FLOAT)

    def get_custom_smarts_features(self) -> Optional[Dict[str, Any]]:
        """
        Get custom SMARTS pattern matches for each atom.

        Returns:
            Dictionary with 'features' tensor and 'names' list, or None if no patterns
        """
        if not self.custom_smarts:
            return None

        if self._mol is None:
            raise InputError("No molecule set. Initialize with a molecule first.")

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

    def get_3d_coordinates(self) -> Optional[np.ndarray]:
        """
        Get 3D coordinates if available (requires molecule initialization).

        Returns:
            np.ndarray or None: 3D coordinates [n_atoms, 3]
        """
        if self._mol is None:
            raise InputError("No molecule set. Initialize with a molecule first.")

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
            raise InputError("No molecule set. Initialize with a molecule first.")

        features = self.get_features()
        node, edge, adj = self.get_graph()

        all_features = {
            'descriptors': features['descriptors'],
            'fingerprints': {k: v for k, v in features.items() if k != 'descriptors'},
            'graph': {'node': node, 'edge': edge, 'adj': adj},
            'metadata': {
                'input_smiles': self.input_smiles,
                'num_atoms': self.num_atoms,
                'num_bonds': self.num_bonds,
                'num_rings': self.num_rings,
                'has_3d': self.has_3d,
                'hydrogens_added': self.add_hs
            }
        }

        if save_to:
            # numpy arrays in a nested dict; np.load(path, allow_pickle=True) reads it.
            np.save(save_to, all_features, allow_pickle=True)

        return all_features

    def get_rdkit_mol(self) -> Chem.Mol:
        """
        Get the prepared RDKit molecule (requires molecule initialization).

        Returns:
            RDKit mol object used internally for featurization.
        """
        if self._mol is None:
            raise InputError("No molecule set. Initialize with a molecule first.")
        return self._mol

    # Aliases for consistency
    extract = get_all_features
    get_feature = get_features

    def __repr__(self) -> str:
        """String representation."""
        if self._mol is not None:
            return (f"MoleculeFeaturizer(smiles='{self.input_smiles}', "
                    f"atoms={self.num_atoms}, bonds={self.num_bonds})")
        return "MoleculeFeaturizer(no molecule set)"
    _PAINS_CATALOG = None
    _BRENK_CATALOG = None
    _SUPPORTED_FP_NAMES = FingerprintGenerator._SUPPORTED_FP_NAMES
    _FP_KEY_ALIASES = FingerprintGenerator._FP_KEY_ALIASES
