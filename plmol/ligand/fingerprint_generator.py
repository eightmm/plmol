"""
Fingerprint generation for small molecules.

Extracts various molecular fingerprints (ECFP, MACCS, RDKit, Avalon, ERG, etc.)
from RDKit mol objects. Lazy-imports optional RDKit submodules to reduce startup cost.
"""

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
    rdReducedGraphs,
)

logger = logging.getLogger(__name__)


class FingerprintGenerator:
    """
    Generates molecular fingerprints from RDKit mol objects.

    Supports ECFP4/6 (Morgan), MACCS, RDKit, AtomPair, TopologicalTorsion,
    Pharmacophore2D, Avalon, ErG, VSA descriptors, and MQN.

    Lazy-imports optional dependencies (Avalon, Pharm2D) on first use.
    """

    _FP_GENERATORS: Optional[Dict[str, Any]] = None

    _SUPPORTED_FP_NAMES = (
        "maccs",
        "ecfp4",
        "ecfp4_feature",
        "ecfp6",
        "rdkit",
        "atom_pair",
        "topological_torsion",
        "erg",
    )

    _FP_KEY_ALIASES = {
        "maccs": "maccs", "maccs_fp": "maccs",
        "morgan": "ecfp4", "morgan_fp": "ecfp4", "ecfp4": "ecfp4", "ecfp4_fp": "ecfp4",
        "morgan_count": "ecfp4_count", "morgan_count_fp": "ecfp4_count",
        "ecfp4_count": "ecfp4_count", "ecfp4_count_fp": "ecfp4_count",
        "feature_morgan": "ecfp4_feature", "feature_morgan_fp": "ecfp4_feature",
        "ecfp4_feature": "ecfp4_feature", "ecfp4_feature_fp": "ecfp4_feature",
        "ecfp6": "ecfp6", "ecfp6_fp": "ecfp6",
        "fcfp6": "ecfp6_feature", "fcfp6_fp": "ecfp6_feature",
        "ecfp6_feature": "ecfp6_feature", "ecfp6_feature_fp": "ecfp6_feature",
        "rdkit": "rdkit", "rdkit_fp": "rdkit",
        "atom_pair": "atom_pair", "atom_pair_fp": "atom_pair",
        "topological_torsion": "topological_torsion",
        "topological_torsion_fp": "topological_torsion",
        "pharmacophore2d": "pharmacophore2d", "pharmacophore2d_fp": "pharmacophore2d",
        "avalon": "avalon", "avalon_fp": "avalon",
        "erg": "erg", "erg_fp": "erg",
        "peoe_vsa": "peoe_vsa", "peoe_vsa_fp": "peoe_vsa",
        "slogp_vsa": "slogp_vsa", "slogp_vsa_fp": "slogp_vsa",
        "smr_vsa": "smr_vsa", "smr_vsa_fp": "smr_vsa",
        "mqn": "mqn", "mqn_fp": "mqn",
    }

    @classmethod
    def _get_fp_generators(cls) -> Dict[str, Any]:
        """Lazily initialize and cache RDKit fingerprint generators."""
        if cls._FP_GENERATORS is None:
            cls._FP_GENERATORS = {
                "ecfp4": rdFingerprintGenerator.GetMorganGenerator(
                    radius=2, fpSize=2048, countSimulation=True, includeChirality=True
                ),
                "ecfp4_feature": rdFingerprintGenerator.GetMorganGenerator(
                    radius=2,
                    fpSize=2048,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
                    countSimulation=True,
                ),
                "ecfp6": rdFingerprintGenerator.GetMorganGenerator(
                    radius=3, fpSize=2048, countSimulation=True, includeChirality=True
                ),
                "ecfp6_feature": rdFingerprintGenerator.GetMorganGenerator(
                    radius=3,
                    fpSize=2048,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
                    countSimulation=True,
                ),
                "rdkit": rdFingerprintGenerator.GetRDKitFPGenerator(
                    minPath=1, maxPath=7, fpSize=2048, countSimulation=True,
                    branchedPaths=True, useBondOrder=True,
                ),
                "atom_pair": rdFingerprintGenerator.GetAtomPairGenerator(
                    minDistance=1, maxDistance=8, fpSize=2048, countSimulation=True
                ),
                "topological_torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                    torsionAtomCount=4, fpSize=2048, countSimulation=True
                ),
            }
        return cls._FP_GENERATORS

    @classmethod
    def _normalize_include_fps(
        cls, include_fps: Optional[Iterable[str]]
    ) -> Tuple[str, ...]:
        """Normalize/validate fingerprint selection."""
        if include_fps is None:
            return cls._SUPPORTED_FP_NAMES
        selected_raw = tuple(dict.fromkeys(str(x).lower() for x in include_fps))
        invalid = [x for x in selected_raw if x not in cls._FP_KEY_ALIASES]
        if invalid:
            raise ValueError(
                f"Unsupported fingerprint names: {invalid}. "
                f"Supported: {list(cls._SUPPORTED_FP_NAMES)}"
            )
        return tuple(dict.fromkeys(cls._FP_KEY_ALIASES[x] for x in selected_raw))

    def get_fingerprints(
        self,
        mol: Chem.Mol,
        include_fps: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract various molecular fingerprints.

        Returns:
            Dictionary of fingerprint tensors
        """
        include = set(self._normalize_include_fps(include_fps))
        gens = self._get_fp_generators()
        fingerprints = {}

        def _bitvect_to_tensor(bitvect) -> torch.Tensor:
            arr = np.zeros((bitvect.GetNumBits(),), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bitvect, arr)
            return torch.from_numpy(arr)

        # MACCS keys
        if 'maccs' in include:
            fingerprints['maccs'] = _bitvect_to_tensor(MACCSkeys.GenMACCSKeys(mol))

        # ECFP4 fingerprints
        if 'ecfp4' in include:
            fingerprints['ecfp4'] = torch.from_numpy(
                gens['ecfp4'].GetFingerprintAsNumPy(mol)
            ).float()
        if 'ecfp4_count' in include:
            fingerprints['ecfp4_count'] = torch.from_numpy(
                gens['ecfp4'].GetCountFingerprintAsNumPy(mol)
            ).float()

        # ECFP4 feature-invariant variant
        if 'ecfp4_feature' in include:
            fingerprints['ecfp4_feature'] = torch.from_numpy(
                gens['ecfp4_feature'].GetFingerprintAsNumPy(mol)
            ).float()

        # ECFP6 / feature-invariant variants
        if 'ecfp6' in include:
            fingerprints['ecfp6'] = torch.from_numpy(
                gens['ecfp6'].GetFingerprintAsNumPy(mol)
            ).float()
        if 'ecfp6_feature' in include:
            fingerprints['ecfp6_feature'] = torch.from_numpy(
                gens['ecfp6_feature'].GetFingerprintAsNumPy(mol)
            ).float()

        # RDKit fingerprint
        if 'rdkit' in include:
            fingerprints['rdkit'] = torch.from_numpy(
                gens['rdkit'].GetFingerprintAsNumPy(mol)
            ).float()

        # Atom pair fingerprint
        if 'atom_pair' in include:
            fingerprints['atom_pair'] = torch.from_numpy(
                gens['atom_pair'].GetFingerprintAsNumPy(mol)
            ).float()

        # Topological torsion
        if 'topological_torsion' in include:
            fingerprints['topological_torsion'] = torch.from_numpy(
                gens['topological_torsion'].GetFingerprintAsNumPy(mol)
            ).float()

        # Pharmacophore 2D (lazy import)
        if 'pharmacophore2d' in include:
            from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
            pharm_fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
            bit_vector = torch.zeros(1024)
            for bit_id in pharm_fp.GetOnBits():
                if bit_id < 1024:
                    bit_vector[bit_id] = 1.0
            fingerprints['pharmacophore2d'] = bit_vector.float()

        # Avalon fingerprint (lazy import, optional)
        if 'avalon' in include:
            try:
                from rdkit.Avalon import pyAvalonTools
                avalon = pyAvalonTools.GetAvalonFP(mol, nBits=2048)
                fingerprints['avalon'] = _bitvect_to_tensor(avalon)
            except ImportError:
                fingerprints['avalon'] = torch.zeros(2048, dtype=torch.float32)

        # ErG pharmacophore fingerprint (315 dims in RDKit)
        if 'erg' in include:
            fingerprints['erg'] = torch.tensor(
                rdReducedGraphs.GetErGFingerprint(mol), dtype=torch.float32
            )

        # MOE-type VSA descriptors (property-partitioned surface area)
        if 'peoe_vsa' in include:
            fingerprints['peoe_vsa'] = torch.tensor(
                rdMolDescriptors.PEOE_VSA_(mol), dtype=torch.float32
            )
        if 'slogp_vsa' in include:
            fingerprints['slogp_vsa'] = torch.tensor(
                rdMolDescriptors.SlogP_VSA_(mol), dtype=torch.float32
            )
        if 'smr_vsa' in include:
            fingerprints['smr_vsa'] = torch.tensor(
                rdMolDescriptors.SMR_VSA_(mol), dtype=torch.float32
            )

        # Molecular Quantum Numbers (42-dim integer fingerprint)
        if 'mqn' in include:
            fingerprints['mqn'] = torch.tensor(
                rdMolDescriptors.MQNs_(mol), dtype=torch.float32
            )

        return fingerprints
