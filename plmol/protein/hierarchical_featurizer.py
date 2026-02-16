"""
Hierarchical Protein Featurizer for Atom-Residue attention models.

This module provides feature extraction with atom-residue mapping
for hierarchical attention mechanisms:
  - Atom attention → Residue pooling → Residue attention → Atom broadcast

Uses AtomFeaturizer for protein-specific atom tokens, ResidueFeaturizer for
residue features, and DualESMFeaturizer for ESMC + ESM3 embeddings.

Usage:
    from plmol.protein import HierarchicalFeaturizer

    featurizer = HierarchicalFeaturizer()
    data = featurizer.featurize("protein.pdb")

    # Access features (integer indices for embedding lookup)
    atom_tokens = data.atom_tokens           # [num_atoms] - token indices (0-186)
    atom_elements = data.atom_elements       # [num_atoms] - element indices (0-7)
    atom_residue_types = data.atom_residue_types  # [num_atoms] - residue indices (0-21)

    # Residue features
    residue_features = data.residue_features # [num_residues, 76]
    esmc_embeddings = data.esmc_embeddings   # [num_residues, 1152]
    esm3_embeddings = data.esm3_embeddings   # [num_residues, 1536]

    # For pooling: atom → residue
    atom_to_residue = data.atom_to_residue   # [num_atoms] index

    # Convert to one-hot in model if needed:
    # atom_tokens_onehot = F.one_hot(data.atom_tokens, num_classes=187)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
from collections import defaultdict

import torch
import numpy as np

from .residue_featurizer import ResidueFeaturizer
from .atom_featurizer import AtomFeaturizer
from .utils import PDBParser, normalize_residue_name, calculate_sidechain_centroid
from ..constants import (
    RESIDUE_ATOM_TOKEN,
    UNK_TOKEN,
    RESIDUE_TOKEN,
    HISTIDINE_VARIANTS,
    CYSTEINE_VARIANTS,
    AMINO_ACID_LETTERS,
    MAX_ATOMS_PER_RESIDUE,
    # Element types for hierarchical models
    SIMPLIFIED_ELEMENT_TYPES,
    NUM_SIMPLIFIED_ELEMENT_TYPES,
    METAL_ELEMENTS,
)

import logging
logger = logging.getLogger(__name__)


# Number of unique atom tokens
NUM_ATOM_TOKENS = UNK_TOKEN + 1  # 187

# Backward compatibility aliases
ELEMENT_TYPES = SIMPLIFIED_ELEMENT_TYPES
NUM_ELEMENT_TYPES = NUM_SIMPLIFIED_ELEMENT_TYPES


@dataclass
class HierarchicalProteinData:
    """
    Hierarchical protein data for atom-residue attention models.

    Atom-level tensors (integer indices for embedding lookup):
        atom_tokens: [num_atoms] - Atom token indices (0-186, 187 classes)
        atom_coords: [num_atoms, 3] - 3D coordinates (raw, not normalized)
        atom_sasa: [num_atoms] - SASA values (normalized by /100)
        atom_elements: [num_atoms] - Element type indices (0-7, 8 classes)
        atom_residue_types: [num_atoms] - Residue type indices (0-21, 22 classes)
        atom_names: List[str] - PDB atom names (CA, CB, etc.)

    Residue-level tensors:
        residue_features: [num_residues, 76] - Residue feature vectors
        residue_ca_coords: [num_residues, 3] - CA coordinates
        residue_sc_coords: [num_residues, 3] - Sidechain centroid coordinates
        residue_names: List[str] - Residue names (ALA, GLY, etc.)
        residue_ids: List[Tuple[str, int]] - (chain, resnum)

    Mapping tensors (for pooling/broadcast):
        atom_to_residue: [num_atoms] - Residue index for each atom
        residue_atom_indices: [num_residues, max_atoms] - Atom indices per residue
        residue_atom_mask: [num_residues, max_atoms] - Valid atom mask
        num_atoms_per_residue: [num_residues] - Atom count per residue

    ESM embeddings (6 tensors total):
        esmc_embeddings: [num_residues, 1152] - ESMC per-residue embeddings
        esmc_bos: [1152] - ESMC BOS token
        esmc_eos: [1152] - ESMC EOS token
        esm3_embeddings: [num_residues, 1536] - ESM3 per-residue embeddings
        esm3_bos: [1536] - ESM3 BOS token
        esm3_eos: [1536] - ESM3 EOS token

    Note:
        Use torch.nn.Embedding or F.one_hot() in your model to convert indices to vectors.
    """
    # Atom-level (integer indices for embedding lookup)
    atom_tokens: torch.Tensor           # [N_atom] token indices (0-186)
    atom_coords: torch.Tensor           # [N_atom, 3] raw coordinates
    atom_sasa: torch.Tensor             # [N_atom] normalized SASA
    atom_elements: torch.Tensor         # [N_atom] element indices (0-7)
    atom_residue_types: torch.Tensor    # [N_atom] residue type indices (0-21)
    atom_names: List[str]               # atom names (CA, CB, etc.)

    # Residue-level
    residue_features: torch.Tensor      # [N_res, 76]
    residue_ca_coords: torch.Tensor     # [N_res, 3]
    residue_sc_coords: torch.Tensor     # [N_res, 3]
    residue_names: List[str]
    residue_ids: List[Tuple[str, int]]

    # Residue vector features (optional) [N_res, 31, 3]
    residue_vector_features: Optional[torch.Tensor] = None

    # ESMC embeddings (3 tensors)
    esmc_embeddings: Optional[torch.Tensor] = None   # [N_res, 1152]
    esmc_bos: Optional[torch.Tensor] = None          # [1152]
    esmc_eos: Optional[torch.Tensor] = None          # [1152]

    # ESM3 embeddings (3 tensors)
    esm3_embeddings: Optional[torch.Tensor] = None   # [N_res, 1536]
    esm3_bos: Optional[torch.Tensor] = None          # [1536]
    esm3_eos: Optional[torch.Tensor] = None          # [1536]

    # Mapping
    atom_to_residue: torch.Tensor = None
    residue_atom_indices: torch.Tensor = None
    residue_atom_mask: torch.Tensor = None
    num_atoms_per_residue: torch.Tensor = None

    @property
    def num_atoms(self) -> int:
        return self.atom_tokens.shape[0]

    @property
    def num_residues(self) -> int:
        return self.residue_features.shape[0]

    @property
    def max_atoms_per_residue(self) -> int:
        return self.residue_atom_indices.shape[1] if self.residue_atom_indices is not None else 0

    @property
    def residue_dim(self) -> int:
        return self.residue_features.shape[1]

    @property
    def num_atom_classes(self) -> int:
        return NUM_ATOM_TOKENS  # 187

    @property
    def num_element_classes(self) -> int:
        return NUM_ELEMENT_TYPES  # 8

    @property
    def num_residue_classes(self) -> int:
        return 22  # 21 amino acids + UNK

    @property
    def esmc_dim(self) -> Optional[int]:
        """Get ESMC embedding dimension if available."""
        return self.esmc_embeddings.shape[-1] if self.esmc_embeddings is not None else None

    @property
    def esm3_dim(self) -> Optional[int]:
        """Get ESM3 embedding dimension if available."""
        return self.esm3_embeddings.shape[-1] if self.esm3_embeddings is not None else None

    @property
    def has_esm(self) -> bool:
        """Check if ESM embeddings are available."""
        return self.esmc_embeddings is not None or self.esm3_embeddings is not None

    def to(self, device: torch.device) -> 'HierarchicalProteinData':
        """Move all tensors to device."""
        return HierarchicalProteinData(
            atom_tokens=self.atom_tokens.to(device),
            atom_coords=self.atom_coords.to(device),
            atom_sasa=self.atom_sasa.to(device),
            atom_elements=self.atom_elements.to(device),
            atom_residue_types=self.atom_residue_types.to(device),
            atom_names=self.atom_names,
            residue_features=self.residue_features.to(device),
            residue_ca_coords=self.residue_ca_coords.to(device),
            residue_sc_coords=self.residue_sc_coords.to(device),
            residue_names=self.residue_names,
            residue_ids=self.residue_ids,
            residue_vector_features=self.residue_vector_features.to(device) if self.residue_vector_features is not None else None,
            esmc_embeddings=self.esmc_embeddings.to(device) if self.esmc_embeddings is not None else None,
            esmc_bos=self.esmc_bos.to(device) if self.esmc_bos is not None else None,
            esmc_eos=self.esmc_eos.to(device) if self.esmc_eos is not None else None,
            esm3_embeddings=self.esm3_embeddings.to(device) if self.esm3_embeddings is not None else None,
            esm3_bos=self.esm3_bos.to(device) if self.esm3_bos is not None else None,
            esm3_eos=self.esm3_eos.to(device) if self.esm3_eos is not None else None,
            atom_to_residue=self.atom_to_residue.to(device),
            residue_atom_indices=self.residue_atom_indices.to(device),
            residue_atom_mask=self.residue_atom_mask.to(device),
            num_atoms_per_residue=self.num_atoms_per_residue.to(device),
        )

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions."""
        dims = {
            'num_atoms': self.num_atoms,
            'num_residues': self.num_residues,
            'residue_dim': self.residue_dim,
            'max_atoms_per_residue': self.max_atoms_per_residue,
            'num_atom_classes': self.num_atom_classes,
            'num_element_classes': self.num_element_classes,
            'num_residue_classes': self.num_residue_classes,
        }
        if self.esmc_dim is not None:
            dims['esmc_dim'] = self.esmc_dim
        if self.esm3_dim is not None:
            dims['esm3_dim'] = self.esm3_dim
        return dims

    def select_residues(
        self,
        residue_indices: Union[List[int], torch.Tensor],
    ) -> 'HierarchicalProteinData':
        """
        Select specific residues and their corresponding atoms.

        Args:
            residue_indices: Indices of residues to select

        Returns:
            New HierarchicalProteinData with only selected residues/atoms
        """
        if not isinstance(residue_indices, torch.Tensor):
            residue_indices = torch.tensor(residue_indices, dtype=torch.long)

        # Get atom mask for selected residues
        atom_mask = torch.isin(self.atom_to_residue, residue_indices)
        atom_indices = atom_mask.nonzero(as_tuple=True)[0]

        # Remap residue indices to 0, 1, 2, ...
        old_to_new = {old.item(): new for new, old in enumerate(residue_indices)}
        new_atom_to_residue = torch.tensor([
            old_to_new[self.atom_to_residue[i].item()]
            for i in atom_indices
        ], dtype=torch.long)

        # Extract atom-level data
        new_atom_tokens = self.atom_tokens[atom_mask]
        new_atom_coords = self.atom_coords[atom_mask]
        new_atom_sasa = self.atom_sasa[atom_mask]
        new_atom_elements = self.atom_elements[atom_mask]
        new_atom_residue_types = self.atom_residue_types[atom_mask]
        new_atom_names = [self.atom_names[i] for i in atom_indices.tolist()]

        # Extract residue-level data
        new_residue_features = self.residue_features[residue_indices]
        new_residue_ca_coords = self.residue_ca_coords[residue_indices]
        new_residue_sc_coords = self.residue_sc_coords[residue_indices]
        new_residue_names = [self.residue_names[i] for i in residue_indices.tolist()]
        new_residue_ids = [self.residue_ids[i] for i in residue_indices.tolist()]

        # Residue vector features (optional)
        new_residue_vector_features = None
        if self.residue_vector_features is not None:
            new_residue_vector_features = self.residue_vector_features[residue_indices]

        # Build new residue_atom_indices and mask (vectorized)
        num_new_residues = len(residue_indices)

        # Vectorized atom count per residue using bincount
        new_num_atoms_per_residue = torch.bincount(
            new_atom_to_residue, minlength=num_new_residues
        )
        max_atoms = new_num_atoms_per_residue.max().item() if num_new_residues > 0 else 1

        new_residue_atom_indices = torch.full((num_new_residues, max_atoms), -1, dtype=torch.long)
        new_residue_atom_mask = torch.zeros(num_new_residues, max_atoms, dtype=torch.bool)

        # Vectorized: sort atoms by residue index, then fill
        sorted_indices = torch.argsort(new_atom_to_residue)
        sorted_residues = new_atom_to_residue[sorted_indices]

        # Use cumsum to find position within each residue
        residue_starts = torch.zeros(num_new_residues + 1, dtype=torch.long)
        residue_starts[1:] = torch.cumsum(new_num_atoms_per_residue, dim=0)

        for new_res_idx in range(num_new_residues):
            start = residue_starts[new_res_idx].item()
            end = residue_starts[new_res_idx + 1].item()
            n_atoms = end - start
            if n_atoms > 0:
                atom_indices = sorted_indices[start:end]
                new_residue_atom_indices[new_res_idx, :n_atoms] = atom_indices
                new_residue_atom_mask[new_res_idx, :n_atoms] = True

        # ESM embeddings: select residue embeddings, keep original BOS/EOS
        new_esmc_embeddings = None
        new_esm3_embeddings = None
        if self.esmc_embeddings is not None:
            new_esmc_embeddings = self.esmc_embeddings[residue_indices]
        if self.esm3_embeddings is not None:
            new_esm3_embeddings = self.esm3_embeddings[residue_indices]

        return HierarchicalProteinData(
            atom_tokens=new_atom_tokens,
            atom_coords=new_atom_coords,
            atom_sasa=new_atom_sasa,
            atom_elements=new_atom_elements,
            atom_residue_types=new_atom_residue_types,
            atom_names=new_atom_names,
            residue_features=new_residue_features,
            residue_ca_coords=new_residue_ca_coords,
            residue_sc_coords=new_residue_sc_coords,
            residue_names=new_residue_names,
            residue_ids=new_residue_ids,
            residue_vector_features=new_residue_vector_features,
            esmc_embeddings=new_esmc_embeddings,
            esmc_bos=self.esmc_bos,  # Keep original BOS/EOS
            esmc_eos=self.esmc_eos,
            esm3_embeddings=new_esm3_embeddings,
            esm3_bos=self.esm3_bos,
            esm3_eos=self.esm3_eos,
            atom_to_residue=new_atom_to_residue,
            residue_atom_indices=new_residue_atom_indices,
            residue_atom_mask=new_residue_atom_mask,
            num_atoms_per_residue=new_num_atoms_per_residue,
        )


class HierarchicalFeaturizer:
    """
    Extract hierarchical features from protein for atom-residue attention.

    Combines:
        - AtomFeaturizer for protein-specific atom tokens (187 classes)
        - ResidueFeaturizer for residue-level features (76 dim + 31x3 vectors)
        - DualESMFeaturizer for ESMC + ESM3 embeddings

    Args:
        esmc_model: ESMC model name (default: "esmc_600m", 1152-dim)
        esm3_model: ESM3 model name (default: "esm3-open", 1536-dim)
        esm_device: Device for ESM models ("cuda" or "cpu")

    Feature dimensions:
        Atom features (integer indices for nn.Embedding):
            - atom_tokens: [N_atom] - Atom token indices (0-186, 187 classes)
            - atom_coords: [N_atom, 3] - 3D coordinates
            - atom_sasa: [N_atom] - SASA values (normalized /100)
            - atom_elements: [N_atom] - Element type indices (0-7, 8 classes)
            - atom_residue_types: [N_atom] - Residue type indices (0-21, 22 classes)

        Residue features (from ResidueFeaturizer):
            - residue_features: [N_res, 76] scalar features
            - residue_vector_features: [N_res, 31, 3] vector features

        ESM embeddings (6 tensors):
            - esmc_embeddings: [N_res, 1152] - ESMC per-residue
            - esmc_bos: [1152] - ESMC BOS token
            - esmc_eos: [1152] - ESMC EOS token
            - esm3_embeddings: [N_res, 1536] - ESM3 per-residue
            - esm3_bos: [1536] - ESM3 BOS token
            - esm3_eos: [1536] - ESM3 EOS token
    """

    def __init__(
        self,
        esmc_model: str = "esmc_600m",
        esm3_model: str = "esm3-open",
        esm_device: str = "cuda",
    ):
        self._atom_featurizer = AtomFeaturizer()

        # Initialize Dual ESM featurizer (ESMC + ESM3)
        from .esm_featurizer import DualESMFeaturizer
        self._esm_featurizer = DualESMFeaturizer(
            esmc_model=esmc_model,
            esm3_model=esm3_model,
            device=esm_device,
        )

    def featurize(self, pdb_path: str) -> HierarchicalProteinData:
        """
        Extract hierarchical features from PDB file.

        Args:
            pdb_path: Path to PDB file

        Returns:
            HierarchicalProteinData with all features and mappings
        """
        # Step 0: Parse PDB once using PDBParser (single source of truth)
        pdb_parser = PDBParser(pdb_path)

        # Step 1: Get atom features from AtomFeaturizer (using pre-parsed data)
        atom_tokens, atom_coords = self._atom_featurizer.get_protein_atom_features_from_parser(pdb_parser)

        # Step 2: Get SASA from AtomFeaturizer (requires file path for FreeSASA)
        atom_sasa, _ = self._atom_featurizer.get_atom_sasa(pdb_path)

        # Ensure same length (SASA might have different atom count due to different parsing)
        orig_token_len = len(atom_tokens)
        orig_sasa_len = len(atom_sasa)
        min_len = min(orig_token_len, orig_sasa_len)

        if orig_token_len != orig_sasa_len:
            logger.warning(
                f"Atom count mismatch in {pdb_path}: tokens={orig_token_len}, SASA={orig_sasa_len}. "
                f"Truncating to {min_len} atoms."
            )

        atom_tokens = atom_tokens[:min_len]
        atom_coords = atom_coords[:min_len]
        atom_sasa = atom_sasa[:min_len]

        # Step 3: Get additional atom info from cached PDBParser (no re-parsing)
        atom_info = self._parse_atom_info(pdb_parser)
        atom_elements = atom_info['elements'][:min_len]
        atom_residue_types = atom_info['residue_tokens'][:min_len]
        atom_names = atom_info['atom_names'][:min_len]
        atom_residue_keys = atom_info['residue_keys'][:min_len]

        # Step 4: Normalize SASA (typical range 0-150 Å²)
        atom_sasa = atom_sasa / 100.0

        # Step 5: Get residue features from ResidueFeaturizer (using pre-parsed data)
        residue_featurizer = ResidueFeaturizer.from_parser(pdb_parser, pdb_path)
        residues = residue_featurizer.get_residues()

        # Build residue coordinate tensor
        num_residues = len(residues)
        residue_coords_full = torch.zeros(num_residues, MAX_ATOMS_PER_RESIDUE, 3)
        residue_types = torch.from_numpy(np.array(residues)[:, 2].astype(int))

        for idx, residue in enumerate(residues):
            # Use cached coordinates (O(1) lookup instead of pandas xs)
            residue_coord_np = residue_featurizer.get_residue_coordinates_numpy(residue)
            residue_coord = torch.from_numpy(residue_coord_np)
            residue_coords_full[idx, :residue_coord.shape[0], :] = residue_coord
            # Sidechain centroid (using unified calculate_sidechain_centroid)
            residue_coords_full[idx, -1, :] = torch.from_numpy(
                calculate_sidechain_centroid(residue_coord_np)
            )

        # Extract residue features
        scalar_features, vector_features = residue_featurizer._extract_residue_features(
            residue_coords_full, residue_types
        )

        # Concatenate scalar features
        residue_one_hot, terminal_flags, self_distance, degree_feature, has_chi, sasa, rf_distance = scalar_features
        residue_features = torch.cat([
            residue_one_hot.float(),      # 21
            terminal_flags.float(),       # 2
            self_distance,                # 10
            degree_feature,               # 20
            has_chi.float(),              # 5
            sasa,                         # 10
            rf_distance,                  # 8
        ], dim=-1)  # Total: 76

        # Residue coordinates: CA and sidechain centroid
        residue_ca_coords = residue_coords_full[:, 1, :]   # CA atom (index 1)
        residue_sc_coords = residue_coords_full[:, -1, :]  # Sidechain centroid (index 14)

        # Step 6: Build atom-residue mapping
        atom_to_residue, residue_atom_indices, residue_atom_mask, num_atoms_per_residue = \
            self._build_mapping(atom_residue_keys, residues)

        # Residue info
        residue_names = []
        residue_ids = []
        INT_TO_3LETTER = {
            0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS',
            5: 'GLN', 6: 'GLU', 7: 'GLY', 8: 'HIS', 9: 'ILE',
            10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE', 14: 'PRO',
            15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL', 20: 'UNK'
        }
        for chain, resnum, restype in residues:
            residue_names.append(INT_TO_3LETTER.get(restype, 'UNK'))
            residue_ids.append((chain, resnum))

        # Residue vector features [N_res, 31, 3]
        self_vector, rf_vector, local_frames = vector_features
        residue_vector_features = torch.cat([
            self_vector,      # [N_res, 20, 3]
            rf_vector,        # [N_res, 8, 3]
            local_frames,     # [N_res, 3, 3]
        ], dim=1)

        # Step 7: Keep categorical features as integer indices (more efficient)
        # Models can use nn.Embedding or F.one_hot() as needed

        # Step 8: Extract dual ESM embeddings (ESMC + ESM3) using pre-parsed data
        esm_result = self._esm_featurizer.extract_from_parser(pdb_parser)

        # ESMC: embeddings [N_res, 1152], bos [1152], eos [1152]
        esmc_embeddings = esm_result['esmc_embeddings']
        esmc_bos = esm_result['esmc_bos_token']
        esmc_eos = esm_result['esmc_eos_token']

        # ESM3: embeddings [N_res, 1536], bos [1536], eos [1536]
        esm3_embeddings = esm_result['esm3_embeddings']
        esm3_bos = esm_result['esm3_bos_token']
        esm3_eos = esm_result['esm3_eos_token']

        # Verify length matches and truncate/pad if needed
        def _adjust_length(emb, target_len, name):
            if emb.shape[0] != target_len:
                logger.warning(
                    f"{name} length ({emb.shape[0]}) != residue count ({target_len}). Adjusting."
                )
                if emb.shape[0] > target_len:
                    return emb[:target_len]
                else:
                    pad = torch.zeros(target_len - emb.shape[0], emb.shape[1])
                    return torch.cat([emb, pad], dim=0)
            return emb

        esmc_embeddings = _adjust_length(esmc_embeddings, num_residues, "ESMC")
        esm3_embeddings = _adjust_length(esm3_embeddings, num_residues, "ESM3")

        return HierarchicalProteinData(
            atom_tokens=atom_tokens,
            atom_coords=atom_coords,
            atom_sasa=atom_sasa,
            atom_elements=atom_elements,
            atom_residue_types=atom_residue_types,
            atom_names=atom_names,
            residue_features=residue_features,
            residue_ca_coords=residue_ca_coords,
            residue_sc_coords=residue_sc_coords,
            residue_names=residue_names,
            residue_ids=residue_ids,
            residue_vector_features=residue_vector_features,
            esmc_embeddings=esmc_embeddings,
            esmc_bos=esmc_bos,
            esmc_eos=esmc_eos,
            esm3_embeddings=esm3_embeddings,
            esm3_bos=esm3_bos,
            esm3_eos=esm3_eos,
            atom_to_residue=atom_to_residue,
            residue_atom_indices=residue_atom_indices,
            residue_atom_mask=residue_atom_mask,
            num_atoms_per_residue=num_atoms_per_residue,
        )

    def featurize_pocket(
        self,
        protein_pdb_path: str,
        ligand,  # Chem.Mol
        cutoff: float = 6.0,
    ) -> HierarchicalProteinData:
        """
        Extract hierarchical features from binding pocket.

        Args:
            protein_pdb_path: Path to protein PDB file
            ligand: RDKit Mol object of ligand (with 3D coords)
            cutoff: Distance cutoff for pocket extraction (default 6.0 Å)

        Returns:
            HierarchicalProteinData with all features and mappings
        """
        import tempfile
        import os
        from rdkit import Chem
        from ..interaction import extract_pocket

        # Extract pocket
        pocket_info = extract_pocket(protein_pdb_path, ligand, cutoff=cutoff)
        pocket_mol = pocket_info.pocket_mol

        # Save pocket to temp PDB
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(Chem.MolToPDBBlock(pocket_mol))
            pocket_pdb = f.name

        try:
            return self.featurize(pocket_pdb)
        finally:
            os.unlink(pocket_pdb)

    def _parse_atom_info(self, pdb_parser: PDBParser) -> Dict:
        """
        Extract atom information from pre-parsed PDB data.

        Uses PDBParser's cached data to avoid re-reading file.

        Args:
            pdb_parser: Pre-initialized PDBParser instance

        Returns dict with:
            - elements: [N_atom] element type indices
            - residue_tokens: [N_atom] residue type for each atom
            - atom_names: List[str] atom names
            - residue_keys: List[(chain, resnum)] for mapping
        """
        elements = []
        residue_tokens = []
        atom_names = []
        residue_keys = []

        for atom in pdb_parser.protein_atoms:
            # Normalize residue name using centralized function
            res_name_clean = normalize_residue_name(atom.res_name, atom.atom_name)

            # Element type (simplified: C, N, O, S, P, Se, Metal, UNK)
            element_upper = atom.element.upper() if atom.element else ''
            if element_upper in ELEMENT_TYPES:
                elem_idx = ELEMENT_TYPES[element_upper]
            elif element_upper in METAL_ELEMENTS:
                elem_idx = ELEMENT_TYPES['METAL']
            else:
                elem_idx = ELEMENT_TYPES['UNK']

            # Residue token
            res_tok = RESIDUE_TOKEN.get(res_name_clean, RESIDUE_TOKEN['UNK'])

            elements.append(elem_idx)
            residue_tokens.append(res_tok)
            atom_names.append(atom.atom_name)
            residue_keys.append((atom.chain_id, atom.res_num))

        return {
            'elements': torch.tensor(elements, dtype=torch.long),
            'residue_tokens': torch.tensor(residue_tokens, dtype=torch.long),
            'atom_names': atom_names,
            'residue_keys': residue_keys,
        }

    def _build_mapping(
        self,
        atom_residue_keys: List[Tuple[str, int]],
        residues: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build atom-residue mapping tensors.

        Args:
            atom_residue_keys: List of (chain, resnum) for each atom
            residues: List of (chain, resnum, restype) tuples

        Returns:
            Tuple of (atom_to_residue, residue_atom_indices, residue_atom_mask, num_atoms_per_residue)
        """
        # Build residue key to index mapping
        residue_to_idx = {(chain, resnum): i for i, (chain, resnum, _) in enumerate(residues)}

        num_atoms = len(atom_residue_keys)
        num_residues = len(residues)

        # Map atoms to residues
        atom_to_residue_list = []
        residue_to_atoms = defaultdict(list)

        for atom_idx, (chain, resnum) in enumerate(atom_residue_keys):
            key = (chain, resnum)
            res_idx = residue_to_idx.get(key, 0)  # default to first residue if not found
            atom_to_residue_list.append(res_idx)
            residue_to_atoms[res_idx].append(atom_idx)

        atom_to_residue = torch.tensor(atom_to_residue_list, dtype=torch.long)

        # Max atoms per residue
        max_atoms_per_res = max(len(atoms) for atoms in residue_to_atoms.values()) if residue_to_atoms else 1

        # Build residue -> atoms mapping
        residue_atom_indices = torch.full((num_residues, max_atoms_per_res), -1, dtype=torch.long)
        residue_atom_mask = torch.zeros(num_residues, max_atoms_per_res, dtype=torch.bool)
        num_atoms_per_residue = torch.zeros(num_residues, dtype=torch.long)

        for res_idx in range(num_residues):
            atom_indices = residue_to_atoms.get(res_idx, [])
            n_atoms = len(atom_indices)
            num_atoms_per_residue[res_idx] = n_atoms
            for j, atom_idx in enumerate(atom_indices):
                if j < max_atoms_per_res:
                    residue_atom_indices[res_idx, j] = atom_idx
                    residue_atom_mask[res_idx, j] = True

        return atom_to_residue, residue_atom_indices, residue_atom_mask, num_atoms_per_residue


def extract_hierarchical_features(
    pdb_path: str,
    esmc_model: str = "esmc_600m",
    esm3_model: str = "esm3-open",
    esm_device: str = "cuda",
) -> HierarchicalProteinData:
    """
    Convenience function to extract hierarchical features.

    Args:
        pdb_path: Path to PDB file
        esmc_model: ESMC model name (default: "esmc_600m", 1152-dim)
        esm3_model: ESM3 model name (default: "esm3-open", 1536-dim)
        esm_device: Device for ESM models ("cuda" or "cpu")

    Returns:
        HierarchicalProteinData with all features including:
            - Atom-level features (integer indices for nn.Embedding)
            - Residue-level features (scalar + vector)
            - ESM embeddings (ESMC + ESM3, 6 tensors total)

    Note:
        Categorical features are stored as integer indices.
        Use F.one_hot() or nn.Embedding in your model as needed.
    """
    featurizer = HierarchicalFeaturizer(
        esmc_model=esmc_model,
        esm3_model=esm3_model,
        esm_device=esm_device,
    )
    return featurizer.featurize(pdb_path)
