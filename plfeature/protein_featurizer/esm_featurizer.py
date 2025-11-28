"""
ESM Embedding Extractor for Protein Sequences.

Supports ESM3 and ESMC models for per-residue embedding extraction.
BOS/EOS tokens are stored separately for flexibility.

Usage:
    from plfeature.esm_featurizer import ESMFeaturizer

    # Single model
    featurizer = ESMFeaturizer(model_type="esmc", model_name="esmc_600m")
    result = featurizer.extract("MKTIIALSYIFCLVFA")
    embeddings = result['embeddings']  # [L, D]
    bos_token = result['bos_token']    # [D]
    eos_token = result['eos_token']    # [D]

    # Extract from PDB (uses sequence from structure)
    result = featurizer.extract_from_pdb("protein.pdb")
"""

import torch
import logging
from typing import Dict, Literal, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ESMFeaturizer:
    """Extract protein embeddings using ESM3 or ESMC models."""

    # Model embedding dimensions
    MODEL_DIMS = {
        # ESMC models
        "esmc_300m": 960,
        "esmc_600m": 1152,
        # ESM3 models
        "esm3-open": 1536,
    }

    def __init__(
        self,
        model_type: Literal["esm3", "esmc"] = "esmc",
        model_name: str = "esmc_600m",
        device: str = "cuda",
    ):
        """
        Initialize ESM embedding extractor.

        Args:
            model_type: "esm3" or "esmc"
            model_name: Model variant name
                ESMC: "esmc_600m" (1152-dim), "esmc_300m" (960-dim)
                ESM3: "esm3-open" (1536-dim)
            device: "cuda" or "cpu"
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.model = None

        self._load_model()

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension for current model."""
        return self.MODEL_DIMS.get(self.model_name, 1152)

    def _load_model(self):
        """Load ESM3 or ESMC model."""
        try:
            if self.model_type == "esm3":
                from esm.models.esm3 import ESM3
                logger.info(f"Loading ESM3 model: {self.model_name}")

                try:
                    self.model = ESM3.from_pretrained(self.model_name)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load ESM3 model '{self.model_name}'.\n"
                        f"Available models: esm3-open\n"
                        f"Error: {e}"
                    )

                if self.device == "cpu":
                    self.model = self.model.float()
                self.model = self.model.to(self.device)

            elif self.model_type == "esmc":
                from esm.models.esmc import ESMC
                logger.info(f"Loading ESMC model: {self.model_name}")

                try:
                    self.model = ESMC.from_pretrained(self.model_name).to(self.device)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load ESMC model '{self.model_name}'.\n"
                        f"Available models: esmc_300m, esmc_600m\n"
                        f"Error: {e}"
                    )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            logger.info(f"Model loaded on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import ESM models.\n"
                f"Install: pip install esm\n"
                f"Error: {e}"
            )

    @torch.no_grad()
    def extract(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Extract per-residue embeddings from a protein sequence.

        Args:
            sequence: Protein sequence (e.g., "MKTIIALSYIFCLVFA")

        Returns:
            Dictionary with:
                - embeddings: [L, D] per-residue embeddings
                - bos_token: [D] BOS (beginning of sequence) token embedding
                - eos_token: [D] EOS (end of sequence) token embedding
                - full_embeddings: [L+2, D] full embeddings including BOS/EOS
        """
        from esm.sdk.api import ESMProtein, LogitsConfig

        protein = ESMProtein(sequence=sequence)

        if self.model_type == "esmc":
            protein_tensor = self.model.encode(protein)
            logits_output = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            full_embeddings = logits_output.embeddings

        elif self.model_type == "esm3":
            from esm.sdk.api import SamplingConfig

            protein_tensor = self.model.encode(protein)
            output = self.model.forward_and_sample(
                protein_tensor,
                SamplingConfig(return_per_residue_embeddings=True)
            )
            full_embeddings = output.per_residue_embedding

        # Convert to tensor
        if not isinstance(full_embeddings, torch.Tensor):
            full_embeddings = torch.tensor(full_embeddings)

        # Remove batch dimension if present
        if full_embeddings.dim() == 3:
            full_embeddings = full_embeddings.squeeze(0)

        # Move to CPU
        full_embeddings = full_embeddings.cpu()

        # Extract BOS/EOS tokens and residue embeddings
        # Full embeddings: [BOS, res1, res2, ..., resL, EOS]
        if full_embeddings.shape[0] == len(sequence) + 2:
            bos_token = full_embeddings[0]           # [D]
            eos_token = full_embeddings[-1]          # [D]
            embeddings = full_embeddings[1:-1]       # [L, D]
        else:
            # No BOS/EOS tokens (shouldn't happen, but handle it)
            bos_token = torch.zeros(full_embeddings.shape[-1])
            eos_token = torch.zeros(full_embeddings.shape[-1])
            embeddings = full_embeddings

        return {
            'embeddings': embeddings,           # [L, D]
            'bos_token': bos_token,             # [D]
            'eos_token': eos_token,             # [D]
            'full_embeddings': full_embeddings, # [L+2, D]
        }

    def extract_from_pdb(
        self,
        pdb_path: Union[str, Path],
        chain_id: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from PDB file.

        Args:
            pdb_path: Path to PDB file
            chain_id: Specific chain to extract (None = all chains concatenated)

        Returns:
            Dictionary with embeddings and special tokens
        """
        sequence = self._get_sequence_from_pdb(pdb_path, chain_id)
        return self.extract(sequence)

    def extract_by_chain(
        self,
        pdb_path: Union[str, Path],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract embeddings for each chain separately.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Dictionary mapping chain_id -> embeddings dict
        """
        sequences = self._get_sequences_by_chain(pdb_path)
        results = {}

        for chain_id, sequence in sequences.items():
            logger.info(f"Extracting embeddings for chain {chain_id}: {len(sequence)} residues")
            results[chain_id] = self.extract(sequence)

        return results

    def _get_sequence_from_pdb(
        self,
        pdb_path: Union[str, Path],
        chain_id: Optional[str] = None,
    ) -> str:
        """
        Extract sequence from PDB file.

        Uses PDBParser for consistent parsing across all featurizers.
        """
        from .pdb_utils import PDBParser

        parser = PDBParser(str(pdb_path))
        return parser.get_sequence(chain_id=chain_id)

    def _get_sequences_by_chain(
        self,
        pdb_path: Union[str, Path],
    ) -> Dict[str, str]:
        """
        Extract sequences for each chain from PDB file.

        Uses PDBParser for consistent parsing across all featurizers.
        """
        from .pdb_utils import PDBParser

        parser = PDBParser(str(pdb_path))
        return parser.get_sequence_by_chain()


class DualESMFeaturizer:
    """Extract embeddings from both ESMC and ESM3 models."""

    def __init__(
        self,
        esmc_model: str = "esmc_600m",
        esm3_model: str = "esm3-open",
        device: str = "cuda",
    ):
        """
        Initialize dual ESM extractor.

        Args:
            esmc_model: ESMC model variant
            esm3_model: ESM3 model variant
            device: "cuda" or "cpu"
        """
        logger.info("Initializing ESMC extractor...")
        self.esmc = ESMFeaturizer(
            model_type="esmc",
            model_name=esmc_model,
            device=device
        )

        logger.info("Initializing ESM3 extractor...")
        self.esm3 = ESMFeaturizer(
            model_type="esm3",
            model_name=esm3_model,
            device=device
        )

    @property
    def esmc_dim(self) -> int:
        return self.esmc.embedding_dim

    @property
    def esm3_dim(self) -> int:
        return self.esm3.embedding_dim

    def extract(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from both models for a single sequence.

        Returns:
            Dictionary with:
                - esmc_embeddings: [L, D1]
                - esmc_bos_token: [D1]
                - esmc_eos_token: [D1]
                - esm3_embeddings: [L, D2]
                - esm3_bos_token: [D2]
                - esm3_eos_token: [D2]
        """
        esmc_result = self.esmc.extract(sequence)
        esm3_result = self.esm3.extract(sequence)

        return {
            'esmc_embeddings': esmc_result['embeddings'],
            'esmc_bos_token': esmc_result['bos_token'],
            'esmc_eos_token': esmc_result['eos_token'],
            'esm3_embeddings': esm3_result['embeddings'],
            'esm3_bos_token': esm3_result['bos_token'],
            'esm3_eos_token': esm3_result['eos_token'],
        }

    def extract_by_chain(
        self,
        sequences_by_chain: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings per chain and concatenate.

        Each chain is processed separately through ESM models,
        preserving proper BOS/EOS token handling per chain.

        Args:
            sequences_by_chain: Dict mapping chain_id -> sequence

        Returns:
            Dictionary with:
                - esmc_embeddings: [total_residues, D1] concatenated
                - esmc_bos_token: [D1] from first chain
                - esmc_eos_token: [D1] from last chain
                - esm3_embeddings: [total_residues, D2] concatenated
                - esm3_bos_token: [D2] from first chain
                - esm3_eos_token: [D2] from last chain
                - chain_boundaries: List of (start, end) indices per chain
        """
        if not sequences_by_chain:
            raise ValueError("No sequences provided")

        # Sort chains for consistent ordering
        sorted_chains = sorted(sequences_by_chain.keys())

        esmc_embeddings_list = []
        esm3_embeddings_list = []
        chain_boundaries = []
        current_idx = 0

        esmc_bos = None
        esmc_eos = None
        esm3_bos = None
        esm3_eos = None

        for i, chain_id in enumerate(sorted_chains):
            sequence = sequences_by_chain[chain_id]
            if not sequence:
                continue

            logger.info(f"Extracting embeddings for chain {chain_id}: {len(sequence)} residues")

            # Extract from both models
            esmc_result = self.esmc.extract(sequence)
            esm3_result = self.esm3.extract(sequence)

            # Store embeddings
            esmc_embeddings_list.append(esmc_result['embeddings'])
            esm3_embeddings_list.append(esm3_result['embeddings'])

            # Track chain boundaries
            seq_len = len(sequence)
            chain_boundaries.append((current_idx, current_idx + seq_len))
            current_idx += seq_len

            # Keep BOS from first chain, EOS from last chain
            if i == 0:
                esmc_bos = esmc_result['bos_token']
                esm3_bos = esm3_result['bos_token']
            if i == len(sorted_chains) - 1:
                esmc_eos = esmc_result['eos_token']
                esm3_eos = esm3_result['eos_token']

        # Concatenate all chain embeddings
        esmc_embeddings = torch.cat(esmc_embeddings_list, dim=0)
        esm3_embeddings = torch.cat(esm3_embeddings_list, dim=0)

        return {
            'esmc_embeddings': esmc_embeddings,
            'esmc_bos_token': esmc_bos,
            'esmc_eos_token': esmc_eos,
            'esm3_embeddings': esm3_embeddings,
            'esm3_bos_token': esm3_bos,
            'esm3_eos_token': esm3_eos,
            'chain_boundaries': chain_boundaries,
        }

    def extract_from_pdb(
        self,
        pdb_path: Union[str, Path],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from PDB file using both models.

        Extracts each chain separately for proper ESM processing.
        """
        from .pdb_utils import PDBParser

        parser = PDBParser(str(pdb_path))
        sequences_by_chain = parser.get_sequence_by_chain()

        return self.extract_by_chain(sequences_by_chain)

    def extract_from_parser(
        self,
        pdb_parser: 'PDBParser',
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings using pre-parsed PDB data.

        Args:
            pdb_parser: Pre-initialized PDBParser instance

        Returns:
            Dictionary with embeddings from both models
        """
        sequences_by_chain = pdb_parser.get_sequence_by_chain()
        return self.extract_by_chain(sequences_by_chain)
