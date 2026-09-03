"""Protein language model embeddings behind one registry and one contract.

Getting a per-residue embedding should not depend on which family the model
comes from. ESM3 and ESMC arrive through Evolutionary Scale's ``esm`` SDK, while
Ankh, ESM-2 and ProtT5 are Hugging Face checkpoints with different tokenizer
conventions. This module hides that behind a name:

    from plmol import embed_sequence, list_protein_language_models

    list_protein_language_models()          # what can be asked for
    embed_sequence("MKTIIALSY", "ankh-base")

Every model returns the same dictionary, so a caller can swap one for another
without touching anything else.

Models are loaded lazily and cached per (name, device), because they are large
and the common pattern -- one Protein per file in a loop -- would otherwise
reload the weights on every iteration.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


from ..arrays import FLOAT
from ..errors import DependencyError, InputError

logger = logging.getLogger(__name__)


def _no_grad(method):
    """``torch.no_grad`` applied when the method runs, not when it is defined.

    The models here are torch models, but plmol imports without torch, so the
    decorator cannot touch it at class-definition time.
    """

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        import torch

        with torch.no_grad():
            return method(*args, **kwargs)

    return wrapper

# Loaded models are gigabytes each, so keep very few.
_MODEL_CACHE: "OrderedDict[tuple, ProteinLanguageModel]" = OrderedDict()
_MODEL_CACHE_MAX = 2


@dataclass(frozen=True)
class PLMSpec:
    """How to load one protein language model and read its output.

    Attributes:
        name: The name callers ask for.
        backend: Which loader handles it, ``"esm"`` or ``"huggingface"``.
        dim: Embedding width, from the model card.
        model_id: Identifier passed to the backend (a Hugging Face repo id, or
            the SDK's own model name).
        family: ``"esmc"``, ``"esm3"``, ``"ankh"``, ``"esm2"`` or ``"prot_t5"``.
        prefix_tokens: Special tokens the tokenizer prepends, stripped from the
            per-residue output and reported as ``bos``.
        suffix_tokens: Special tokens appended, reported as ``eos``.
        tokenize_as_chars: T5-derived tokenizers (Ankh, ProtT5) expect the
            sequence pre-split into single residues; ESM-2's does not.
        install_hint: Shown when the backend's package is missing.
    """

    name: str
    backend: str
    dim: int
    model_id: str
    family: str
    prefix_tokens: int = 1
    suffix_tokens: int = 1
    tokenize_as_chars: bool = False
    install_hint: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


PLM_REGISTRY: Dict[str, PLMSpec] = {}


def register_plm(spec: PLMSpec) -> PLMSpec:
    """Add a model to the registry. Re-registering a name replaces it."""
    PLM_REGISTRY[spec.name] = spec
    return spec


def list_protein_language_models() -> List[str]:
    """Names accepted by :func:`load_plm` and the ``embedding`` mode."""
    return sorted(PLM_REGISTRY)


def get_plm_spec(name: str) -> PLMSpec:
    """Registry entry for a model name.

    Raises:
        InputError: If the name is not registered.
    """
    try:
        return PLM_REGISTRY[name]
    except KeyError:
        raise InputError(
            f"Unknown protein language model {name!r}. "
            f"Available: {', '.join(list_protein_language_models())}"
        ) from None


def plm_dim(name: str) -> int:
    """Embedding width of a registered model, without loading it."""
    return get_plm_spec(name).dim


def resolve_device(device: str = "auto") -> str:
    """``"auto"`` picks cuda when it is actually available, else cpu."""
    if device == "auto":
        # "auto" is resolved even when no model is asked for, so a missing
        # torch means cpu rather than an error.
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class ProteinLanguageModel(ABC):
    """One loaded model. Subclasses differ only in how they run the forward pass."""

    def __init__(self, spec: PLMSpec, device: str):
        self.spec = spec
        self.device = device

    @property
    def dim(self) -> int:
        return self.spec.dim

    @abstractmethod
    def _forward(self, sequence: str) -> "torch.Tensor":
        """Return the raw ``(L + prefix + suffix, D)`` hidden states on CPU."""

    def embed(self, sequence: str) -> Dict[str, object]:
        """Per-residue embeddings for one sequence.

        Args:
            sequence: One-letter amino acid sequence.

        Returns:
            Dict with ``embeddings`` ``(L, D)``, ``bos`` ``(D,)``, ``eos``
            ``(D,)``, ``model``, ``dim`` and ``sequence``. Models without a
            BOS or EOS token report zeros there rather than None, so the shape
            contract holds for every model.

        Raises:
            InputError: If the sequence is empty.
        """
        if not sequence:
            raise InputError("Cannot embed an empty sequence.")

        hidden = self._forward(sequence)
        # The models are torch models; everything past this line is numpy, so
        # the embeddings have the same type as every other feature.
        if hasattr(hidden, "detach"):
            hidden = hidden.detach().cpu().numpy()
        hidden = np.asarray(hidden, dtype=FLOAT)
        if hidden.ndim == 3:
            hidden = hidden[0]

        return _split_special_tokens(hidden, sequence, self.spec)


def _split_special_tokens(
    hidden, sequence: str, spec: PLMSpec
) -> Dict[str, object]:
    """Peel the tokenizer's special tokens off the per-residue rows."""
    width = hidden.shape[-1]
    expected = len(sequence) + spec.prefix_tokens + spec.suffix_tokens

    if hidden.shape[0] == expected:
        prefix, suffix = spec.prefix_tokens, spec.suffix_tokens
    elif hidden.shape[0] == len(sequence):
        # Some checkpoints return residues only.
        prefix = suffix = 0
    else:
        raise InputError(
            f"{spec.name} returned {hidden.shape[0]} rows for a sequence of "
            f"{len(sequence)} residues; expected {expected} with "
            f"{spec.prefix_tokens} prefix and {spec.suffix_tokens} suffix tokens."
        )

    zeros = np.zeros(width, dtype=FLOAT)
    bos = hidden[prefix - 1] if prefix else zeros
    eos = hidden[hidden.shape[0] - suffix] if suffix else zeros
    embeddings = hidden[prefix : hidden.shape[0] - suffix] if suffix else hidden[prefix:]

    return {
        "embeddings": embeddings,
        "bos": bos,
        "eos": eos,
        "model": spec.name,
        "dim": int(width),
        "sequence": sequence,
    }


class EsmSdkModel(ProteinLanguageModel):
    """ESMC and ESM3 through Evolutionary Scale's ``esm`` SDK."""

    def __init__(self, spec: PLMSpec, device: str):
        super().__init__(spec, device)
        try:
            if spec.family == "esm3":
                from esm.models.esm3 import ESM3

                model = ESM3.from_pretrained(spec.model_id)
                if device == "cpu":
                    model = model.float()
                self.model = model.to(device)
            else:
                from esm.models.esmc import ESMC

                self.model = ESMC.from_pretrained(spec.model_id).to(device)
        except ImportError as exc:
            raise DependencyError(
                f"{spec.name} needs the esm package. {spec.install_hint}"
            ) from exc

    @_no_grad
    def _forward(self, sequence: str) -> "torch.Tensor":
        from esm.sdk.api import ESMProtein, LogitsConfig

        protein = ESMProtein(sequence=sequence)
        tensor = self.model.encode(protein)

        if self.spec.family == "esm3":
            from esm.sdk.api import SamplingConfig

            output = self.model.forward_and_sample(
                tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
            hidden = output.per_residue_embedding
        else:
            output = self.model.logits(
                tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            hidden = output.embeddings

        import torch

        return hidden if isinstance(hidden, torch.Tensor) else torch.as_tensor(hidden)


class HuggingFaceModel(ProteinLanguageModel):
    """Ankh, ESM-2 and ProtT5 through transformers.

    Ankh and ProtT5 are T5 encoder-decoders; only the encoder produces the
    per-residue representation, so ``T5EncoderModel`` is loaded rather than the
    full model.
    """

    def __init__(self, spec: PLMSpec, device: str):
        super().__init__(spec, device)
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
            if spec.family in ("ankh", "prot_t5"):
                from transformers import T5EncoderModel

                model = T5EncoderModel.from_pretrained(spec.model_id)
            else:
                model = AutoModel.from_pretrained(spec.model_id)
            self.model = model.to(device).eval()
        except ImportError as exc:
            raise DependencyError(
                f"{spec.name} needs transformers. {spec.install_hint}"
            ) from exc

    @_no_grad
    def _forward(self, sequence: str) -> "torch.Tensor":
        if self.spec.tokenize_as_chars:
            encoded = self.tokenizer(
                [list(sequence)],
                is_split_into_words=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                sequence, add_special_tokens=True, return_tensors="pt"
            )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return self.model(**encoded).last_hidden_state


_BACKENDS = {"esm": EsmSdkModel, "huggingface": HuggingFaceModel}


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------

_ESM_HINT = "Install it with: pip install 'plmol[esm]'"
_HF_HINT = "Install it with: pip install 'plmol[plm]'"

for _spec in (
    # Evolutionary Scale SDK. Both wrap the sequence in BOS/EOS.
    PLMSpec("esmc_300m", "esm", 960, "esmc_300m", "esmc", install_hint=_ESM_HINT),
    PLMSpec("esmc_600m", "esm", 1152, "esmc_600m", "esmc", install_hint=_ESM_HINT),
    PLMSpec("esm3-open", "esm", 1536, "esm3-open", "esm3", install_hint=_ESM_HINT),
    # Ankh is a T5 encoder-decoder; its tokenizer appends </s> and prepends
    # nothing, and expects one token per residue. Dimensions and token layout
    # are from the model cards, not measured here.
    PLMSpec(
        "ankh-base", "huggingface", 768, "ElnaggarLab/ankh-base", "ankh",
        prefix_tokens=0, suffix_tokens=1, tokenize_as_chars=True, install_hint=_HF_HINT,
    ),
    PLMSpec(
        "ankh-large", "huggingface", 1536, "ElnaggarLab/ankh-large", "ankh",
        prefix_tokens=0, suffix_tokens=1, tokenize_as_chars=True, install_hint=_HF_HINT,
    ),
    # ESM-2 needs no SDK, only transformers.
    PLMSpec(
        "esm2_t12_35m", "huggingface", 480, "facebook/esm2_t12_35M_UR50D", "esm2",
        install_hint=_HF_HINT,
    ),
    PLMSpec(
        "esm2_t33_650m", "huggingface", 1280, "facebook/esm2_t33_650M_UR50D", "esm2",
        install_hint=_HF_HINT,
    ),
    PLMSpec(
        "prot_t5_xl", "huggingface", 1024, "Rostlab/prot_t5_xl_uniref50", "prot_t5",
        prefix_tokens=0, suffix_tokens=1, tokenize_as_chars=True, install_hint=_HF_HINT,
    ),
):
    register_plm(_spec)
del _spec


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_plm(name: str, device: str = "auto") -> ProteinLanguageModel:
    """Load a registered model, reusing an already loaded one when possible.

    Args:
        name: A name from :func:`list_protein_language_models`.
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Raises:
        InputError: If the name is not registered.
        DependencyError: If the backend's package is not installed.
    """
    spec = get_plm_spec(name)
    resolved = resolve_device(device)
    key = (spec.name, resolved)

    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        _MODEL_CACHE.move_to_end(key)
        return cached

    backend = _BACKENDS.get(spec.backend)
    if backend is None:
        raise InputError(f"Model {name!r} names an unknown backend {spec.backend!r}.")

    logger.debug("Loading %s on %s", spec.name, resolved)
    model = backend(spec, resolved)

    _MODEL_CACHE[key] = model
    while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
        _MODEL_CACHE.popitem(last=False)
    return model


def clear_plm_cache() -> None:
    """Drop loaded models, freeing their memory."""
    _MODEL_CACHE.clear()


def embed_sequence(
    sequence: str,
    model: str = "esmc_600m",
    device: str = "auto",
) -> Dict[str, object]:
    """Per-residue embeddings for one sequence.

    Args:
        sequence: One-letter amino acid sequence.
        model: A name from :func:`list_protein_language_models`.
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        ``embeddings`` ``(L, D)``, ``bos`` ``(D,)``, ``eos`` ``(D,)``,
        ``model``, ``dim``, ``sequence``.
    """
    return load_plm(model, device=device).embed(sequence)


def embed_sequences(
    sequences: Dict[str, str],
    model: str = "esmc_600m",
    device: str = "auto",
) -> Dict[str, Dict[str, object]]:
    """Embed several named sequences with one loaded model.

    Args:
        sequences: Mapping of label (a chain id, say) to sequence.
        model: A name from :func:`list_protein_language_models`.
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
    """
    loaded = load_plm(model, device=device)
    return {label: loaded.embed(seq) for label, seq in sequences.items() if seq}
