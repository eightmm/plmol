"""The array operations plmol used to reach into torch for.

plmol computes features; it does not train anything. Everything it asked torch
for was construction and shuffling -- no autograd, no modules, no device. Those
all have numpy spellings, but a few differ from the obvious one in ways that
change results, so they live here once instead of being re-derived per file:

- ``normalize`` divides by ``max(norm, eps)``, which is what
  ``torch.nn.functional.normalize`` does. Dividing by ``norm + eps`` is the
  usual guess and is wrong for every vector that is not tiny.
- ``pad_last`` pads only the final axis, which is what ``F.pad`` with a
  two-element pad does; ``np.pad`` given the same pair pads every axis.
- ``pairwise_distances`` subtracts directly rather than expanding
  ``|a|^2 + |b|^2 - 2ab``. torch.cdist switches to the expansion above 25 rows,
  where it loses precision on raw crystallographic coordinates.

``to_torch`` is the other direction: plmol returns numpy, and this hands the
whole result to torch in one call for models that want tensors.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .errors import DependencyError

#: Default float width. torch defaulted to 32-bit; numpy defaults to 64-bit, so
#: every array plmol builds says so explicitly and this is the name for it.
FLOAT = np.float32

#: Index width for tokens, edge indices and the like.
INT = np.int64


def normalize(vectors: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Unit vectors along *axis*, guarding zero-length rows.

    Mirrors ``torch.nn.functional.normalize``: the divisor is
    ``max(norm, eps)``, so a zero vector comes back as zero rather than as
    something scaled by ``1/eps``.
    """
    vectors = np.asarray(vectors)
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(norm, eps)


def pad_last(array: np.ndarray, before: int, after: int, value: float = 0.0) -> np.ndarray:
    """Pad only the last axis, as ``F.pad(x, (before, after))`` does."""
    widths = [(0, 0)] * array.ndim
    widths[-1] = (before, after)
    return np.pad(array, widths, mode="constant", constant_values=value)


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Euclidean distance between every row of *left* and every row of *right*.

    Leading axes broadcast, so ``(B, P, 3)`` against ``(B, R, 3)`` gives
    ``(B, P, R)``. The subtraction is direct: the ``|a|^2 + |b|^2 - 2ab``
    expansion cancels catastrophically in float32 on coordinates tens of
    Angstrom from the origin.
    """
    return np.linalg.norm(left[..., :, None, :] - right[..., None, :, :], axis=-1)


def one_hot(indices: np.ndarray, num_classes: int, dtype=FLOAT) -> np.ndarray:
    """``(..., num_classes)`` one-hot rows, as ``F.one_hot`` produces."""
    indices = np.asarray(indices, dtype=INT)
    out = np.zeros(indices.shape + (num_classes,), dtype=dtype)
    np.put_along_axis(out.reshape(-1, num_classes), indices.reshape(-1, 1), 1, axis=1)
    return out


def to_numpy(value: Any) -> Any:
    """Whatever *value* holds, with every tensor turned into a numpy array.

    Walks dicts, lists, tuples and dataclasses. Anything else is returned as it
    came.
    """
    if hasattr(value, "detach"):
        if getattr(value, "is_sparse", False):
            value = value.to_dense()
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_numpy(item) for item in value)
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        import dataclasses

        return dataclasses.replace(
            value, **{f: to_numpy(getattr(value, f)) for f in value.__dataclass_fields__}
        )
    return value


def to_torch(value: Any, device: Optional[str] = None) -> Any:
    """Whatever *value* holds, with every numpy array turned into a tensor.

    plmol's featurizers return numpy. This is the one call that hands a whole
    result to a torch model::

        graph = to_torch(protein.featurize(mode="graph"))

    Walks dicts, lists, tuples and dataclasses, leaving everything else alone.

    Args:
        value: Any featurizer output, or a piece of one.
        device: Optional torch device to place the tensors on.

    Raises:
        DependencyError: If torch is not installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise DependencyError(
            "to_torch needs PyTorch. Install it with `pip install torch`, or "
            "use the numpy arrays the featurizers already return."
        ) from exc

    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            # from_numpy shares memory, and warns on a read-only array while
            # handing back a tensor that must not be written. Copy those.
            if not item.flags.writeable or not item.flags.c_contiguous:
                item = np.array(item, copy=True, order="C")
            tensor = torch.from_numpy(item)
            return tensor.to(device) if device is not None else tensor
        if isinstance(item, dict):
            return {key: convert(sub) for key, sub in item.items()}
        if isinstance(item, list):
            return [convert(sub) for sub in item]
        if isinstance(item, tuple):
            return tuple(convert(sub) for sub in item)
        if hasattr(item, "__dataclass_fields__") and not isinstance(item, type):
            import dataclasses

            return dataclasses.replace(
                item, **{f: convert(getattr(item, f)) for f in item.__dataclass_fields__}
            )
        return item

    return convert(value)
