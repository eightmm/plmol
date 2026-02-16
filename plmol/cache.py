"""Small cache utilities used across featurization layers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Generic, Iterator, MutableMapping, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Simple bounded LRU cache."""

    def __init__(self, max_size: int = 128):
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = int(max_size)
        self._data: "OrderedDict[K, V]" = OrderedDict()

    def __contains__(self, key: K) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: K, default: V | None = None):  # type: ignore[override]
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def to_dict(self) -> Dict[K, V]:
        return dict(self._data)
