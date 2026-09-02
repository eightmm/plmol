"""
Core Molecular Representations for plmol

Defines the base classes and specific implementations for Protein and Ligand.
Designed to hold multi-view data (Sequence, Graph, 3D, Surface).
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Iterable
import numpy as np

logger = logging.getLogger(__name__)


class TempFileOwner:
    """Ownership of temporary files created by alternative constructors.

    ``from_mmcif`` and friends write a converted structure to a temporary path
    and hand it to the object. Whoever owns that path has to delete it, and
    relying on ``__del__`` alone leaks on a reference cycle, so this also makes
    owners usable as context managers::

        with Protein.from_mmcif(path) as protein:
            ...
    """

    @property
    def _owned_temp_paths(self) -> List[str]:
        # Created on demand so a subclass that skips __init__ still works.
        paths = self.__dict__.get("_temp_paths")
        if paths is None:
            paths = []
            self.__dict__["_temp_paths"] = paths
        return paths

    @_owned_temp_paths.setter
    def _owned_temp_paths(self, value: List[str]) -> None:
        self.__dict__["_temp_paths"] = list(value)

    def cleanup(self) -> None:
        """Remove temporary files this object owns. Safe to call repeatedly."""
        for path in self._owned_temp_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                logger.debug("Could not remove temporary file %s", path)
        self._owned_temp_paths.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False

    def __del__(self) -> None:
        # Exceptions here are unraisable, and interpreter shutdown can have
        # already torn down what cleanup needs.
        try:
            self.cleanup()
        except Exception:
            pass


class BaseMolecule(TempFileOwner, ABC):
    """
    Abstract base class for all molecular entities.
    
    Holds multi-view representations:
    1. Sequence (1D)
    2. Graph (2D/Topology)
    3. Conformer/Coords (3D)
    4. Surface (Mesh)
    """
    
    def __init__(self):
        # 1D Representation
        self._sequence: Optional[str] = None
        
        # 2D/Graph Representation
        # Graph is stored as adjacency or edge list + node features
        self._graph: Optional[Dict[str, Any]] = None 
        
        # 3D Representation
        # Coords: (N, 3) numpy array
        self._coords: Optional[np.ndarray] = None
        self._atoms: List[str] = []  # Atom types/names
        
        # Surface Representation
        # Mesh data: vertices, faces, normals
        self._surface: Optional[Dict[str, np.ndarray]] = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {}

    @property
    def sequence(self) -> str:
        return self._sequence

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def has_3d(self) -> bool:
        return self._coords is not None

    @property
    def has_surface(self) -> bool:
        return self._surface is not None

    def set_surface(self, points: np.ndarray, normals: np.ndarray,
                     faces: Optional[np.ndarray] = None):
        """Store surface data with standardized keys."""
        self._surface = {
            "points": points,
            "normals": normals,
            "verts": points,
        }
        if faces is not None:
            self._surface["faces"] = faces

    def get_surface(self) -> Optional[Dict[str, np.ndarray]]:
        return self._surface

    @abstractmethod
    def featurize(self, mode: Union[str, Iterable[str]] = "all", **kwargs) -> Dict[str, Any]:
        """Generate features for the molecule."""
        pass
