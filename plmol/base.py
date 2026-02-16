"""
Core Molecular Representations for plmol

Defines the base classes and specific implementations for Protein and Ligand.
Designed to hold multi-view data (Sequence, Graph, 3D, Surface).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import numpy as np
from pathlib import Path

class BaseMolecule(ABC):
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

    def set_surface(self, points: np.ndarray, faces: np.ndarray, normals: np.ndarray):
        """Store surface mesh data with standardized keys."""
        self._surface = {
            "points": points,
            "faces": faces,
            "normals": normals,
            "verts": points,
        }

    def get_surface(self) -> Optional[Dict[str, np.ndarray]]:
        return self._surface

    @abstractmethod
    def featurize(self, mode: str = "all"):
        """Generate features for the molecule."""
        pass
