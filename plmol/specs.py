"""Feature specs/schemas for stable request/response contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from .errors import InputError


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    allowed_modes: tuple[str, ...]
    default_modes: tuple[str, ...]
    output_keys: tuple[str, ...]


LIGAND_SPEC = FeatureSpec(
    name="ligand",
    allowed_modes=("graph", "surface", "voxel", "fingerprint", "smiles", "sequence", "all"),
    default_modes=("graph", "fingerprint", "smiles", "sequence"),
    output_keys=("graph", "surface", "voxel", "fingerprint", "smiles", "sequence"),
)

PROTEIN_SPEC = FeatureSpec(
    name="protein",
    allowed_modes=("graph", "surface", "voxel", "sequence", "backbone", "all"),
    default_modes=("graph", "sequence"),
    output_keys=("graph", "surface", "voxel", "sequence", "backbone"),
)

INTERACTION_SPEC = FeatureSpec(
    name="interaction",
    allowed_modes=("graph",),
    default_modes=("graph",),
    output_keys=("edges", "edge_features", "interactions", "metadata"),
)

FEATURE_SPECS: Mapping[str, FeatureSpec] = {
    "ligand": LIGAND_SPEC,
    "protein": PROTEIN_SPEC,
    "interaction": INTERACTION_SPEC,
}


def normalize_modes(spec: FeatureSpec, mode: str | Iterable[str] | None) -> List[str]:
    if mode is None:
        return list(spec.default_modes)
    if isinstance(mode, str):
        modes = [mode.lower()]
    else:
        modes = [str(m).lower() for m in mode]
    if any(m == "all" for m in modes):
        return list(spec.allowed_modes[:-1])
    invalid = [m for m in modes if m not in spec.allowed_modes]
    if invalid:
        raise InputError(
            f"Unsupported mode(s) for {spec.name}: {invalid}. "
            f"Allowed: {list(spec.allowed_modes)}"
        )
    return modes


def normalize_requests(requests: str | Iterable[str]) -> List[str]:
    if isinstance(requests, str):
        reqs = [requests.lower()]
    else:
        reqs = [str(r).lower() for r in requests]
    if any(r == "all" for r in reqs):
        return ["ligand", "protein", "interaction"]
    invalid = [r for r in reqs if r not in FEATURE_SPECS]
    if invalid:
        raise InputError(f"Unsupported request(s): {invalid}. Allowed: {list(FEATURE_SPECS)}")
    return reqs
