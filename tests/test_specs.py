from plmol.specs import FEATURE_SPECS, normalize_modes, normalize_requests
from plmol.errors import InputError


def test_normalize_requests_all():
    reqs = normalize_requests("all")
    assert reqs == ["ligand", "protein", "nucleic_acid", "interaction"]


def test_normalize_modes_ligand_defaults():
    modes = normalize_modes(FEATURE_SPECS["ligand"], None)
    assert "graph" in modes
    assert "fingerprint" in modes


def test_normalize_modes_all_uses_defaults():
    modes = normalize_modes(FEATURE_SPECS["ligand"], "all")
    assert modes == list(FEATURE_SPECS["ligand"].default_modes)


def test_normalize_modes_invalid():
    try:
        normalize_modes(FEATURE_SPECS["ligand"], ["bad_mode"])
    except InputError:
        return
    raise AssertionError("Expected InputError for invalid mode")


def test_interaction_spec_includes_supported_optional_outputs():
    keys = set(FEATURE_SPECS["interaction"].output_keys)
    assert {"protein_coords", "ligand_coords", "metal_features"} <= keys
    assert {"contact_edges", "contact_distances", "num_contacts"} <= keys
