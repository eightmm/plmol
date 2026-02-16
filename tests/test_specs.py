from plmol.specs import FEATURE_SPECS, normalize_modes, normalize_requests
from plmol.errors import InputError


def test_normalize_requests_all():
    reqs = normalize_requests("all")
    assert reqs == ["ligand", "protein", "interaction"]


def test_normalize_modes_ligand_defaults():
    modes = normalize_modes(FEATURE_SPECS["ligand"], None)
    assert "graph" in modes
    assert "fingerprint" in modes


def test_normalize_modes_invalid():
    try:
        normalize_modes(FEATURE_SPECS["ligand"], ["bad_mode"])
    except InputError:
        return
    raise AssertionError("Expected InputError for invalid mode")
