"""Tests for exported names that no other test file exercises.

These cover the contract surface -- the spec dataclass, the error hierarchy, the
parser interface and the hierarchical data container -- rather than
featurization behaviour, which the per-module test files own.
"""

import dataclasses

import pytest
import torch

import plmol
from plmol import (
    DependencyError,
    FeatureError,
    FeatureSpec,
    FEATURE_SPECS,
    HierarchicalProteinData,
    InputError,
    MMCIFParser,
    PlmolError,
    StructureParser,
)
from plmol.parsers import PDBParser
from plmol.specs import normalize_modes


class TestExportsResolve:
    def test_every_declared_export_exists(self):
        missing = [name for name in plmol.__all__ if not hasattr(plmol, name)]
        assert missing == []

    def test_version_is_declared(self):
        assert isinstance(plmol.__version__, str)
        assert plmol.__version__


class TestFeatureSpec:
    def test_is_a_frozen_dataclass(self):
        assert dataclasses.is_dataclass(FeatureSpec)
        spec = FEATURE_SPECS["ligand"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "other"

    @pytest.mark.parametrize("name", ["ligand", "protein", "nucleic_acid", "interaction"])
    def test_defaults_are_allowed_modes(self, name):
        spec = FEATURE_SPECS[name]
        assert set(spec.default_modes) <= set(spec.allowed_modes)
        assert spec.name == name

    @pytest.mark.parametrize("name", ["ligand", "protein", "nucleic_acid"])
    def test_molecule_output_keys_are_modes(self, name):
        """For molecules output_keys names modes; interaction is the exception."""
        spec = FEATURE_SPECS[name]
        assert set(spec.output_keys) <= set(spec.allowed_modes)

    def test_interaction_output_keys_name_result_fields(self):
        """interaction has one mode; its output_keys are keys of that result."""
        spec = FEATURE_SPECS["interaction"]
        assert spec.allowed_modes == ("graph",)
        assert "edges" in spec.output_keys
        assert not set(spec.output_keys) <= set(spec.allowed_modes)

    def test_normalize_modes_accepts_a_string_or_a_sequence(self):
        spec = FEATURE_SPECS["ligand"]
        assert normalize_modes(spec, "graph") == ["graph"]
        assert set(normalize_modes(spec, ["graph", "descriptor"])) == {"graph", "descriptor"}

    def test_normalize_modes_rejects_an_unknown_mode(self):
        with pytest.raises(InputError):
            normalize_modes(FEATURE_SPECS["ligand"], "not_a_mode")


class TestErrorHierarchy:
    @pytest.mark.parametrize("error", [InputError, DependencyError, FeatureError])
    def test_all_errors_derive_from_the_base(self, error):
        assert issubclass(error, PlmolError)
        assert issubclass(error, Exception)

    def test_catching_the_base_catches_the_specific_ones(self):
        with pytest.raises(PlmolError):
            raise FeatureError("boom")

    def test_errors_carry_their_message(self):
        assert str(FeatureError("extraction failed")) == "extraction failed"


class TestStructureParserInterface:
    @pytest.mark.parametrize("parser", [PDBParser, MMCIFParser])
    def test_bundled_parsers_implement_the_interface(self, parser):
        assert issubclass(parser, StructureParser)

    def test_the_interface_itself_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            StructureParser()

    def test_an_incomplete_subclass_cannot_be_instantiated(self):
        class Incomplete(StructureParser):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_parsers_expose_the_documented_surface(self, example_pdb):
        parser = PDBParser(example_pdb)
        for attribute in ("protein_atoms", "all_atoms", "file_path"):
            assert hasattr(parser, attribute)
        for method in ("get_sequence", "get_sequence_by_chain", "get_atom_coords"):
            assert callable(getattr(parser, method))
        assert parser.get_sequence()
        assert parser.get_atom_coords().shape[1] == 3


def _minimal_hierarchical_data(num_atoms=4, num_residues=2):
    """A HierarchicalProteinData with no ESM tensors attached."""
    return HierarchicalProteinData(
        atom_tokens=torch.zeros(num_atoms, dtype=torch.long),
        atom_coords=torch.zeros(num_atoms, 3),
        atom_sasa=torch.zeros(num_atoms),
        atom_elements=torch.zeros(num_atoms, dtype=torch.long),
        atom_residue_types=torch.zeros(num_atoms, dtype=torch.long),
        atom_names=["CA"] * num_atoms,
        residue_features=torch.zeros(num_residues, 81),
        residue_ca_coords=torch.zeros(num_residues, 3),
        residue_sc_coords=torch.zeros(num_residues, 3),
        residue_names=["ALA"] * num_residues,
        residue_ids=[("A", i + 1) for i in range(num_residues)],
    )


class TestHierarchicalProteinData:
    """The container is testable without ESM; the featurizer that fills it is not."""

    def test_counts_follow_the_tensors(self):
        data = _minimal_hierarchical_data(num_atoms=6, num_residues=3)
        assert data.num_atoms == 6
        assert data.num_residues == 3

    def test_esm_tensors_are_optional(self):
        data = _minimal_hierarchical_data()
        assert data.has_esm is False
        assert data.esmc_embeddings is None
        assert data.esm3_embeddings is None

    def test_feature_dims_are_reported(self):
        dims = _minimal_hierarchical_data().get_feature_dims()
        assert isinstance(dims, dict)
        assert dims["residue_dim"] == 81

    def test_moving_to_cpu_is_a_no_op_that_returns_data(self):
        data = _minimal_hierarchical_data()
        moved = data.to("cpu")
        assert isinstance(moved, HierarchicalProteinData)
        assert moved.num_atoms == data.num_atoms

    def test_constructing_it_does_not_require_esm(self):
        """HierarchicalFeaturizer needs the esm package; the container does not."""
        assert _minimal_hierarchical_data().has_esm is False


class TestHierarchicalDataSerialization:
    """to_dict replaces the flattening the removed batch CLI used to do."""

    def test_round_trips_through_torch_save(self, tmp_path):
        import torch as _torch

        data = _minimal_hierarchical_data()
        path = tmp_path / "features.pt"
        _torch.save(data.to_dict(), path)
        loaded = _torch.load(path, weights_only=False)
        assert loaded["residue_features"].shape == data.residue_features.shape
        assert loaded["residue_names"] == data.residue_names

    def test_covers_every_field_and_keeps_unset_ones_none(self):
        data = _minimal_hierarchical_data()
        as_dict = data.to_dict()
        assert set(as_dict) == set(data.__dataclass_fields__)
        assert as_dict["esmc_embeddings"] is None
