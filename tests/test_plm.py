"""Tests for protein language model embeddings.

The ESM and Hugging Face backends cannot run here -- neither package is
installed, and both would download gigabytes. So the model call itself is the
only untested line: everything around it (registry, dispatch, caching, special
token handling, chain splitting, mode wiring, CLI) is exercised through a fake
backend, and the missing-package path is asserted directly.
"""

import sys

import pytest
import numpy as np

from plmol import (
    DependencyError,
    InputError,
    Protein,
    embed_sequence,
    embed_sequences,
    list_protein_language_models,
    plm_dim,
)
from plmol.protein import plm


class FakeModel(plm.ProteinLanguageModel):
    """Deterministic stand-in: row i is filled with the value i."""

    loads = 0

    def __init__(self, spec, device):
        super().__init__(spec, device)
        FakeModel.loads += 1

    def _forward(self, sequence):
        rows = len(sequence) + self.spec.prefix_tokens + self.spec.suffix_tokens
        return np.repeat(np.arange(rows, dtype=np.float32),
            self.spec.dim
        ).reshape(rows, self.spec.dim)


@pytest.fixture
def fake_models():
    """Register fake models covering each special-token layout, then clean up."""
    plm._BACKENDS["fake"] = FakeModel
    specs = [
        plm.PLMSpec("fake-8", "fake", 8, "fake", "fake"),                       # BOS + EOS
        plm.PLMSpec("fake-eos", "fake", 4, "fake", "fake", prefix_tokens=0),    # EOS only
        plm.PLMSpec("fake-bare", "fake", 4, "fake", "fake",
                    prefix_tokens=0, suffix_tokens=0),                          # neither
    ]
    for spec in specs:
        plm.register_plm(spec)
    FakeModel.loads = 0
    yield
    plm.clear_plm_cache()
    for spec in specs:
        plm.PLM_REGISTRY.pop(spec.name, None)
    plm._BACKENDS.pop("fake", None)


# -- Registry -----------------------------------------------------------------


class TestRegistry:
    def test_the_documented_models_are_registered(self):
        names = list_protein_language_models()
        for expected in ("esmc_600m", "esm3-open", "ankh-base", "ankh-large",
                         "esm2_t33_650m", "prot_t5_xl"):
            assert expected in names

    @pytest.mark.parametrize(
        "name,dim",
        [("esmc_300m", 960), ("esmc_600m", 1152), ("esm3-open", 1536),
         ("ankh-base", 768), ("ankh-large", 1536), ("esm2_t33_650m", 1280)],
    )
    def test_dimensions_are_reported_without_loading(self, name, dim):
        assert plm_dim(name) == dim

    def test_every_entry_names_a_real_backend(self):
        for spec in plm.PLM_REGISTRY.values():
            assert spec.backend in plm._BACKENDS
            assert spec.dim > 0
            assert spec.model_id

    def test_unknown_model_lists_the_known_ones(self):
        with pytest.raises(InputError, match="Unknown protein language model"):
            plm_dim("not-a-model")
        with pytest.raises(InputError) as excinfo:
            plm.load_plm("not-a-model")
        assert "ankh-base" in str(excinfo.value)

    def test_device_auto_resolves_to_something_real(self):
        assert plm.resolve_device("auto") in ("cuda", "cpu")
        assert plm.resolve_device("cpu") == "cpu"


# -- Embedding contract -------------------------------------------------------


class TestEmbeddingContract:
    def test_shapes_and_keys(self, fake_models):
        result = embed_sequence("MKTIIALSY", "fake-8", device="cpu")
        assert set(result) == {"embeddings", "bos", "eos", "model", "dim", "sequence"}
        assert result["embeddings"].shape == (9, 8)
        assert result["bos"].shape == (8,)
        assert result["eos"].shape == (8,)
        assert result["dim"] == 8
        assert result["model"] == "fake-8"
        assert result["sequence"] == "MKTIIALSY"

    def test_residue_rows_are_aligned(self, fake_models):
        """Row i of embeddings must be residue i, not a special token."""
        result = embed_sequence("MKTIIALSY", "fake-8", device="cpu")
        # FakeModel row values are the raw index; with one BOS the first
        # residue is raw row 1.
        assert result["embeddings"][0, 0] == 1.0
        assert result["bos"][0] == 0.0
        assert result["eos"][0] == 10.0

    def test_eos_only_layout(self, fake_models):
        result = embed_sequence("MKT", "fake-eos", device="cpu")
        assert result["embeddings"].shape == (3, 4)
        assert result["embeddings"][0, 0] == 0.0
        assert np.all(result["bos"] == 0)
        assert result["eos"][0] == 3.0

    def test_model_without_special_tokens(self, fake_models):
        result = embed_sequence("MKT", "fake-bare", device="cpu")
        assert result["embeddings"].shape == (3, 4)
        assert np.all(result["bos"] == 0)
        assert np.all(result["eos"] == 0)

    def test_embeddings_are_float32(self, fake_models):
        result = embed_sequence("MKT", "fake-8", device="cpu")
        assert result["embeddings"].dtype == np.float32

    def test_empty_sequence_is_rejected(self, fake_models):
        with pytest.raises(InputError, match="empty sequence"):
            embed_sequence("", "fake-8", device="cpu")

    def test_unexpected_row_count_is_reported(self, fake_models):
        """A tokenizer layout that disagrees with the spec must not misalign."""
        model = plm.load_plm("fake-8", device="cpu")
        with pytest.raises(InputError, match="rows for a sequence"):
            plm._split_special_tokens(np.zeros((99, 8), dtype=np.float32), "MKT", model.spec)


class TestModelCache:
    def test_a_model_is_loaded_once_per_device(self, fake_models):
        embed_sequence("MKT", "fake-8", device="cpu")
        embed_sequence("AAAA", "fake-8", device="cpu")
        plm.load_plm("fake-8", device="cpu")
        assert FakeModel.loads == 1

    def test_clearing_the_cache_forces_a_reload(self, fake_models):
        plm.load_plm("fake-8", device="cpu")
        plm.clear_plm_cache()
        plm.load_plm("fake-8", device="cpu")
        assert FakeModel.loads == 2

    def test_several_sequences_share_one_load(self, fake_models):
        results = embed_sequences({"A": "MKT", "B": "AAAA"}, "fake-8", device="cpu")
        assert set(results) == {"A", "B"}
        assert results["A"]["embeddings"].shape == (3, 8)
        assert results["B"]["embeddings"].shape == (4, 8)
        assert FakeModel.loads == 1

    def test_empty_sequences_are_skipped(self, fake_models):
        results = embed_sequences({"A": "MKT", "B": ""}, "fake-8", device="cpu")
        assert set(results) == {"A"}


# -- Missing backends ---------------------------------------------------------


class TestMissingDependencies:
    """The first thing a user hits is the package not being installed."""

    @pytest.mark.parametrize("name", ["esmc_600m", "esm3-open"])
    def test_esm_models_report_how_to_install(self, monkeypatch, name):
        monkeypatch.setitem(sys.modules, "esm", None)
        monkeypatch.setitem(sys.modules, "esm.models.esmc", None)
        monkeypatch.setitem(sys.modules, "esm.models.esm3", None)
        plm.clear_plm_cache()
        with pytest.raises(DependencyError, match=r"pip install 'plmol\[esm\]'"):
            plm.load_plm(name, device="cpu")

    @pytest.mark.parametrize("name", ["ankh-base", "ankh-large", "esm2_t33_650m"])
    def test_huggingface_models_report_how_to_install(self, monkeypatch, name):
        monkeypatch.setitem(sys.modules, "transformers", None)
        plm.clear_plm_cache()
        with pytest.raises(DependencyError, match=r"pip install 'plmol\[plm\]'"):
            plm.load_plm(name, device="cpu")


# -- Protein.featurize wiring -------------------------------------------------


class TestEmbeddingMode:
    def test_single_sequence(self, fake_models, example_pdb):
        result = Protein.from_pdb(example_pdb).featurize(
            mode="embedding", embedding_kwargs={"model": "fake-8", "device": "cpu"}
        )["embedding"]
        assert result["dim"] == 8
        assert result["embeddings"].shape == (len(result["sequence"]), 8)

    def test_by_chain(self, fake_models, example_pdb):
        result = Protein.from_pdb(example_pdb).featurize(
            mode="embedding",
            embedding_kwargs={"model": "fake-8", "device": "cpu", "by_chain": True},
        )["embedding"]
        assert set(result["by_chain"]) == {"A", "B"}
        for chain in result["by_chain"].values():
            assert chain["embeddings"].shape == (len(chain["sequence"]), 8)
        assert result["dim"] == 8

    def test_chains_are_embedded_separately(self, fake_models, example_pdb):
        """Joined and per-chain must differ; a chain break is not a residue."""
        protein = Protein.from_pdb(example_pdb)
        joined = protein.featurize(
            mode="embedding", embedding_kwargs={"model": "fake-8", "device": "cpu"}
        )["embedding"]
        per_chain = protein.featurize(
            mode="embedding",
            embedding_kwargs={"model": "fake-8", "device": "cpu", "by_chain": True},
        )["embedding"]
        total = sum(c["embeddings"].shape[0] for c in per_chain["by_chain"].values())
        assert total == joined["embeddings"].shape[0]

    def test_not_part_of_all(self, example_pdb):
        assert "embedding" not in Protein.from_pdb(example_pdb).featurize(mode="all")

    def test_is_an_allowed_mode(self):
        from plmol.specs import PROTEIN_SPEC

        assert "embedding" in PROTEIN_SPEC.allowed_modes
        assert "embedding" not in PROTEIN_SPEC.output_keys

    def test_sequence_only_protein_can_be_embedded(self, fake_models):
        protein = Protein.from_sequence("MKTIIALSYIFCLVFA")
        result = protein.featurize(
            mode="embedding", embedding_kwargs={"model": "fake-8", "device": "cpu"}
        )["embedding"]
        assert result["embeddings"].shape == (16, 8)


# -- Hierarchical featurizer without ESM --------------------------------------


class TestHierarchicalWithoutEsm:
    def test_constructs_and_runs_without_the_esm_package(self, example_pdb):
        from plmol import HierarchicalFeaturizer

        featurizer = HierarchicalFeaturizer(esmc_model=None, esm3_model=None)
        data = featurizer.featurize(example_pdb)
        assert data.has_esm is False
        assert data.num_residues > 0
        assert data.residue_features.shape[0] == data.num_residues

    def test_one_model_can_be_enabled_alone(self, fake_models, example_pdb):
        from plmol import HierarchicalFeaturizer

        featurizer = HierarchicalFeaturizer(esmc_model="fake-8", esm3_model=None)
        data = featurizer.featurize(example_pdb)
        assert data.esmc_embeddings is not None
        assert data.esmc_embeddings.shape == (data.num_residues, 8)
        assert data.esm3_embeddings is None
        assert data.has_esm is True

    def test_embeddings_are_numpy_like_every_other_feature(self, fake_models, example_pdb):
        """The models are torch models; what comes out of them is not."""
        from plmol import HierarchicalFeaturizer

        data = HierarchicalFeaturizer(esmc_model="fake-8", esm3_model=None).featurize(example_pdb)
        for name in ("esmc_embeddings", "esmc_bos", "esmc_eos"):
            value = getattr(data, name)
            assert isinstance(value, np.ndarray), name
            assert value.dtype == np.float32, name

    def test_to_moves_the_whole_container_including_embeddings(self, fake_models, example_pdb):
        torch = pytest.importorskip("torch")
        from plmol import HierarchicalFeaturizer

        data = HierarchicalFeaturizer(esmc_model="fake-8", esm3_model=None).featurize(example_pdb)
        moved = data.to()
        assert isinstance(moved.residue_features, torch.Tensor)
        assert isinstance(moved.esmc_embeddings, torch.Tensor)
        assert moved.esmc_embeddings.dtype == torch.float32
        # Fields that were never set stay unset rather than becoming empty tensors.
        assert moved.esm3_embeddings is None
        assert np.array_equal(moved.residue_features.numpy(), data.residue_features)
