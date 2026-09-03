"""Tests for the normalized graph view, batching, and dimension lookup."""

import numpy as np
import pytest
import torch

from plmol import (
    BOND_VIEW_CHANNELS,
    BOND_VIEW_DROPPED_CHANNELS,
    FEATURE_DIMS,
    InputError,
    Ligand,
    Protein,
    as_graph,
    collate,
    feature_dims,
)

CANONICAL = {
    "node_features",
    "node_tokens",
    "node_vector_features",
    "edge_index",
    "edge_features",
    "edge_vector_features",
    "coords",
    "num_nodes",
    "num_edges",
    "source",
}

LIGAND_MODES = ["graph", "bond_graph", "fragment_graph"]
PROTEIN_MODES = ["graph", "atom_graph"]


@pytest.fixture(scope="module")
def ligand_views(request):
    sdf = request.path.parent.parent / "examples" / "10gs_ligand.sdf"
    lig = Ligand.from_sdf(str(sdf))
    return {mode: lig.featurize(mode=mode)[mode] for mode in LIGAND_MODES}


@pytest.fixture(scope="module")
def protein_views(request):
    pdb = request.path.parent.parent / "examples" / "10gs_protein.pdb"
    return {
        mode: Protein.from_pdb(str(pdb)).featurize(mode=mode)[mode]
        for mode in PROTEIN_MODES
    }


# -- Normalization ------------------------------------------------------------


def test_every_view_yields_the_same_keys(ligand_views, protein_views):
    for view in list(ligand_views.values()) + list(protein_views.values()):
        assert set(as_graph(view)) == CANONICAL


@pytest.mark.parametrize("mode", LIGAND_MODES)
def test_ligand_views_normalize(ligand_views, mode):
    g = as_graph(ligand_views[mode])
    assert g["edge_index"].shape == (2, g["num_edges"])
    assert g["edge_index"].dtype == np.int64
    assert g["node_features"].shape[0] == g["num_nodes"]
    assert g["edge_features"].shape[0] == g["num_edges"]
    assert g["coords"].shape == (g["num_nodes"], 3)
    if g["num_edges"]:
        assert int(g["edge_index"].max()) < g["num_nodes"]


@pytest.mark.parametrize("mode", PROTEIN_MODES)
def test_protein_views_normalize(protein_views, mode):
    g = as_graph(protein_views[mode])
    assert g["edge_index"].shape == (2, g["num_edges"])
    assert g["node_features"].shape[0] == g["num_nodes"]
    assert int(g["edge_index"].max()) < g["num_nodes"]


def test_dense_ligand_graph_becomes_sparse_edges(ligand_views):
    """The dense adjacency is unrolled with the same bond mask the library uses."""
    view = ligand_views["graph"]
    g = as_graph(view)
    mask = torch.as_tensor(view["bond_mask"]).clone()
    mask.fill_diagonal_(False)
    assert g["num_edges"] == int(mask.sum())
    adjacency = torch.as_tensor(view["adjacency"]).float()
    src, dst = g["edge_index"]
    kept = list(BOND_VIEW_CHANNELS)
    assert np.array_equal(g["edge_features"], adjacency[src, dst][:, kept])


def test_residue_graph_keeps_vectors_separate(protein_views):
    """SE(3) models need vector features unflattened."""
    g = as_graph(protein_views["graph"])
    assert g["node_vector_features"] is not None
    assert g["node_vector_features"].ndim == 3
    assert g["node_vector_features"].shape[-1] == 3
    assert g["node_vector_features"].shape[0] == g["num_nodes"]
    assert g["edge_vector_features"].shape[0] == g["num_edges"]


def test_atom_graph_exposes_tokens_and_continuous_features(protein_views):
    g = as_graph(protein_views["atom_graph"])
    assert g["node_tokens"] is not None
    assert g["node_tokens"].dtype == np.int64
    assert g["node_tokens"].shape == (g["num_nodes"], 3)
    assert g["node_features"].dtype == np.float32


def test_numpy_input_comes_back_as_tensors(ligand_views):
    """Ligand.featurize returns numpy; the normalized view is always torch."""
    assert isinstance(ligand_views["graph"]["node_features"], np.ndarray)
    g = as_graph(ligand_views["graph"])
    assert isinstance(g["node_features"], np.ndarray)
    assert isinstance(g["edge_features"], np.ndarray)


def test_unrecognized_input_is_rejected():
    with pytest.raises(InputError):
        as_graph({"something": 1})
    with pytest.raises(InputError):
        as_graph([1, 2, 3])


def test_normalizing_twice_is_stable(ligand_views):
    once = as_graph(ligand_views["bond_graph"])
    twice = as_graph(once)
    assert np.array_equal(once["edge_index"], twice["edge_index"])
    assert np.array_equal(once["node_features"], twice["node_features"])


# -- Batching -----------------------------------------------------------------


@pytest.fixture
def small_batch():
    smiles = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    return [
        as_graph(Ligand.from_smiles(s).featurize(mode="graph")["graph"]) for s in smiles
    ]


def test_collate_offsets_edges_per_graph(small_batch):
    batch = collate(small_batch)
    assert batch["num_nodes"] == sum(g["num_nodes"] for g in small_batch)
    assert batch["num_edges"] == sum(g["num_edges"] for g in small_batch)
    assert int(batch["edge_index"].max()) < batch["num_nodes"]
    for i in range(batch["num_graphs"]):
        lo, hi = int(batch["ptr"][i]), int(batch["ptr"][i + 1])
        selected = batch["batch"][batch["edge_index"][0]] == i
        edges = batch["edge_index"][:, selected]
        assert int(edges.min()) >= lo and int(edges.max()) < hi


def test_collate_batch_vector_and_ptr(small_batch):
    batch = collate(small_batch)
    assert batch["batch"].shape == (batch["num_nodes"],)
    assert batch["ptr"].shape == (batch["num_graphs"] + 1,)
    assert int(batch["ptr"][0]) == 0
    assert int(batch["ptr"][-1]) == batch["num_nodes"]
    for i, graph in enumerate(small_batch):
        assert int((batch["batch"] == i).sum()) == graph["num_nodes"]


def test_collate_accepts_raw_featurize_output():
    smiles = ["CCO", "c1ccccc1"]
    raw = [Ligand.from_smiles(s).featurize(mode="graph")["graph"] for s in smiles]
    normalized = [as_graph(v) for v in raw]
    assert np.array_equal(collate(raw)["edge_index"], collate(normalized)["edge_index"])


def test_collate_single_graph_is_a_passthrough(small_batch):
    batch = collate(small_batch[:1])
    assert batch["num_graphs"] == 1
    assert np.array_equal(batch["edge_index"], small_batch[0]["edge_index"])
    assert bool((batch["batch"] == 0).all())


def test_collate_rejects_incompatible_widths(ligand_views):
    graph = as_graph(ligand_views["graph"])
    bond = as_graph(ligand_views["bond_graph"])
    with pytest.raises(InputError, match="width"):
        collate([graph, bond])


def test_collate_rejects_an_empty_sequence():
    with pytest.raises(InputError):
        collate([])


def test_collate_preserves_vector_features(protein_views):
    graph = as_graph(protein_views["graph"])
    batch = collate([graph, graph])
    assert batch["node_vector_features"].shape[0] == 2 * graph["num_nodes"]
    assert batch["edge_vector_features"].shape[0] == 2 * graph["num_edges"]


def test_collate_preserves_tokens(protein_views):
    graph = as_graph(protein_views["atom_graph"])
    batch = collate([graph, graph])
    assert batch["node_tokens"].shape == (2 * graph["num_nodes"], 3)


# -- Dimensions ---------------------------------------------------------------


@pytest.mark.parametrize("mode", LIGAND_MODES)
def test_recorded_ligand_dims_match_reality(ligand_views, mode):
    g = as_graph(ligand_views[mode])
    dims = feature_dims("ligand", mode)
    assert g["node_features"].shape[-1] == dims["node_features"]
    assert g["edge_features"].shape[-1] == dims["edge_features"]


@pytest.mark.parametrize("mode", PROTEIN_MODES)
def test_recorded_protein_dims_match_reality(protein_views, mode):
    g = as_graph(protein_views[mode])
    dims = feature_dims("protein", mode)
    assert g["node_features"].shape[-1] == dims["node_features"]
    assert g["edge_features"].shape[-1] == dims["edge_features"]
    if "node_vector_features" in dims:
        assert g["node_vector_features"].shape[1] == dims["node_vector_features"]
    if "edge_vector_features" in dims:
        assert g["edge_vector_features"].shape[1] == dims["edge_vector_features"]
    if "node_tokens" in dims:
        assert g["node_tokens"].shape[-1] == dims["node_tokens"]


def test_feature_dims_returns_a_copy():
    dims = feature_dims("ligand", "graph")
    dims["node_features"] = 0
    assert feature_dims("ligand", "graph")["node_features"] == 98


def test_feature_dims_rejects_unknown_keys():
    with pytest.raises(InputError, match="No recorded dimensions"):
        feature_dims("ligand", "nope")
    with pytest.raises(InputError):
        feature_dims("nope", "graph")


def test_every_recorded_mode_is_reachable():
    """FEATURE_DIMS must not name modes the specs do not allow."""
    from plmol.specs import FEATURE_SPECS

    for molecule, modes in FEATURE_DIMS.items():
        allowed = set(FEATURE_SPECS[molecule].allowed_modes)
        for mode in modes:
            assert mode in allowed, f"{molecule}.{mode} is not an allowed mode"


class TestBondViewChannels:
    """Guards the claim that the dropped adjacency channels carry nothing.

    ``BOND_VIEW_CHANNELS`` removes 8 of the 10 pair channels from every sparse
    bond view. If a future change makes one of them informative on a bond,
    these tests fail rather than silently discarding signal.
    """

    SMILES = [
        "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "C1=CC2=CC=CC3=C2C(=C1)C=C3", "OP(=O)(O)O", "C/C=C/C", "C/C=C\\C",
        "FC(F)(F)c1ccc(Br)cc1I", "CSSC", "N#Cc1ccccc1", "c1ccsc1", "CC.CC",
    ]

    @classmethod
    def _bond_rows(cls):
        rows = []
        for smiles in cls.SMILES:
            view = Ligand.from_smiles(smiles).featurize(mode="graph")["graph"]
            adjacency = torch.as_tensor(view["adjacency"]).float()
            mask = torch.as_tensor(view["bond_mask"]).clone()
            mask.fill_diagonal_(False)
            src, dst = torch.where(mask)
            if src.numel():
                rows.append(adjacency[src, dst])
        return np.concatenate(rows)

    def test_the_two_sets_partition_the_adjacency(self, ligand_views):
        width = torch.as_tensor(ligand_views["graph"]["adjacency"]).shape[-1]
        assert set(BOND_VIEW_CHANNELS) | set(BOND_VIEW_DROPPED_CHANNELS) == set(range(width))
        assert not set(BOND_VIEW_CHANNELS) & set(BOND_VIEW_DROPPED_CHANNELS)

    def test_dropped_channels_are_constant_or_collinear_on_bonds(self):
        rows = self._bond_rows()
        constant, collinear = [], []
        for channel in BOND_VIEW_DROPPED_CHANNELS:
            column = rows[:, channel]
            if float(column.std()) == 0.0:
                constant.append(channel)
            else:
                # The only survivor is the euclidean distance, which is the
                # bond length that channel 20 already carries.
                other = rows[:, 20]
                corr = np.corrcoef(np.stack([column, other]))[0, 1]
                assert abs(float(corr)) > 0.9999, f"channel {channel} is informative"
                collinear.append(channel)
        assert constant == [27, 28, 29, 30, 31, 32, 35]
        assert collinear == [33]

    def test_kept_channels_are_not_all_constant(self):
        """A sanity check in the other direction: the kept set carries signal."""
        rows = self._bond_rows()
        kept = rows[:, list(BOND_VIEW_CHANNELS)]
        assert int((kept.std(axis=0) > 0).sum()) > 20

    def test_the_dense_adjacency_is_unchanged(self, ligand_views):
        """Only the sparse views drop channels; the dense contract is 37 wide."""
        adjacency = torch.as_tensor(ligand_views["graph"]["adjacency"])
        assert adjacency.shape[-1] == 37


class TestProteinAtomNodeHasNoComplement:
    def test_no_column_is_one_minus_burial_index(self, protein_views):
        view = protein_views["atom_graph"]
        g = as_graph(view)
        burial = torch.as_tensor(view["burial_index"]).float()
        for column in range(g["node_features"].shape[1]):
            assert not np.allclose(g["node_features"][:, column], 1.0 - burial), (
                f"column {column} is the complement of burial_index"
            )

    def test_relative_sasa_is_still_available_on_the_raw_view(self, protein_views):
        view = protein_views["atom_graph"]
        assert "relative_sasa" in view
        burial = torch.as_tensor(view["burial_index"]).float()
        relative = torch.as_tensor(view["relative_sasa"]).float()
        assert np.allclose(burial, 1.0 - relative)


class TestNucleicAcidViews:
    """as_graph claims to cover every plmol graph view, so it must cover these."""

    NA_MODES = ["graph", "atom_graph"]

    @pytest.fixture
    def na_views(self, dna_pdb):
        from plmol import NucleicAcid

        return {
            mode: NucleicAcid.from_pdb(dna_pdb).featurize(mode=mode)[mode]
            for mode in self.NA_MODES
        }

    @pytest.mark.parametrize("mode", NA_MODES)
    def test_normalizes_to_the_canonical_shape(self, na_views, mode):
        g = as_graph(na_views[mode])
        assert set(g) == CANONICAL
        assert g["edge_index"].shape == (2, g["num_edges"])
        assert g["node_features"].shape[0] == g["num_nodes"]
        assert g["edge_features"].shape[0] == g["num_edges"]
        assert g["coords"].shape == (g["num_nodes"], 3)
        assert int(g["edge_index"].max()) < g["num_nodes"]

    @pytest.mark.parametrize("mode", NA_MODES)
    def test_recorded_dims_match_reality(self, na_views, mode):
        g = as_graph(na_views[mode])
        dims = feature_dims("nucleic_acid", mode)
        assert g["node_features"].shape[-1] == dims["node_features"]
        assert g["edge_features"].shape[-1] == dims["edge_features"]
        assert g["node_tokens"].shape[-1] == dims["node_tokens"]

    def test_atom_graph_nodes_are_purely_token_valued(self, na_views):
        g = as_graph(na_views["atom_graph"])
        assert g["node_features"].shape[1] == 0
        assert g["node_tokens"] is not None

    def test_sources_are_distinguishable(self, na_views):
        assert as_graph(na_views["graph"])["source"] == "nucleic_residue_graph"
        assert as_graph(na_views["atom_graph"])["source"] == "nucleic_atom_graph"

    def test_collate_works_across_nucleic_graphs(self, na_views):
        g = as_graph(na_views["graph"])
        batch = collate([g, g])
        assert batch["num_nodes"] == 2 * g["num_nodes"]
        assert int(batch["edge_index"].max()) < batch["num_nodes"]

    def test_backbone_is_not_a_graph(self, dna_pdb):
        from plmol import NucleicAcid

        backbone = NucleicAcid.from_pdb(dna_pdb).featurize(mode="backbone")["backbone"]
        with pytest.raises(InputError, match="not graphs"):
            as_graph(backbone)
