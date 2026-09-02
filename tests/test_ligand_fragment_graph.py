"""Tests for the fragment-level graph view of a ligand."""

import numpy as np
import pytest

from plmol import (
    InputError,
    Ligand,
    LigandFeaturizer,
    MoleculeFeaturizer,
    build_fragment_graph,
)
from plmol.ligand.fragment import fragment_molecule


def fragment_graph(smiles: str, **kwargs) -> dict:
    return Ligand.from_smiles(smiles).featurize(mode="fragment_graph", **kwargs)[
        "fragment_graph"
    ]


# -- Shapes and counts -------------------------------------------------------


@pytest.mark.parametrize(
    "smiles,num_fragments,num_edges",
    [
        ("CC(=O)Oc1ccccc1C(=O)O", 4, 6),  # aspirin: 3 rotatable bonds cut
        ("CCCCCC", 4, 6),
        ("CCO", 1, 0),  # no rotatable bonds
        ("C", 1, 0),
        ("C1CC1", 1, 0),
    ],
)
def test_fragment_and_edge_counts(smiles, num_fragments, num_edges):
    fg = fragment_graph(smiles)
    assert fg["num_fragments"] == num_fragments
    assert fg["num_fragment_edges"] == num_edges
    assert fg["node_features"].shape == (num_fragments, 62)
    assert fg["coords"].shape == (num_fragments, 3)
    assert fg["adjacency"].shape == (num_fragments, num_fragments)
    assert fg["edge_index"].shape == (2, num_edges)
    assert fg["edge_features"].shape == (num_edges, 39)
    assert fg["edge_cleaved_bond"].shape == (num_edges, 2)


def test_edges_are_both_directions_of_each_cleaved_bond():
    fg = fragment_graph("CC(=O)Oc1ccccc1C(=O)O")
    assert fg["num_fragment_edges"] % 2 == 0
    edges = set(zip(fg["edge_index"][0].tolist(), fg["edge_index"][1].tolist()))
    assert all((dst, src) in edges for src, dst in edges)
    assert np.array_equal(fg["adjacency"], fg["adjacency"].T)


def test_disconnected_molecule_without_rotatable_bonds_is_one_fragment():
    """Fragmentation cuts bonds; it does not split connected components."""
    fg = fragment_graph("CC.CC")
    assert fg["num_fragments"] == 1
    assert fg["num_fragment_edges"] == 0
    assert fg["fragment_smiles"] == ["CC.CC"]


# -- Consistency with the atom graph and fragment mode ------------------------


def test_edge_features_start_with_the_cleaved_bond(example_sdf):
    lig = Ligand.from_sdf(example_sdf)
    graph = lig.featurize(mode="graph")["graph"]
    fg = lig.featurize(mode="fragment_graph")["fragment_graph"]
    cut = fg["edge_cleaved_bond"]
    assert np.array_equal(fg["edge_features"][:, :37], graph["adjacency"][cut[:, 0], cut[:, 1]])


def test_every_cleaved_pair_is_a_real_bond(example_sdf):
    lig = Ligand.from_sdf(example_sdf)
    graph = lig.featurize(mode="graph")["graph"]
    fg = lig.featurize(mode="fragment_graph")["fragment_graph"]
    cut = fg["edge_cleaved_bond"]
    bond_type_onehot = graph["adjacency"][cut[:, 0], cut[:, 1], :4]
    assert np.all(bond_type_onehot.sum(axis=-1) > 0)


def test_coords_are_fragment_centroids(example_sdf):
    lig = Ligand.from_sdf(example_sdf)
    graph = lig.featurize(mode="graph")["graph"]
    fg = lig.featurize(mode="fragment_graph")["fragment_graph"]
    for frag_idx, atoms in enumerate(fg["fragment_atom_indices"]):
        assert np.allclose(fg["coords"][frag_idx], graph["coords"][atoms].mean(axis=0), atol=1e-5)


def test_matches_fragment_mode(example_sdf):
    lig = Ligand.from_sdf(example_sdf)
    fragment = lig.featurize(mode="fragment")["fragment"]
    fg = lig.featurize(mode="fragment_graph")["fragment_graph"]
    assert np.array_equal(fg["node_features"], fragment["fragment_features"])
    assert np.array_equal(fg["adjacency"], fragment["fragment_adjacency"].astype(bool))
    assert np.array_equal(fg["atom_to_fragment"], fragment["atom_to_fragment"])
    assert fg["fragment_smiles"] == fragment["fragment_smiles"]


def test_edge_endpoints_follow_atom_to_fragment():
    fg = fragment_graph("CC(=O)Oc1ccccc1C(=O)O")
    a2f, cut, edges = fg["atom_to_fragment"], fg["edge_cleaved_bond"], fg["edge_index"]
    for k in range(fg["num_fragment_edges"]):
        assert a2f[cut[k, 0]] == edges[0, k]
        assert a2f[cut[k, 1]] == edges[1, k]


# -- Small-fragment merging ---------------------------------------------------


@pytest.mark.parametrize("min_fragment_size", [1, 2, 3, 4])
def test_merging_keeps_edges_and_adjacency_in_step(min_fragment_size):
    fg = fragment_graph(
        "CCCCCCCCCC", fragment_kwargs={"min_fragment_size": min_fragment_size}
    )
    assert int(fg["adjacency"].sum()) == fg["num_fragment_edges"]
    assert fg["node_features"].shape[0] == fg["num_fragments"]


def test_merging_drops_internalized_bonds():
    loose = fragment_graph("CCCCCC", fragment_kwargs={"min_fragment_size": 1})
    merged = fragment_graph("CCCCCC", fragment_kwargs={"min_fragment_size": 4})
    assert merged["num_fragments"] < loose["num_fragments"]
    assert merged["num_fragment_edges"] < loose["num_fragment_edges"]


def test_brics_method_is_supported():
    fg = fragment_graph("CC(=O)Oc1ccccc1C(=O)O", fragment_kwargs={"method": "brics"})
    assert fg["node_features"].shape[0] == fg["num_fragments"]
    assert int(fg["adjacency"].sum()) == fg["num_fragment_edges"]


# -- Geometry features --------------------------------------------------------


def test_geometry_is_positive_with_3d(example_sdf):
    fg = Ligand.from_sdf(example_sdf).featurize(mode="fragment_graph")["fragment_graph"]
    centroid_distance = fg["edge_features"][:, 37]
    bond_length = fg["edge_features"][:, 38]
    assert np.all(centroid_distance > 0.0)
    # Cleaved bonds are ordinary covalent bonds.
    assert np.all(bond_length > 0.9) and np.all(bond_length < 2.5)


def test_geometry_is_zero_without_conformer():
    fg = fragment_graph(
        "CCCCCC", graph_kwargs={"generate_conformer": False}
    )
    assert not fg["coords"].any()
    assert np.allclose(fg["edge_features"][:, 37:], 0.0)


# -- API surface --------------------------------------------------------------


def test_mode_is_not_in_defaults():
    assert "fragment_graph" not in Ligand.from_smiles("CCO").featurize(mode="all")


def test_ligand_and_featurizer_paths_agree():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    from_ligand = Ligand.from_smiles(smiles).featurize(mode="fragment_graph")["fragment_graph"]
    from_featurizer = LigandFeaturizer(smiles).get_fragment_graph()
    assert np.array_equal(
        from_ligand["node_features"], np.asarray(from_featurizer["node_features"])
    )
    assert np.array_equal(
        from_ligand["edge_index"], np.asarray(from_featurizer["edge_index"])
    )
    assert from_ligand["fragment_smiles"] == from_featurizer["fragment_smiles"]


def test_mismatched_molecule_is_rejected():
    lig = Ligand.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    graph = lig.featurize(mode="graph")["graph"]
    other = MoleculeFeaturizer("CCO").get_rdkit_mol()
    with pytest.raises(InputError):
        build_fragment_graph(
            other,
            fragment_molecule(other),
            adjacency=graph["adjacency"],
            coords=graph["coords"],
        )


def test_missing_fragment_features_is_rejected():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    lig = Ligand.from_smiles(smiles)
    graph = lig.featurize(mode="graph")["graph"]
    mol = MoleculeFeaturizer(smiles).get_rdkit_mol()
    bare = fragment_molecule(mol, compute_features=False)
    with pytest.raises(InputError, match="compute_features"):
        build_fragment_graph(
            mol, bare, adjacency=graph["adjacency"], coords=graph["coords"]
        )


def test_reassigning_smiles_rebuilds_the_fragment_graph():
    lig = Ligand.from_smiles("CCO")
    assert lig.featurize(mode="fragment_graph")["fragment_graph"]["num_fragments"] == 1
    lig.smiles = "CC(=O)Oc1ccccc1C(=O)O"
    assert lig.featurize(mode="fragment_graph")["fragment_graph"]["num_fragments"] == 4


# -- Hierarchy ----------------------------------------------------------------


def test_three_views_describe_the_same_molecule(example_sdf):
    lig = Ligand.from_sdf(example_sdf)
    graph = lig.featurize(mode="graph")["graph"]
    bond = lig.featurize(mode="bond_graph")["bond_graph"]
    frag = lig.featurize(mode="fragment_graph")["fragment_graph"]
    num_atoms = graph["node_features"].shape[0]
    assert bond["bond_index"].max() < num_atoms
    assert frag["atom_to_fragment"].shape == (num_atoms,)
    assert frag["node_features"].shape[0] == frag["num_fragments"]
    assert sum(len(a) for a in frag["fragment_atom_indices"]) == num_atoms
