"""Tests for the bond-wise (line) graph view of a ligand."""

import numpy as np
import pytest

from plmol import (
    BOND_VIEW_CHANNELS,
    InputError,
    Ligand,
    LigandFeaturizer,
    MoleculeFeaturizer,
    build_bond_graph,
)


def bond_graph(smiles: str) -> dict:
    return Ligand.from_smiles(smiles).featurize(mode="bond_graph")["bond_graph"]


def expected_edge_count(atom_to_bonds) -> int:
    """Line graph edges: both directions of every pair of bonds sharing an atom."""
    return 2 * sum(len(b) * (len(b) - 1) // 2 for b in atom_to_bonds)


# -- Shapes and counts -------------------------------------------------------


@pytest.mark.parametrize(
    "smiles,num_bonds",
    [
        ("CC(=O)Oc1ccccc1C(=O)O", 13),  # aspirin: 13 heavy atoms, one ring
        ("CCO", 2),
        ("CO", 1),
        ("C1CC1", 3),
    ],
)
def test_bond_count_matches_molecule(smiles, num_bonds):
    bg = bond_graph(smiles)
    assert bg["num_bonds"] == num_bonds
    assert bg["node_features"].shape == (num_bonds, len(BOND_VIEW_CHANNELS))
    assert bg["bond_index"].shape == (num_bonds, 2)
    assert bg["coords"].shape == (num_bonds, 3)
    assert bg["adjacency"].shape == (num_bonds, num_bonds)


def test_edge_count_follows_line_graph_identity():
    for smiles in ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "C1CC1", "CC(C)(C)C", "c1ccccc1"]:
        bg = bond_graph(smiles)
        assert bg["num_bond_edges"] == expected_edge_count(bg["atom_to_bonds"])
        assert bg["edge_index"].shape == (2, bg["num_bond_edges"])
        assert bg["edge_features"].shape == (bg["num_bond_edges"], 96)
        assert bg["edge_shared_atom"].shape == (bg["num_bond_edges"],)


def test_cyclopropane_is_a_triangle():
    """Every bond touches both others, so the line graph is K3."""
    bg = bond_graph("C1CC1")
    assert bg["num_bonds"] == 3
    assert bg["num_bond_edges"] == 6
    assert bg["adjacency"].sum() == 6
    assert not bg["adjacency"].diagonal().any()


def test_single_bond_molecule_has_no_edges():
    bg = bond_graph("CO")
    assert bg["num_bonds"] == 1
    assert bg["num_bond_edges"] == 0
    assert bg["edge_index"].shape == (2, 0)
    assert bg["edge_features"].shape == (0, 96)


@pytest.mark.parametrize("smiles", ["C", "[Na+]", "O"])
def test_bondless_molecule_yields_empty_graph(smiles):
    bg = bond_graph(smiles)
    assert bg["num_bonds"] == 0
    assert bg["num_bond_edges"] == 0
    assert bg["node_features"].shape == (0, len(BOND_VIEW_CHANNELS))
    assert bg["edge_index"].shape == (2, 0)
    assert bg["edge_features"].shape == (0, 96)
    assert bg["coords"].shape == (0, 3)
    assert bg["adjacency"].shape == (0, 0)


def test_disconnected_fragments_share_no_atoms():
    bg = bond_graph("CC.CC")
    assert bg["num_bonds"] == 2
    assert bg["num_bond_edges"] == 0


# -- Consistency with the atom graph -----------------------------------------


def test_node_features_come_from_atom_adjacency(example_sdf):
    lig = Ligand.from_sdf(example_sdf)
    graph = lig.featurize(mode="graph")["graph"]
    bg = lig.featurize(mode="bond_graph")["bond_graph"]
    begin, end = bg["bond_index"][:, 0], bg["bond_index"][:, 1]
    kept = list(BOND_VIEW_CHANNELS)
    assert np.array_equal(bg["node_features"], graph["adjacency"][begin, end][:, kept])


def test_edge_features_start_with_shared_atom_features(example_sdf):
    lig = Ligand.from_sdf(example_sdf)
    graph = lig.featurize(mode="graph")["graph"]
    bg = lig.featurize(mode="bond_graph")["bond_graph"]
    shared = bg["edge_shared_atom"]
    assert np.array_equal(bg["edge_features"][:, :94], graph["node_features"][shared])


def test_bond_coords_are_midpoints(example_sdf):
    lig = Ligand.from_sdf(example_sdf)
    graph = lig.featurize(mode="graph")["graph"]
    bg = lig.featurize(mode="bond_graph")["bond_graph"]
    begin, end = bg["bond_index"][:, 0], bg["bond_index"][:, 1]
    midpoints = (graph["coords"][begin] + graph["coords"][end]) / 2.0
    assert np.allclose(bg["coords"], midpoints, atol=1e-6)


# -- Structural invariants ---------------------------------------------------


def test_edge_index_is_symmetric():
    bg = bond_graph("CC(=O)Oc1ccccc1C(=O)O")
    edges = set(zip(bg["edge_index"][0].tolist(), bg["edge_index"][1].tolist()))
    assert all((dst, src) in edges for src, dst in edges)
    assert np.array_equal(bg["adjacency"], bg["adjacency"].T)


def test_shared_atom_belongs_to_both_bonds():
    bg = bond_graph("CC(=O)Oc1ccccc1C(=O)O")
    src, dst = bg["edge_index"]
    for k in range(bg["num_bond_edges"]):
        atom = bg["edge_shared_atom"][k]
        assert atom in bg["bond_index"][src[k]]
        assert atom in bg["bond_index"][dst[k]]


def test_atom_to_bonds_is_the_inverse_of_bond_index():
    bg = bond_graph("CC(=O)Oc1ccccc1C(=O)O")
    for atom_idx, incident in enumerate(bg["atom_to_bonds"]):
        for b in incident:
            assert atom_idx in bg["bond_index"][b]
    total = sum(len(b) for b in bg["atom_to_bonds"])
    assert total == 2 * bg["num_bonds"]


# -- Angle features ----------------------------------------------------------


def test_angles_are_in_range_with_3d(example_sdf):
    bg = Ligand.from_sdf(example_sdf).featurize(mode="bond_graph")["bond_graph"]
    cos_theta = bg["edge_features"][:, 94]
    theta = bg["edge_features"][:, 95]
    assert np.all(cos_theta >= -1.0) and np.all(cos_theta <= 1.0)
    assert np.all(theta >= 0.0) and np.all(theta <= 1.0)
    assert np.allclose(np.arccos(np.clip(cos_theta, -1, 1)) / np.pi, theta, atol=1e-5)
    # A real conformer must produce non-trivial angles.
    assert np.abs(cos_theta).max() > 0.0


def test_angles_are_computed_from_a_generated_conformer():
    """SMILES input gets a conformer by default, so angles are real."""
    bg = bond_graph("CCC")
    cos_theta = bg["edge_features"][:, 94]
    assert np.abs(cos_theta).max() > 0.0


def test_angles_are_zero_without_conformer():
    bg = Ligand.from_smiles("CCO").featurize(
        mode="bond_graph", graph_kwargs={"generate_conformer": False}
    )["bond_graph"]
    assert not bg["coords"].any()
    assert np.allclose(bg["edge_features"][:, 94:], 0.0)


# -- API surface -------------------------------------------------------------


def test_mode_is_not_in_defaults():
    assert "bond_graph" not in Ligand.from_smiles("CCO").featurize(mode="all")


def test_ligand_and_featurizer_paths_agree():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    from_ligand = Ligand.from_smiles(smiles).featurize(mode="bond_graph")["bond_graph"]
    from_featurizer = LigandFeaturizer(smiles).get_bond_graph()
    assert np.array_equal(
        from_ligand["node_features"], np.asarray(from_featurizer["node_features"])
    )
    assert np.array_equal(
        from_ligand["edge_index"], np.asarray(from_featurizer["edge_index"])
    )
    assert from_ligand["atom_to_bonds"] == from_featurizer["atom_to_bonds"]


def test_low_level_function_matches_the_mode():
    """build_bond_graph needs the canonicalized mol the atom graph came from."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    lig = Ligand.from_smiles(smiles)
    graph = lig.featurize(mode="graph")["graph"]
    from_mode = lig.featurize(mode="bond_graph")["bond_graph"]
    direct = build_bond_graph(
        MoleculeFeaturizer(smiles).get_rdkit_mol(),
        adjacency=graph["adjacency"],
        node_features=graph["node_features"],
        coords=graph["coords"],
    )
    assert np.array_equal(from_mode["node_features"], np.asarray(direct["node_features"]))
    assert np.array_equal(from_mode["edge_index"], np.asarray(direct["edge_index"]))


def test_mismatched_molecule_is_rejected():
    lig = Ligand.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    graph = lig.featurize(mode="graph")["graph"]
    with pytest.raises(InputError):
        build_bond_graph(
            MoleculeFeaturizer("CCO").get_rdkit_mol(),
            adjacency=graph["adjacency"],
            node_features=graph["node_features"],
            coords=graph["coords"],
        )


def test_reassigning_smiles_rebuilds_the_bond_graph():
    lig = Ligand.from_smiles("CCO")
    assert lig.featurize(mode="bond_graph")["bond_graph"]["num_bonds"] == 2
    lig.smiles = "CC(=O)Oc1ccccc1C(=O)O"
    assert lig.featurize(mode="bond_graph")["bond_graph"]["num_bonds"] == 13
