from plmol import Complex, Protein
from plmol.errors import InputError
from plmol.ligand.featurizer import LigandFeaturizer
import numpy as np


def test_complex_ligand_features_from_sequence_protein():
    protein = Protein.from_sequence("ACDEFG")
    complex_obj = Complex.from_inputs(protein=protein, ligand="CCO")

    out = complex_obj.ligand(mode=["graph", "fingerprint", "smiles"])
    assert "graph" in out
    assert "fingerprint" in out
    assert "smiles" in out
    assert isinstance(out["graph"], dict)
    assert "node_features" in out["graph"]
    assert "adjacency" in out["graph"]
    assert "bond_mask" in out["graph"]
    assert "distance_matrix" in out["graph"]
    assert "distance_bounds" in out["graph"]
    assert isinstance(out["graph"]["node_features"], np.ndarray)
    assert isinstance(out["graph"]["adjacency"], np.ndarray)
    assert isinstance(out["graph"]["bond_mask"], np.ndarray)
    assert isinstance(out["graph"]["distance_matrix"], np.ndarray)
    assert isinstance(out["graph"]["distance_bounds"], np.ndarray)
    assert isinstance(out["graph"]["coords"], np.ndarray)
    assert out["graph"]["coords"].shape[1] == 3
    assert isinstance(out["fingerprint"]["ecfp4"], np.ndarray)
    assert isinstance(out["fingerprint"]["maccs"], np.ndarray)
    assert isinstance(out["fingerprint"]["descriptors"], np.ndarray)
    assert isinstance(out["fingerprint"]["ecfp6"], np.ndarray)
    assert isinstance(out["fingerprint"]["ecfp4_feature"], np.ndarray)
    assert isinstance(out["fingerprint"]["rdkit"], np.ndarray)
    assert isinstance(out["fingerprint"]["erg"], np.ndarray)
    assert out["graph"]["adjacency"].shape[-1] == 37
    assert out["graph"]["bond_mask"].shape[:2] == out["graph"]["adjacency"].shape[:2]
    assert out["graph"]["distance_matrix"].shape[:2] == out["graph"]["adjacency"].shape[:2]
    assert out["graph"]["distance_bounds"].shape[-1] == 2


def test_complex_protein_sequence_mode():
    protein = Protein.from_sequence("ACDEFG")
    complex_obj = Complex.from_inputs(protein=protein, ligand="CCO")

    out = complex_obj.protein(mode=["sequence"])
    assert "sequence" in out
    assert out["sequence"] == "ACDEFG"


def test_complex_ligand_cache_reuse():
    protein = Protein.from_sequence("ACDEFG")
    complex_obj = Complex.from_inputs(protein=protein, ligand="CCO")
    a = complex_obj.ligand(mode=["graph", "fingerprint"])
    b = complex_obj.ligand(mode=["graph", "fingerprint"])
    assert a is b


def test_complex_invalid_request_raises():
    protein = Protein.from_sequence("ACDEFG")
    complex_obj = Complex.from_inputs(protein=protein, ligand="CCO")
    try:
        complex_obj.featurize(requests=["invalid"])
    except InputError:
        return
    raise AssertionError("Expected InputError for invalid request")


def test_adjacency_to_bond_edges_conversion():
    protein = Protein.from_sequence("ACDEFG")
    complex_obj = Complex.from_inputs(protein=protein, ligand="CCO")
    graph = complex_obj.ligand(mode=["graph"])["graph"]
    edge_index, edge_features = LigandFeaturizer.adjacency_to_bond_edges(graph["adjacency"])
    assert edge_index.shape[0] == 2
    assert edge_features.shape[0] == edge_index.shape[1]
    assert edge_features.shape[1] == graph["adjacency"].shape[-1]
