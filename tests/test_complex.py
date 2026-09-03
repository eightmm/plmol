"""Tests for plmol/complex.py — Complex / MolecularComplex class."""

import pytest
import numpy as np

from plmol import Complex, MolecularComplex, NucleicAcid, Protein
from plmol.errors import InputError
from plmol.ligand.core import Ligand
from plmol.ligand.featurizer import LigandFeaturizer
from plmol.complex import _freeze


# ---------------------------------------------------------------------------
# _freeze helper
# ---------------------------------------------------------------------------

class TestFreeze:
    def test_dict(self):
        result = _freeze({"b": 2, "a": 1})
        assert isinstance(result, tuple)
        # Should be sorted by key
        assert result[0][0] == "a"

    def test_list(self):
        result = _freeze([1, 2, 3])
        assert result == (1, 2, 3)

    def test_set(self):
        result = _freeze({3, 1, 2})
        assert result == (1, 2, 3)

    def test_nested(self):
        result = _freeze({"key": [1, {"inner": 2}]})
        assert isinstance(result, tuple)

    def test_scalar(self):
        assert _freeze(42) == 42
        assert _freeze("hello") == "hello"
        assert _freeze(None) is None


# ---------------------------------------------------------------------------
# MolecularComplex alias and construction
# ---------------------------------------------------------------------------

def test_molecular_complex_is_alias():
    assert Complex is MolecularComplex


def test_from_inputs_protein_and_ligand():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    assert "protein" in mc.molecules
    assert "ligand" in mc.molecules
    assert isinstance(mc.molecules["protein"], Protein)
    assert isinstance(mc.molecules["ligand"], Ligand)


def test_protein_obj_property():
    protein = Protein.from_sequence("ACDEFG")
    mc = MolecularComplex.from_inputs(protein=protein, ligand="CCO")
    assert mc.protein_obj is mc.molecules["protein"]


def test_ligand_obj_property():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    assert mc.ligand_obj is mc.molecules["ligand"]


def test_protein_obj_setter():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    new_protein = Protein.from_sequence("MKLV")
    mc.protein_obj = new_protein
    assert mc.molecules["protein"] is new_protein


def test_ligand_obj_setter_clears_cache():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    _ = mc.ligand(mode=["graph"])
    mc.ligand_obj = Ligand.from_smiles("c1ccccc1")
    result = mc.ligand(mode=["graph"])
    assert "graph" in result


def test_extra_molecule_via_kwargs():
    protein = Protein.from_sequence("ACDEFG")
    ligand = Ligand.from_smiles("CCO")
    extra = Ligand.from_smiles("c1ccccc1")
    mc = MolecularComplex.from_inputs(protein=protein, ligand=ligand, cofactor=extra)
    assert "cofactor" in mc.molecules
    assert mc.molecules["cofactor"] is extra


def test_molecules_dict_direct():
    protein = Protein.from_sequence("ACDEFG")
    ligand = Ligand.from_smiles("CCO")
    mc = MolecularComplex(molecules={"protein": protein, "ligand": ligand})
    assert mc.protein_obj is protein
    assert mc.ligand_obj is ligand


def test_ligand_missing_raises():
    mc = MolecularComplex(molecules={"protein": Protein.from_sequence("ACDEFG")})
    with pytest.raises(InputError):
        mc.ligand()


def test_protein_missing_raises():
    mc = MolecularComplex(molecules={"ligand": Ligand.from_smiles("CCO")})
    with pytest.raises(InputError):
        mc.protein()


def test_featurize_protein_and_ligand():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    out = mc.featurize(
        requests=["protein", "ligand"],
        protein_kwargs={"mode": ["sequence"]},
        ligand_kwargs={"mode": ["smiles"]},
    )
    assert "protein" in out
    assert "ligand" in out
    assert "sequence" in out["protein"]


def test_featurize_nucleic_acid_request():
    mc = MolecularComplex.from_inputs(
        protein=Protein.from_sequence("ACDEFG"),
        ligand="CCO",
        nucleic_acid=NucleicAcid.from_sequence("ATGC"),
    )
    out = mc.featurize(
        requests=["nucleic_acid"],
        nucleic_acid_kwargs={"mode": ["sequence"]},
    )
    assert "nucleic_acid" in out
    assert "sequence" in out["nucleic_acid"]


def test_nucleic_acid_all_sequence_only():
    mc = MolecularComplex.from_inputs(nucleic_acid=NucleicAcid.from_sequence("ATGC"))
    out = mc.nucleic_acid(mode="all")
    assert list(out) == ["sequence"]


def test_featurize_all_includes_present_nucleic_acid_without_requiring_it():
    mc = MolecularComplex.from_inputs(
        protein=Protein.from_sequence("ACDEFG"),
        ligand="CCO",
        nucleic_acid=NucleicAcid.from_sequence("ATGC"),
    )
    out = mc.featurize(requests="all", interaction_kwargs={"knn_cutoff": None})
    assert "nucleic_acid" in out


def test_set_ligand_clears_cache():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    a = mc.ligand(mode=["graph"])
    mc.set_ligand("c1ccccc1")
    b = mc.ligand(mode=["graph"])
    assert a is not b


def test_set_protein_clears_cache():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    _ = mc.protein(mode=["sequence"])
    mc.set_protein(Protein.from_sequence("MKLV"))
    out = mc.protein(mode=["sequence"])
    assert out["sequence"] == "MKLV"


def test_cache_size_respected():
    mc = MolecularComplex.from_inputs(
        protein=Protein.from_sequence("ACDEFG"), ligand="CCO", cache_size=4
    )
    assert mc._cache.max_size == 4


def test_protein_obj_setter_to_none_removes_key():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    mc.protein_obj = None
    assert "protein" not in mc.molecules


def test_ligand_obj_setter_to_none_removes_key():
    mc = MolecularComplex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand="CCO")
    mc.ligand_obj = None
    assert "ligand" not in mc.molecules


# ---------------------------------------------------------------------------
# Complex.from_inputs / from_files
# ---------------------------------------------------------------------------

class TestComplexFromInputs:
    def test_sequence_and_smiles(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        assert c.ligand_obj is not None
        assert c.protein_obj is not None

    def test_from_files(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        assert c.ligand_obj is not None
        assert c.protein_obj is not None


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------

class TestComplexSetters:
    def test_set_ligand(self, ethanol_smiles, aspirin_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        c.ligand(mode=["smiles"])
        c.set_ligand(aspirin_smiles)
        result = c.ligand(mode=["smiles"])
        assert result["smiles"] != ethanol_smiles

    def test_set_protein(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        c.protein(mode=["sequence"])
        new_protein = Protein.from_sequence("WWWWWW")
        c.set_protein(new_protein)
        result = c.protein(mode=["sequence"])
        assert result["sequence"] == "WWWWWW"


# ---------------------------------------------------------------------------
# Ligand featurization
# ---------------------------------------------------------------------------

class TestComplexLigand:
    def test_no_ligand_raises(self):
        c = Complex.from_inputs(protein=Protein.from_sequence("AAA"))
        with pytest.raises(InputError, match="Ligand"):
            c.ligand()

    def test_graph_and_fingerprint(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        result = c.ligand(mode=["graph", "fingerprint"])
        assert "graph" in result
        assert "fingerprint" in result

    def test_cache_hit(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        a = c.ligand(mode=["graph"])
        b = c.ligand(mode=["graph"])
        assert a is b

    def test_all_uses_ligand_default_modes(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        result = c.ligand(mode="all")
        assert set(result) == {"graph", "fingerprint", "smiles", "sequence"}


# ---------------------------------------------------------------------------
# Protein featurization
# ---------------------------------------------------------------------------

class TestComplexProtein:
    def test_no_protein_raises(self):
        c = Complex.from_inputs(ligand=Ligand.from_smiles("CCO"))
        with pytest.raises(InputError, match="Protein"):
            c.protein()

    def test_sequence_mode(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        result = c.protein(mode=["sequence"])
        assert result["sequence"] == "ACDEFG"

    def test_all_sequence_only_protein(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("ACDEFG"), ligand=ethanol_smiles)
        result = c.protein(mode="all")
        assert result == {"sequence": "ACDEFG"}

    def test_graph_mode_real(self, example_pdb, ethanol_smiles):
        c = Complex.from_inputs(protein=example_pdb, ligand=ethanol_smiles)
        result = c.protein(mode=["graph"])
        assert "graph" in result


# ---------------------------------------------------------------------------
# Interaction featurization
# ---------------------------------------------------------------------------

class TestComplexInteraction:
    def test_missing_components(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"))
        with pytest.raises(InputError, match="both"):
            c.interaction()

    def test_real_interaction(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        result = c.interaction()
        assert isinstance(result, dict)
        assert "edges" in result
        assert "protein_coords" in result
        assert "ligand_coords" in result
        assert result["protein_coords"].shape[1] == 3
        assert result["ligand_coords"].shape[1] == 3
        assert "metal_sites" in result
        assert "metal_features" in result

    def test_with_pocket_cutoff(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        result = c.interaction(pocket_cutoff=6.0)
        assert isinstance(result, dict)

    def test_with_contact_edges(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        result = c.interaction(include_contacts=True, contact_cutoff=4.5, knn_cutoff=2)
        assert "contact_edges" in result
        assert "contact_distances" in result
        assert result["contact_edges"].shape[0] == 2
        assert result["num_contacts"] == result["contact_edges"].shape[1]


# ---------------------------------------------------------------------------
# featurize() API
# ---------------------------------------------------------------------------

class TestComplexFeaturize:
    def test_all_requests(self, example_pdb, example_sdf):
        c = Complex.from_files(example_pdb, example_sdf)
        result = c.featurize(requests="all")
        assert "ligand" in result
        assert "protein" in result
        assert "interaction" in result

    def test_single_request(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        result = c.featurize(requests=["ligand"])
        assert "ligand" in result
        assert "protein" not in result

    def test_invalid_request(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        with pytest.raises(InputError):
            c.featurize(requests=["bad_request"])

    def test_with_kwargs(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        result = c.featurize(
            requests=["ligand"],
            ligand_kwargs={"mode": ["smiles"]},
        )
        assert "smiles" in result["ligand"]

    def test_explicit_missing_nucleic_acid_raises(self, ethanol_smiles):
        c = Complex.from_inputs(protein=Protein.from_sequence("A"), ligand=ethanol_smiles)
        with pytest.raises(InputError):
            c.featurize(requests=["nucleic_acid"])


# ---------------------------------------------------------------------------
# Original standalone tests (test_complex.py)
# ---------------------------------------------------------------------------

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


class TestTheTwoBlocksCanBeJoined:
    """A complex result numbers the ligand's atoms twice.

    ``graph`` mode canonicalises the atom order -- that is what lets the bond
    and fragment graphs line up with it -- while the interaction featurizer
    indexes the molecule as it was handed over, which for a file is the file's
    own order. Both are self-consistent; nothing said they disagree, so
    joining an interaction edge to a ligand node paired the wrong atoms.
    """

    @pytest.fixture
    def both(self, example_pdb, example_sdf):
        from plmol import Ligand, Protein
        from plmol.complex import MolecularComplex

        return MolecularComplex(
            molecules={"protein": Protein.from_pdb(example_pdb),
                       "ligand": Ligand.from_sdf(example_sdf)}
        ).featurize(requests=["ligand", "interaction"])

    def test_the_orders_really_do_differ(self, both):
        """If they ever stop differing the mapping becomes the identity and
        this test says so rather than quietly passing."""
        graph = np.asarray(both["ligand"]["graph"]["coords"])
        inter = np.asarray(both["interaction"]["ligand_coords"])
        assert graph.shape == inter.shape
        assert not np.allclose(graph, inter, atol=1e-4)

    def test_the_mapping_lines_them_up(self, both):
        order = np.asarray(both["interaction"]["ligand_atom_order"])
        graph = np.asarray(both["ligand"]["graph"]["coords"])
        inter = np.asarray(both["interaction"]["ligand_coords"])
        assert order.shape == (graph.shape[0],)
        assert np.allclose(inter[order], graph, atol=1e-4)

    def test_it_is_a_permutation(self, both):
        order = np.asarray(both["interaction"]["ligand_atom_order"])
        assert sorted(order.tolist()) == list(range(len(order))), "no atom twice, none missing"

    def test_an_interaction_edge_reaches_a_real_graph_node(self, both):
        order = np.asarray(both["interaction"]["ligand_atom_order"])
        inverse = np.full(len(order), -1, dtype=np.int64)
        inverse[order] = np.arange(len(order))
        ligand_endpoints = np.asarray(both["interaction"]["edges"])[1]
        nodes = inverse[ligand_endpoints]
        assert (nodes >= 0).all()
        assert nodes.max() < np.asarray(both["ligand"]["graph"]["coords"]).shape[0]

    def test_explicit_hydrogens_are_marked_rather_than_guessed(self):
        """The graph keeps hydrogens when the molecule carries them and the
        interaction block never does, so those nodes have no counterpart."""
        from rdkit import Chem

        from plmol.complex import _ligand_graph_to_interaction_index

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        order = _ligand_graph_to_interaction_index(mol)
        assert order.shape == (mol.GetNumAtoms(),)
        assert (order == -1).sum() == mol.GetNumAtoms() - 3, "one per hydrogen"
        assert sorted(order[order >= 0].tolist()) == [0, 1, 2]
