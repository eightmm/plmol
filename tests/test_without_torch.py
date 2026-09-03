"""plmol must import and featurize with torch absent.

torch is an optional dependency: the featurizers return numpy, and torch is
needed only to convert those arrays or to run a protein language model. This
runs a real import in a subprocess with torch blocked at the meta path, because
unloading it inside the test process would break every other test.
"""

import subprocess
import sys
import textwrap

import pytest

BLOCKER = '''
import sys
from importlib.abc import MetaPathFinder


class Blocker(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {blocked!r}:
            raise ImportError("blocked: " + name)


for module in [m for m in sys.modules if m.split(".")[0] in {blocked!r}]:
    del sys.modules[module]
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, {root!r})
'''


def run_without(blocked, body, root):
    """Run *body* in a fresh interpreter with *blocked* packages unimportable."""
    script = BLOCKER.format(blocked=set(blocked), root=root) + textwrap.dedent(body)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


@pytest.fixture(scope="module")
def root():
    import plmol
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(plmol.__file__)))


class TestImportWithoutTorch:
    def test_the_package_imports(self, root):
        assert run_without(["torch"], """
            import plmol
            print(plmol.__version__)
        """, root)

    def test_torch_is_not_pulled_in_by_featurizing(self, root):
        """Not just importable -- nothing on the feature path reaches for it."""
        assert run_without(["torch"], """
            import sys
            from plmol import Protein
            Protein.from_pdb(sys.argv[0] if False else PDB).featurize(mode="graph")
            print("torch" not in sys.modules)
        """.replace("PDB", repr(f"{root}/examples/10gs_protein.pdb")), root) == "True"


class TestFeaturizingWithoutTorch:
    @pytest.mark.parametrize(
        "mode", ["sequence", "graph", "atom_graph", "backbone", "surface", "voxel"]
    )
    def test_every_protein_mode(self, root, mode):
        out = run_without(["torch"], f"""
            from plmol import Protein
            result = Protein.from_pdb({root + '/examples/10gs_protein.pdb'!r}).featurize(mode={mode!r})
            print(sorted(result))
        """, root)
        assert mode in out

    @pytest.mark.parametrize(
        "mode", ["graph", "bond_graph", "fragment_graph", "fingerprint", "descriptor"]
    )
    def test_every_ligand_mode(self, root, mode):
        out = run_without(["torch"], f"""
            from plmol import Ligand
            result = Ligand.from_sdf({root + '/examples/10gs_ligand.sdf'!r}).featurize(mode={mode!r})
            print(sorted(result))
        """, root)
        assert mode in out

    def test_a_complex(self, root):
        out = run_without(["torch"], f"""
            from plmol import Ligand, Protein
            from plmol.complex import MolecularComplex
            complex_ = MolecularComplex(molecules={{
                "protein": Protein.from_pdb({root + '/examples/10gs_protein.pdb'!r}),
                "ligand": Ligand.from_sdf({root + '/examples/10gs_ligand.sdf'!r}),
            }})
            print(sorted(complex_.featurize(requests="all")))
        """, root)
        assert "interaction" in out and "protein" in out

    def test_with_scipy_and_freesasa_gone_as_well(self, root):
        """The three optional dependencies are independent of each other."""
        out = run_without(["torch", "scipy", "freesasa"], f"""
            from plmol import Protein, resolve_sasa_backend, resolve_spatial_backend
            surface = Protein.from_pdb({root + '/examples/10gs_protein.pdb'!r}).featurize(mode="surface")
            print(resolve_sasa_backend(), resolve_spatial_backend(),
                  surface["surface"]["points"].shape[0] > 0)
        """, root)
        assert out == "native native True"


class TestToTorch:
    def test_the_error_names_the_install(self, root):
        out = run_without(["torch"], """
            import numpy as np
            from plmol import DependencyError, to_torch
            try:
                to_torch({"x": np.zeros(3)})
            except DependencyError as exc:
                print("pip install torch" in str(exc))
        """, root)
        assert out == "True"

    def test_round_trip_when_torch_is_present(self, example_pdb):
        torch = pytest.importorskip("torch")
        import numpy as np

        from plmol import Protein, to_numpy, to_torch

        graph = Protein.from_pdb(example_pdb).featurize(mode="atom_graph")["atom_graph"]
        tensors = to_torch(graph)
        assert isinstance(tensors["coords"], torch.Tensor)
        assert tensors["coords"].dtype == torch.float32
        assert np.array_equal(to_numpy(tensors)["coords"], graph["coords"])
