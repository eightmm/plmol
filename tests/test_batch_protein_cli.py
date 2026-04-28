"""Tests for protein batch featurization CLI helpers."""

from pathlib import Path
from types import SimpleNamespace

import torch

from plmol.cli.batch_protein_featurize import process_single_file_shared_featurizer


class FakeHierarchicalFeaturizer:
    def featurize(self, _pdb_path):
        return SimpleNamespace(
            atom_tokens=torch.tensor([1, 2], dtype=torch.long),
            atom_coords=torch.zeros((2, 3), dtype=torch.float32),
            atom_sasa=torch.zeros(2, dtype=torch.float32),
            atom_elements=torch.tensor([0, 1], dtype=torch.long),
            atom_residue_types=torch.tensor([0, 0], dtype=torch.long),
            atom_names=["N", "CA"],
            residue_features=torch.zeros((1, 76), dtype=torch.float32),
            residue_ca_coords=torch.zeros((1, 3), dtype=torch.float32),
            residue_sc_coords=torch.zeros((1, 3), dtype=torch.float32),
            residue_names=["ALA"],
            residue_ids=[1],
            esmc_embeddings=torch.zeros((1, 1152), dtype=torch.float32),
            esmc_bos=torch.zeros(1152, dtype=torch.float32),
            esmc_eos=torch.zeros(1152, dtype=torch.float32),
            esm3_embeddings=torch.zeros((1, 1536), dtype=torch.float32),
            esm3_bos=torch.zeros(1536, dtype=torch.float32),
            esm3_eos=torch.zeros(1536, dtype=torch.float32),
            residue_vector_features=torch.zeros((1, 31, 3), dtype=torch.float32),
            atom_to_residue=torch.tensor([0, 0], dtype=torch.long),
            residue_atom_indices=[[0, 1]],
            residue_atom_mask=torch.ones((1, 2), dtype=torch.bool),
            num_atoms_per_residue=torch.tensor([2], dtype=torch.long),
            num_atoms=2,
            num_residues=1,
        )


def _copy_as_protein_file(src: str, dst_dir: Path) -> Path:
    dst = dst_dir / "10gs_protein.pdb"
    dst.write_text(Path(src).read_text())
    return dst


def test_shared_protein_resume_skips_existing(tmp_path, example_pdb):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    pdb_path = _copy_as_protein_file(example_pdb, input_dir)
    output_path = output_dir / "10gs_protein.pt"
    torch.save({"sentinel": True}, output_path)

    result = process_single_file_shared_featurizer(
        pdb_path,
        str(input_dir),
        str(output_dir),
        FakeHierarchicalFeaturizer(),
        resume=True,
    )

    assert result == ("10gs", True, "skipped (exists)")
    assert torch.load(output_path, weights_only=False) == {"sentinel": True}


def test_shared_protein_overwrites_existing_without_resume(tmp_path, example_pdb):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    pdb_path = _copy_as_protein_file(example_pdb, input_dir)
    output_path = output_dir / "10gs_protein.pt"
    torch.save({"sentinel": True}, output_path)

    result = process_single_file_shared_featurizer(
        pdb_path,
        str(input_dir),
        str(output_dir),
        FakeHierarchicalFeaturizer(),
        resume=False,
    )

    assert result == ("10gs", True, "1 res")
    saved = torch.load(output_path, weights_only=False)
    assert saved["num_residues"] == 1
    assert saved["pdb_id"] == "10gs"
