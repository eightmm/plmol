"""Tests for ligand batch featurization CLI helpers."""

from pathlib import Path

import torch

from plmol.cli.batch_ligand_featurize import process_single_ligand


def test_process_single_ligand_resume_skips_existing(tmp_path, example_sdf):
    output_path = tmp_path / "10gs.pt"
    torch.save({"sentinel": True}, output_path)

    result = process_single_ligand(
        "10gs",
        [Path(example_sdf)],
        "examples",
        str(tmp_path),
        resume=True,
    )

    assert result == ("10gs", True, "skipped (exists)")
    assert torch.load(output_path, weights_only=False) == {"sentinel": True}


def test_process_single_ligand_writes_descriptor_names(tmp_path, example_sdf):
    result = process_single_ligand(
        "10gs",
        [Path(example_sdf)],
        "examples",
        str(tmp_path),
        resume=False,
    )

    assert result[1] is True
    saved = torch.load(tmp_path / "10gs.pt", weights_only=False)
    assert saved["descriptors"].shape[0] == 62
    assert len(saved["descriptor_names"]) == 62
