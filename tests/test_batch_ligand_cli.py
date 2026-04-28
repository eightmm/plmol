"""Tests for ligand batch featurization CLI helpers."""

from pathlib import Path

import torch

from plmol.cli.batch_ligand_featurize import find_ligand_files, normalize_extensions, process_single_ligand


def test_normalize_extensions_accepts_common_forms():
    assert normalize_extensions(["sdf", ".MOL", " sdf "]) == [".sdf", ".mol"]


def test_find_ligand_files_respects_extension_filter(tmp_path):
    (tmp_path / "a.SDF").write_text("")
    (tmp_path / "a.mol2").write_text("")
    (tmp_path / "b.pdb").write_text("")

    found = find_ligand_files(str(tmp_path), extensions=["sdf"])

    assert set(found) == {"a"}
    assert [path.suffix for path in found["a"]] == [".SDF"]


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
