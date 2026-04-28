#!/usr/bin/env python
"""
Batch feature extraction for protein PDB files.

Usage:
    python scripts/batch_protein_featurize.py --input_dir /data/proteins --output_dir /data/features
    python scripts/batch_protein_featurize.py --input_dir /data/proteins --output_dir /data/features --num_workers 4
    python scripts/batch_protein_featurize.py --input_dir /data/proteins --output_dir /data/features --standardize
    python scripts/batch_protein_featurize.py --input_dir /data/proteins --output_dir /data/features --standardize --ptm_handling preserve
"""

import argparse
import logging
import os
import time
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from tqdm import tqdm

from plmol.protein.hierarchical_featurizer import HierarchicalFeaturizer
from plmol import PDBStandardizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


DEFAULT_PROTEIN_PATTERN = "*protein.pdb"


def resolve_device(device: str) -> str:
    """Resolve the ESM device, using CUDA when available for the auto setting."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def find_protein_files(input_dir: str, pattern: str = DEFAULT_PROTEIN_PATTERN) -> List[Path]:
    """Find protein PDB files recursively."""
    input_path = Path(input_dir)
    files = sorted(input_path.rglob(pattern))
    return files


def get_output_path(pdb_path: Path, input_dir: str, output_dir: str) -> Path:
    """Get output path preserving directory structure."""
    rel_path = pdb_path.relative_to(input_dir)
    output_path = Path(output_dir) / rel_path.with_suffix('.pt')
    return output_path


def standardize_pdb_to_tmp(
    pdb_path: Path,
    ptm_handling: str = 'unk',
    remove_hydrogens: bool = True
) -> str:
    """
    Standardize PDB file and save to /tmp.

    Returns:
        Path to standardized PDB file in /tmp
    """
    # Create unique temp file path
    tmp_filename = f"plmol_{uuid.uuid4().hex}_{pdb_path.stem}.pdb"
    tmp_path = os.path.join(tempfile.gettempdir(), tmp_filename)

    # Standardize
    standardizer = PDBStandardizer(
        remove_hydrogens=remove_hydrogens,
        ptm_handling=ptm_handling
    )
    standardizer.standardize(str(pdb_path), tmp_path)

    return tmp_path


def _build_save_dict(data, pdb_id: str, pdb_path: Path, standardize: bool, ptm_handling: str) -> dict:
    """Build the output dict from featurized data."""
    return {
        # Atom-level (integer indices for nn.Embedding lookup)
        'atom_tokens': data.atom_tokens,           # [N_atom] - indices 0-186 (187 classes)
        'atom_coords': data.atom_coords,           # [N_atom, 3]
        'atom_sasa': data.atom_sasa,               # [N_atom]
        'atom_elements': data.atom_elements,       # [N_atom] - indices 0-7 (8 classes)
        'atom_residue_types': data.atom_residue_types,  # [N_atom] - indices 0-21 (22 classes)
        'atom_names': data.atom_names,

        # Residue-level
        'residue_features': data.residue_features,  # [N_res, 76]
        'residue_ca_coords': data.residue_ca_coords,  # [N_res, 3]
        'residue_sc_coords': data.residue_sc_coords,  # [N_res, 3]
        'residue_names': data.residue_names,
        'residue_ids': data.residue_ids,

        # ESM embeddings (6 tensors)
        'esmc_embeddings': data.esmc_embeddings,   # [N_res, 1152]
        'esmc_bos': data.esmc_bos,                 # [1152]
        'esmc_eos': data.esmc_eos,                 # [1152]
        'esm3_embeddings': data.esm3_embeddings,   # [N_res, 1536]
        'esm3_bos': data.esm3_bos,                 # [1536]
        'esm3_eos': data.esm3_eos,                 # [1536]

        # Residue vector features
        'residue_vector_features': data.residue_vector_features,  # [N_res, 31, 3]

        # Mapping
        'atom_to_residue': data.atom_to_residue,   # [N_atom]
        'residue_atom_indices': data.residue_atom_indices,
        'residue_atom_mask': data.residue_atom_mask,
        'num_atoms_per_residue': data.num_atoms_per_residue,

        # Metadata
        'num_atoms': data.num_atoms,
        'num_residues': data.num_residues,
        'pdb_id': pdb_id,
        'source_path': str(pdb_path),
        'standardized': standardize,
        'ptm_handling': ptm_handling if standardize else None,
    }


def process_single_file(args: Tuple) -> Tuple[str, bool, str]:
    """
    Process a single PDB file (for multi-process mode).

    Returns:
        Tuple of (pdb_id, success, message)
    """
    if len(args) == 6:
        pdb_path, input_dir, output_dir, standardize, ptm_handling, resume = args
        esmc_model = "esmc_600m"
        esm3_model = "esm3-open"
        device = resolve_device("auto")
    else:
        (
            pdb_path,
            input_dir,
            output_dir,
            standardize,
            ptm_handling,
            resume,
            esmc_model,
            esm3_model,
            device,
        ) = args
    pdb_id = pdb_path.stem.replace('_protein', '')
    tmp_pdb_path = None

    try:
        output_path = get_output_path(pdb_path, input_dir, output_dir)

        # Skip if already processed and resume mode is enabled
        if resume and output_path.exists():
            return (pdb_id, True, "skipped (exists)")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Standardize if requested
        if standardize:
            tmp_pdb_path = standardize_pdb_to_tmp(pdb_path, ptm_handling)
            pdb_to_process = tmp_pdb_path
        else:
            pdb_to_process = str(pdb_path)

        # Initialize featurizer (per-process)
        featurizer = HierarchicalFeaturizer(
            esmc_model=esmc_model,
            esm3_model=esm3_model,
            esm_device=device,
        )

        # Extract features
        data = featurizer.featurize(pdb_to_process)

        # Save
        torch.save(_build_save_dict(data, pdb_id, pdb_path, standardize, ptm_handling), output_path)

        return (pdb_id, True, f"ok ({data.num_residues} residues)")

    except Exception as e:
        return (pdb_id, False, str(e))

    finally:
        # Clean up temp file
        if tmp_pdb_path and os.path.exists(tmp_pdb_path):
            os.remove(tmp_pdb_path)


def process_single_file_shared_featurizer(
    pdb_path: Path,
    input_dir: str,
    output_dir: str,
    featurizer: HierarchicalFeaturizer,
    standardize: bool = False,
    ptm_handling: str = 'unk',
    resume: bool = False,
) -> Tuple[str, bool, str]:
    """
    Process a single PDB file with shared featurizer (for single-process mode).
    """
    pdb_id = pdb_path.stem.replace('_protein', '')
    tmp_pdb_path = None

    try:
        output_path = get_output_path(pdb_path, input_dir, output_dir)

        # Skip if already processed and resume mode is enabled
        if resume and output_path.exists():
            return (pdb_id, True, "skipped (exists)")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Standardize if requested
        if standardize:
            tmp_pdb_path = standardize_pdb_to_tmp(pdb_path, ptm_handling)
            pdb_to_process = tmp_pdb_path
        else:
            pdb_to_process = str(pdb_path)

        # Extract features
        data = featurizer.featurize(pdb_to_process)

        # Save
        torch.save(_build_save_dict(data, pdb_id, pdb_path, standardize, ptm_handling), output_path)

        return (pdb_id, True, f"{data.num_residues} res")

    except Exception as e:
        return (pdb_id, False, str(e))

    finally:
        # Clean up temp file
        if tmp_pdb_path and os.path.exists(tmp_pdb_path):
            os.remove(tmp_pdb_path)


def main():
    parser = argparse.ArgumentParser(description='Batch feature extraction for protein PDB files')
    parser.add_argument('--input-dir', '--input_dir', dest='input_dir', type=str, required=True, help='Input directory containing PDB files')
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir', type=str, required=True, help='Output directory for feature files')
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int, default=1, help='Number of parallel workers (default: 1)')
    parser.add_argument('--resume', action='store_true', help='Skip already processed files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    parser.add_argument('--pattern', type=str, default=DEFAULT_PROTEIN_PATTERN, help='Recursive glob pattern for input PDB files (default: *protein.pdb)')
    parser.add_argument('--all-pdb', action='store_true', help='Scan all *.pdb files instead of only *protein.pdb')
    parser.add_argument('--esmc-model', '--esmc_model', dest='esmc_model', type=str, default='esmc_600m', help='ESMC model name')
    parser.add_argument('--esm3-model', '--esm3_model', dest='esm3_model', type=str, default='esm3-open', help='ESM3 model name')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device for ESM models (default: auto)')
    parser.add_argument('--standardize', action='store_true', help='Standardize PDB files before featurization')
    parser.add_argument('--ptm-handling', '--ptm_handling', dest='ptm_handling', type=str, default='unk',
                        choices=['base_aa', 'unk', 'preserve', 'remove'],
                        help='PTM handling mode (default: unk)')
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        parser.error(f"input directory does not exist or is not a directory: {args.input_dir}")
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.all_pdb and args.pattern != DEFAULT_PROTEIN_PATTERN:
        parser.error("--all-pdb cannot be combined with --pattern")

    pattern = "*.pdb" if args.all_pdb else args.pattern
    device = resolve_device(args.device)

    # Find all protein files
    logger.info(f"Scanning {args.input_dir} for protein files matching {pattern!r}...")
    pdb_files = find_protein_files(args.input_dir, pattern)
    logger.info(f"Found {len(pdb_files)} protein files")
    logger.info(f"ESM device: {device} (requested: {args.device})")

    if args.standardize:
        logger.info(f"Standardization enabled (ptm_handling: {args.ptm_handling})")

    if args.limit:
        pdb_files = pdb_files[:args.limit]
        logger.info(f"Limited to {len(pdb_files)} files")

    # Filter already processed if resume
    if args.resume:
        original_count = len(pdb_files)
        pdb_files = [
            f for f in pdb_files
            if not get_output_path(f, args.input_dir, args.output_dir).exists()
        ]
        skipped = original_count - len(pdb_files)
        logger.info(f"Resuming: {skipped} already processed, {len(pdb_files)} remaining")

    if not pdb_files:
        logger.info("No files to process")
        return

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Statistics
    success_count = 0
    fail_count = 0
    failed_files = []

    start_time = time.time()

    if args.num_workers == 1:
        # Single process mode - share featurizer
        logger.info("Initializing featurizer...")
        featurizer = HierarchicalFeaturizer(
            esmc_model=args.esmc_model,
            esm3_model=args.esm3_model,
            esm_device=device,
        )
        logger.info("Featurizer ready")

        with tqdm(pdb_files, desc="Processing", unit="file") as pbar:
            for pdb_path in pbar:
                pdb_id, success, msg = process_single_file_shared_featurizer(
                    pdb_path, args.input_dir, args.output_dir, featurizer,
                    args.standardize, args.ptm_handling, args.resume
                )

                if success:
                    success_count += 1
                    pbar.set_postfix_str(f"{pdb_id}: {msg}")
                else:
                    fail_count += 1
                    failed_files.append((pdb_id, msg))
                    pbar.set_postfix_str(f"{pdb_id}: FAILED")
    else:
        # Multi-process mode
        logger.info(f"Using {args.num_workers} workers")

        tasks = [
            (
                f,
                args.input_dir,
                args.output_dir,
                args.standardize,
                args.ptm_handling,
                args.resume,
                args.esmc_model,
                args.esm3_model,
                device,
            )
            for f in pdb_files
        ]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_file, task): task[0] for task in tasks}

            with tqdm(total=len(futures), desc="Processing", unit="file") as pbar:
                for future in as_completed(futures):
                    pdb_id, success, msg = future.result()

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        failed_files.append((pdb_id, msg))

                    pbar.update(1)
                    pbar.set_postfix_str(f"ok={success_count}, fail={fail_count}")

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed: {success_count + fail_count}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if elapsed > 0:
        logger.info(f"Speed: {(success_count + fail_count) / elapsed:.2f} files/sec")

    if failed_files:
        logger.info("\nFailed files:")
        for pdb_id, error in failed_files[:20]:
            logger.info(f"  {pdb_id}: {error[:80]}")
        if len(failed_files) > 20:
            logger.info(f"  ... and {len(failed_files) - 20} more")

        # Save failed list
        fail_log = Path(args.output_dir) / "failed_files.txt"
        with open(fail_log, 'w') as f:
            for pdb_id, error in failed_files:
                f.write(f"{pdb_id}\t{error}\n")
        logger.info(f"\nFailed files saved to: {fail_log}")


if __name__ == '__main__':
    main()
