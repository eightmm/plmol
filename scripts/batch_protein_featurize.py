#!/usr/bin/env python
"""
Batch feature extraction for protein PDB files.

Usage:
    python scripts/batch_featurize.py --input_dir /mnt/data/PLI/P-L --output_dir /mnt/data/PLI/P-L-features
    python scripts/batch_featurize.py --input_dir /mnt/data/PLI/P-L --output_dir /mnt/data/PLI/P-L-features --num_workers 4
    python scripts/batch_featurize.py --input_dir /mnt/data/PLI/P-L --output_dir /mnt/data/PLI/P-L-features --resume
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plfeature.protein_featurizer import HierarchicalFeaturizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_protein_files(input_dir: str) -> List[Path]:
    """Find all *protein.pdb files recursively."""
    input_path = Path(input_dir)
    files = sorted(input_path.rglob("*protein.pdb"))
    return files


def get_output_path(pdb_path: Path, input_dir: str, output_dir: str) -> Path:
    """Get output path preserving directory structure."""
    rel_path = pdb_path.relative_to(input_dir)
    output_path = Path(output_dir) / rel_path.with_suffix('.pt')
    return output_path


def process_single_file(args: Tuple[Path, str, str]) -> Tuple[str, bool, str]:
    """
    Process a single PDB file.

    Returns:
        Tuple of (pdb_id, success, message)
    """
    pdb_path, input_dir, output_dir = args
    pdb_id = pdb_path.stem.replace('_protein', '')

    try:
        output_path = get_output_path(pdb_path, input_dir, output_dir)

        # Skip if already processed
        if output_path.exists():
            return (pdb_id, True, "skipped (exists)")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize featurizer (per-process)
        featurizer = HierarchicalFeaturizer()

        # Extract features
        data = featurizer.featurize(str(pdb_path))

        # Convert to dict for saving
        save_dict = {
            # Atom-level (one-hot encoded)
            'atom_tokens': data.atom_tokens,           # [N_atom, 187]
            'atom_coords': data.atom_coords,           # [N_atom, 3]
            'atom_sasa': data.atom_sasa,               # [N_atom]
            'atom_elements': data.atom_elements,       # [N_atom, 8]
            'atom_residue_types': data.atom_residue_types,  # [N_atom, 22]
            'atom_names': data.atom_names,

            # Residue-level
            'residue_features': data.residue_features,  # [N_res, 76]
            'residue_ca_coords': data.residue_ca_coords,
            'residue_sc_coords': data.residue_sc_coords,
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
            'atom_to_residue': data.atom_to_residue,
            'residue_atom_indices': data.residue_atom_indices,
            'residue_atom_mask': data.residue_atom_mask,
            'num_atoms_per_residue': data.num_atoms_per_residue,

            # Metadata
            'num_atoms': data.num_atoms,
            'num_residues': data.num_residues,
            'pdb_id': pdb_id,
            'source_path': str(pdb_path),
        }

        # Save
        torch.save(save_dict, output_path)

        return (pdb_id, True, f"ok ({data.num_residues} residues)")

    except Exception as e:
        return (pdb_id, False, str(e))


def process_single_file_shared_featurizer(
    pdb_path: Path,
    input_dir: str,
    output_dir: str,
    featurizer: HierarchicalFeaturizer
) -> Tuple[str, bool, str]:
    """
    Process a single PDB file with shared featurizer (for single-process mode).
    """
    pdb_id = pdb_path.stem.replace('_protein', '')

    try:
        output_path = get_output_path(pdb_path, input_dir, output_dir)

        # Skip if already processed
        if output_path.exists():
            return (pdb_id, True, "skipped")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract features
        data = featurizer.featurize(str(pdb_path))

        # Convert to dict for saving
        save_dict = {
            # Atom-level (one-hot encoded)
            'atom_tokens': data.atom_tokens,           # [N_atom, 187]
            'atom_coords': data.atom_coords,           # [N_atom, 3]
            'atom_sasa': data.atom_sasa,               # [N_atom]
            'atom_elements': data.atom_elements,       # [N_atom, 8]
            'atom_residue_types': data.atom_residue_types,  # [N_atom, 22]
            'atom_names': data.atom_names,

            # Residue-level
            'residue_features': data.residue_features,  # [N_res, 76]
            'residue_ca_coords': data.residue_ca_coords,
            'residue_sc_coords': data.residue_sc_coords,
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
            'atom_to_residue': data.atom_to_residue,
            'residue_atom_indices': data.residue_atom_indices,
            'residue_atom_mask': data.residue_atom_mask,
            'num_atoms_per_residue': data.num_atoms_per_residue,

            # Metadata
            'num_atoms': data.num_atoms,
            'num_residues': data.num_residues,
            'pdb_id': pdb_id,
            'source_path': str(pdb_path),
        }

        # Save
        torch.save(save_dict, output_path)

        return (pdb_id, True, f"{data.num_residues} res")

    except Exception as e:
        return (pdb_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='Batch feature extraction for protein PDB files')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing PDB files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for feature files')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers (default: 1)')
    parser.add_argument('--resume', action='store_true', help='Skip already processed files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    parser.add_argument('--esmc_model', type=str, default='esmc_600m', help='ESMC model name')
    parser.add_argument('--esm3_model', type=str, default='esm3-open', help='ESM3 model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device for ESM models')
    args = parser.parse_args()

    # Find all protein files
    logger.info(f"Scanning {args.input_dir} for protein.pdb files...")
    pdb_files = find_protein_files(args.input_dir)
    logger.info(f"Found {len(pdb_files)} protein files")

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
            esm_device=args.device,
        )
        logger.info("Featurizer ready")

        with tqdm(pdb_files, desc="Processing", unit="file") as pbar:
            for pdb_path in pbar:
                pdb_id, success, msg = process_single_file_shared_featurizer(
                    pdb_path, args.input_dir, args.output_dir, featurizer
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

        tasks = [(f, args.input_dir, args.output_dir) for f in pdb_files]

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
