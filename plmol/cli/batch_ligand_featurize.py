#!/usr/bin/env python
"""
Batch feature extraction for ligand files.

Supports multiple file formats: SDF, MOL2, MOL, PDB.
If multiple formats exist for the same ligand, tries each until one loads successfully.

Usage:
    python scripts/batch_ligand_featurize.py --input_dir /data/ligands --output_dir /data/ligand-features
    python scripts/batch_ligand_featurize.py --input_dir /data/ligands --output_dir /data/ligand-features --num_workers 4
    python scripts/batch_ligand_featurize.py --input_dir /data/ligands --output_dir /data/ligand-features --resume
    python scripts/batch_ligand_featurize.py --input_dir /data/ligands --output_dir /data/ligand-features --graph_only
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from plmol.constants import IO_SUPPORTED_LIGAND_EXTENSIONS

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

from plmol import MoleculeFeaturizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Supported file extensions in priority order
SUPPORTED_EXTENSIONS = IO_SUPPORTED_LIGAND_EXTENSIONS


def load_ligand(file_path: Path) -> Optional[Chem.Mol]:
    """
    Load ligand from file based on extension.

    Args:
        file_path: Path to ligand file

    Returns:
        RDKit Mol object or None if loading failed
    """
    ext = file_path.suffix.lower()
    mol = None

    try:
        if ext == '.sdf':
            suppl = Chem.SDMolSupplier(str(file_path), removeHs=False)
            if suppl and len(suppl) > 0:
                mol = suppl[0]
        elif ext == '.mol2':
            mol = Chem.MolFromMol2File(str(file_path), removeHs=False)
        elif ext == '.mol':
            mol = Chem.MolFromMolFile(str(file_path), removeHs=False)
        elif ext == '.pdb':
            mol = Chem.MolFromPDBFile(str(file_path), removeHs=False)
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(f"Failed to load ligand file {file_path}: {e}")

    return mol


def find_ligand_files(input_dir: str) -> Dict[str, List[Path]]:
    """
    Find all ligand files and group by ligand ID.

    Returns:
        Dictionary mapping ligand_id -> list of file paths (sorted by extension priority)
    """
    input_path = Path(input_dir)
    ligand_files = defaultdict(list)

    for ext in SUPPORTED_EXTENSIONS:
        for file_path in input_path.rglob(f"*{ext}"):
            # Extract ligand ID (filename without extension)
            ligand_id = file_path.stem
            # Remove common suffixes like _ligand
            for suffix in ['_ligand', '_lig', '_mol']:
                if ligand_id.endswith(suffix):
                    ligand_id = ligand_id[:-len(suffix)]
                    break
            ligand_files[ligand_id].append(file_path)

    # Sort each ligand's files by extension priority
    for ligand_id in ligand_files:
        ligand_files[ligand_id].sort(
            key=lambda p: SUPPORTED_EXTENSIONS.index(p.suffix.lower())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS else 999
        )

    return dict(ligand_files)


def get_output_path(ligand_id: str, input_dir: str, output_dir: str,
                    file_path: Optional[Path] = None) -> Path:
    """Get output path, preserving directory structure if possible."""
    if file_path:
        try:
            rel_path = file_path.relative_to(input_dir)
            output_path = Path(output_dir) / rel_path.parent / f"{ligand_id}.pt"
        except ValueError:
            output_path = Path(output_dir) / f"{ligand_id}.pt"
    else:
        output_path = Path(output_dir) / f"{ligand_id}.pt"
    return output_path


def process_single_ligand(
    ligand_id: str,
    file_paths: List[Path],
    input_dir: str,
    output_dir: str,
    add_hydrogens: bool = False,
    canonicalize: bool = True,
    graph_only: bool = False,
) -> Tuple[str, bool, str]:
    """
    Process a single ligand, trying multiple file formats if needed.

    Args:
        ligand_id: Unique identifier for the ligand
        file_paths: List of file paths to try loading from
        input_dir: Input directory root
        output_dir: Output directory root
        add_hydrogens: Whether to add hydrogens (default: False, heavy atoms only)
        canonicalize: Whether to canonicalize atom order (default: True)
        graph_only: If True, only extract graph features (no descriptors/fingerprints)

    Returns:
        Tuple of (ligand_id, success, message)
    """
    try:
        output_path = get_output_path(ligand_id, input_dir, output_dir, file_paths[0])

        # Skip if already processed
        if output_path.exists():
            return (ligand_id, True, "skipped (exists)")

        # Try loading from each file format
        mol = None
        loaded_from = None
        for file_path in file_paths:
            mol = load_ligand(file_path)
            if mol is not None:
                loaded_from = file_path
                break

        if mol is None:
            formats = [p.suffix for p in file_paths]
            return (ligand_id, False, f"failed to load from {formats}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize featurizer with ligand
        featurizer = MoleculeFeaturizer(
            mol,
            add_hs=add_hydrogens,
            canonicalize=canonicalize
        )

        # Extract graph features
        node, edge, adj = featurizer.get_graph()

        # Build save dictionary
        save_dict = {
            # Graph features
            'node_feats': node['node_feats'],               # [N_atoms, ~42]
            'coords': node['coords'],                       # [N_atoms, 3]
            'edge_feats': edge['edge_feats'],               # [N_edges, ~10]
            'edge_index': edge['edges'],                    # [2, N_edges]
            'adjacency': adj,                               # [N_atoms, N_atoms, ~10]

            # Metadata
            'num_atoms': featurizer.num_atoms,
            'num_bonds': featurizer.num_bonds,
            'num_rings': featurizer.num_rings,
            'has_3d': featurizer.has_3d,
            'smiles': featurizer.input_smiles,
            'ligand_id': ligand_id,
            'source_path': str(loaded_from),
            'source_format': loaded_from.suffix,
            'graph_only': graph_only,
            'canonicalized': canonicalize,
        }

        # Add molecular descriptors and fingerprints if not graph_only
        if not graph_only:
            features = featurizer.get_features()
            save_dict['descriptors'] = features['descriptors']            # [40]
            save_dict.update({k: v for k, v in features.items() if k != 'descriptors'})

        # Save
        torch.save(save_dict, output_path)

        mode_str = "graph" if graph_only else "all"
        return (ligand_id, True, f"ok ({featurizer.num_atoms} atoms, {loaded_from.suffix}, {mode_str})")

    except Exception as e:
        return (ligand_id, False, str(e)[:100])


def process_wrapper(args: Tuple) -> Tuple[str, bool, str]:
    """Wrapper for multiprocessing."""
    ligand_id, file_paths, input_dir, output_dir, add_hydrogens, canonicalize, graph_only = args
    return process_single_ligand(
        ligand_id, file_paths, input_dir, output_dir,
        add_hydrogens, canonicalize, graph_only
    )


def main():
    parser = argparse.ArgumentParser(
        description='Batch feature extraction for ligand files'
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Input directory containing ligand files (SDF, MOL2, MOL, PDB)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for feature files'
    )
    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip already processed files'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of ligands to process'
    )
    parser.add_argument(
        '--add_hydrogens', action='store_true',
        help='Add explicit hydrogens to ligands (default: heavy atoms only)'
    )
    parser.add_argument(
        '--no_canonicalize', action='store_true',
        help='Do not canonicalize atom order (default: canonicalize for ML consistency)'
    )
    parser.add_argument(
        '--graph_only', action='store_true',
        help='Extract only graph features (no descriptors/fingerprints)'
    )
    args = parser.parse_args()

    add_hydrogens = args.add_hydrogens
    canonicalize = not args.no_canonicalize
    graph_only = args.graph_only

    # Log options
    logger.info(f"Options: hydrogen={add_hydrogens}, canonicalize={canonicalize}, graph_only={graph_only}")

    # Find all ligand files
    logger.info(f"Scanning {args.input_dir} for ligand files...")
    ligand_files = find_ligand_files(args.input_dir)
    logger.info(f"Found {len(ligand_files)} unique ligands")

    # Count formats
    format_counts = defaultdict(int)
    for files in ligand_files.values():
        for f in files:
            format_counts[f.suffix.lower()] += 1
    logger.info(f"Format distribution: {dict(format_counts)}")

    # Convert to list of tuples for processing
    ligand_list = list(ligand_files.items())

    if args.limit:
        ligand_list = ligand_list[:args.limit]
        logger.info(f"Limited to {len(ligand_list)} ligands")

    # Filter already processed if resume
    if args.resume:
        original_count = len(ligand_list)
        ligand_list = [
            (lid, files) for lid, files in ligand_list
            if not get_output_path(lid, args.input_dir, args.output_dir, files[0]).exists()
        ]
        skipped = original_count - len(ligand_list)
        logger.info(f"Resuming: {skipped} already processed, {len(ligand_list)} remaining")

    if not ligand_list:
        logger.info("No ligands to process")
        return

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Statistics
    success_count = 0
    fail_count = 0
    failed_ligands = []

    start_time = time.time()

    if args.num_workers == 1:
        # Single process mode
        with tqdm(ligand_list, desc="Processing", unit="ligand") as pbar:
            for ligand_id, file_paths in pbar:
                lid, success, msg = process_single_ligand(
                    ligand_id, file_paths, args.input_dir, args.output_dir,
                    add_hydrogens, canonicalize, graph_only
                )

                if success:
                    success_count += 1
                    pbar.set_postfix_str(f"{lid}: {msg}")
                else:
                    fail_count += 1
                    failed_ligands.append((lid, msg))
                    pbar.set_postfix_str(f"{lid}: FAILED")
    else:
        # Multi-process mode
        logger.info(f"Using {args.num_workers} workers")

        tasks = [
            (lid, files, args.input_dir, args.output_dir, add_hydrogens, canonicalize, graph_only)
            for lid, files in ligand_list
        ]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_wrapper, task): task[0] for task in tasks}

            with tqdm(total=len(futures), desc="Processing", unit="ligand") as pbar:
                for future in as_completed(futures):
                    lid, success, msg = future.result()

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        failed_ligands.append((lid, msg))

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
        logger.info(f"Speed: {(success_count + fail_count) / elapsed:.2f} ligands/sec")

    if failed_ligands:
        logger.info("\nFailed ligands:")
        for lid, error in failed_ligands[:20]:
            logger.info(f"  {lid}: {error[:80]}")
        if len(failed_ligands) > 20:
            logger.info(f"  ... and {len(failed_ligands) - 20} more")

        # Save failed list
        fail_log = Path(args.output_dir) / "failed_ligands.txt"
        with open(fail_log, 'w') as f:
            for lid, error in failed_ligands:
                f.write(f"{lid}\t{error}\n")
        logger.info(f"\nFailed ligands saved to: {fail_log}")


if __name__ == '__main__':
    main()
