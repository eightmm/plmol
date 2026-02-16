#!/usr/bin/env python
"""Compatibility wrapper for plmol.cli.batch_ligand_featurize."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plmol.cli.batch_ligand_featurize import main


if __name__ == '__main__':
    main()
