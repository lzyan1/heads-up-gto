#!/usr/bin/env python3
"""Run from any working directory: python /path/to/scripts/precompute_rivers.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heads_up_gto.river.dataset import main

if __name__ == '__main__':
    main()
