"""Learning curve on the football task (see bench/data/football.py for the rows and features).

    python -m bench.run_football [--quick]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

from .common.curve import run_curve
from .data.football import build

OUT = Path(__file__).parent / "results" / "football.json"
SIZES = (25, 50, 100, 200, 10**6)


def main(quick=False):
    src, pool, test, meta = build()
    sizes = SIZES[:2] if quick else SIZES
    seeds = list(range(3 if quick else 20))
    run_curve("football", src, pool, test, sizes, seeds, OUT.with_suffix(".quick.json") if quick else OUT, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
