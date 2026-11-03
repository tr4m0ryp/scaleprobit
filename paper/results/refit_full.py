"""Refit the scaleprobit arms at the full target season and store per-row test predictions.

The benchmark JSON files hold metrics only (per-row predictions were not stored). At the full target
size every subsample is the same set of games, so one deterministic refit reproduces the bench numbers
exactly and gives the per-row outputs the reliability diagram and the per-game paired SE need.

    PYTHONPATH=<repo> python paper/results/refit_full.py [football|hoops ...]

Writes results/preds_<dataset>_full.npz (y, game ids, one probability array per arm) and prints the
log-loss of each arm next to the value stored in bench/results/<dataset>.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from bench.common.arms_scale import SourceFits, scale_arms
from scaleprobit.evaluate import log_loss_rows

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = ("closed_form", "target_only_cv", "source_only", "ours_centred_cv", "pooled_indicator")


def run(name: str) -> None:
    if name == "football":
        from bench.data.football import build
    else:
        from bench.data.hoops import build
    src, pool, test, _ = build()
    p, lam = scale_arms(SourceFits(src), pool, test, ablations=False)
    stored = json.loads((REPO / "bench" / "results" / f"{name}.json").read_text())["sizes"]["1000000"]["arms"]
    for a in ARMS:
        print(f"[{name}] {a:22s} refit {log_loss_rows(test.y, p[a]).mean():.6f}  bench {stored[a]['log_loss']:.6f}")
    print(f"[{name}] chosen lam: {lam}")
    _, gid = np.unique(test.groups, return_inverse=True)
    np.savez_compressed(HERE / f"preds_{name}_full.npz", y=test.y.astype(np.int8), game=gid.astype(np.int32),
                        lam_ours=lam["ours_centred_cv"], lam_target=lam["target_only_cv"],
                        **{a: p[a].astype(np.float64) for a in ARMS})


if __name__ == "__main__":
    for n in sys.argv[1:] or ["football", "hoops"]:
        run(n)
