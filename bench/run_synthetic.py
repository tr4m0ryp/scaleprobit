"""Synthetic regime map: gain of the source-centred ridge over each arm, by target size and shift.

Each (shift, size, seed) cell draws a fresh source (20,000 rows), target and test set (5,000 rows)
from ``make_transfer_pair``; the source parameters are random per seed. Draws are independent, so the
paired SE is the plain standard error of the per-draw mean difference over seeds.

    python -m bench.run_synthetic [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

from scaleprobit.datasets import make_transfer_pair
from scaleprobit.evaluate import log_loss_rows

from .common.arms_generic import _lr
from .common.arms_scale import SourceFits, scale_arms
from .common.split import Split
from .common.stats import arm_metrics

OUT = Path(__file__).parent / "results" / "synthetic.json"
OURS = "ours_centred_cv"


def to_split(b, offset: int) -> Split:
    n = b.y.shape[0]
    u = b.distance / np.sqrt(b.horizon)
    flex = np.column_stack([u, b.distance, b.horizon, np.log(b.X)])
    return Split(b.X, b.y, b.distance, b.horizon, np.ones(n), offset + np.arange(n), flex, b.proba)


def cell(shift, n, seed):
    d = make_transfer_pair(n_source=20000, n_target=n, n_test=5000, shift=shift, n_features=4,
                           random_state=10_000 * seed + n + int(shift * 1000))
    src, tgt, test = to_split(d.source, 0), to_split(d.target, 10**6), to_split(d.test, 2 * 10**6)
    p, lam = scale_arms(SourceFits(src), tgt, test)
    p["logreg_target_only"] = _lr(tgt, test)
    p["oracle"] = test.proba
    return {a: log_loss_rows(test.y, v) for a, v in p.items()}, {a: arm_metrics(test.y, v) for a, v in p.items()}, lam


def main(quick=False):
    shifts = (0.0, 0.1, 0.3, 0.6)
    sizes = (50, 100, 200, 400, 800)
    seeds = range(3 if quick else 20)
    res, t0 = {"dataset": "synthetic", "shifts": shifts, "sizes": sizes, "n_seeds": len(seeds),
               "se_method": "standard error over independent draws (seeds) of the per-draw mean difference",
               "cells": {}}, time.time()
    for shift in shifts:
        for n in sizes:
            diffs, mets, lams = {}, {}, []
            for s in seeds:
                L, M, lam = cell(shift, n, s)
                lams.append(lam[OURS])
                for a in L:
                    diffs.setdefault(a, []).append(float(np.mean(L[OURS] - L[a])))
                    mets.setdefault(a, []).append(M[a])
            c = {"chosen_lam_ours": lams, "arms": {}}
            for a in mets:
                c["arms"][a] = {k: float(np.mean([m[k] for m in mets[a]])) for k in mets[a][0]}
                if a != OURS:
                    d = np.asarray(diffs[a])
                    c["arms"][a]["paired_ours_minus_arm"] = {
                        "mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(d.size)),
                        "n_draws": int(d.size), "n_test_rows_per_draw": 5000, "draw_wins": int(np.sum(d < 0))}
            res["cells"][f"shift={shift:g},n={n}"] = c
            pr = c["arms"]
            print(f"shift {shift:g} n {n} ({time.time() - t0:.0f}s): ours {pr[OURS]['log_loss']:.4f} "
                  + " ".join(f"{a}={pr[a]['paired_ours_minus_arm']['mean_diff']:+.4f}"
                             f"({pr[a]['paired_ours_minus_arm']['se']:.4f})" for a in pr if a != OURS), flush=True)
    res["wall_seconds"] = time.time() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
