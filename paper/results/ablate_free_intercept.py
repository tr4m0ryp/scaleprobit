"""Post-hoc ablation at the full target season: leave the scale intercept ``a`` unpenalised.

The benchmark cut this ablation (bench/results/SCOPE.md). Here it is run at the full target season
only, where every subsample is the same set of games, so one deterministic fit per dataset is the
whole arm. The package penalty is wrapped so that, for the source-centred variant, the intercept's
penalty factor is 0 (free) while every other coordinate keeps its pull to the source fit; lam is
still chosen by the same grouped 5-fold CV. The other arms are not refitted: ours is read from
results/preds_<dataset>_full.npz (refit_full.py).

    PYTHONPATH=<repo>/src:<repo> python paper/results/ablate_free_intercept.py [football|hoops ...]

Writes results/abl_free_a_<dataset>.npz (the ablation's per-row test probabilities and its lam).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

import scaleprobit.core as core
from bench.common.arms_scale import SourceFits
from scaleprobit import ScaleProbitTransferCV

HERE = Path(__file__).resolve().parent
_penalty = core.penalty


def _free_intercept_penalty(k, lam, factors, centre, fit_bias):
    pen, c, pinned = _penalty(k, lam, factors, centre, fit_bias)
    if centre is not None:
        pen = pen.copy()
        pen[0], pinned = 0.0, pinned.copy()
        pinned[0] = False
    return pen, c, pinned


def run(name: str) -> None:
    if name == "football":
        from bench.data.football import build
    else:
        from bench.data.hoops import build
    src, pool, test, _ = build()
    fits = SourceFits(src)
    core.penalty = _free_intercept_penalty
    try:
        m = ScaleProbitTransferCV(prior=fits.prior).fit(pool.X, pool.y, **pool.structural(), groups=pool.groups)
    finally:
        core.penalty = _penalty
    p = m.predict_proba(test.X, **test.structural())[:, 1]
    ref = np.load(HERE / f"preds_{name}_full.npz")
    assert np.array_equal(ref["y"], test.y.astype(np.int8)), "test rows differ from refit_full.py"
    np.savez_compressed(HERE / f"abl_free_a_{name}.npz", p=p.astype(np.float64), lam=m.lam_)
    print(f"[{name}] free-intercept lam {m.lam_}")


if __name__ == "__main__":
    for n in sys.argv[1:] or ["football", "hoops"]:
        run(n)
