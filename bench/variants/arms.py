"""Variants of the source-centred ridge, each answering "what would have to be true for it to win".

- ``dense_grid``: the penalty sweep. A 13-point log grid 1..1e6 instead of 7 points, so CV can stop
  between the decades where the default grid jumps.
- ``slopes_anchored``: only the scale intercept and the bias adapt freely-ish; the slopes get a
  10,000x stronger pull. True if the shift between tasks is a level shift (pace, home edge), not a
  change of the feature effects, which a small target cannot identify anyway.
- ``one_se``: pick the largest lam whose CV loss is within one standard error of the best. True if CV
  on a few dozen groups picks too small a lam by noise (the quickstart's symptom).
- ``repeated_cv``: the same grid scored on 3 shuffled group 5-fold splits and averaged. True if the
  selection noise is fold-assignment noise.
"""
from __future__ import annotations

import numpy as np

from scaleprobit import ScaleProbitClassifier, ScaleProbitTransferCV

from ..common.split import Split

DENSE = tuple(np.logspace(0, 6, 13))


def _p(m, test: Split) -> np.ndarray:
    return m.predict_proba(test.X, **test.structural())[:, 1]


def _shuffled_folds(groups, seed, k=5):
    """Whole groups dealt at random into ``k`` folds."""
    uniq = np.random.default_rng(seed).permutation(np.unique(groups))
    chunks = np.array_split(uniq, min(k, uniq.size))
    idx = np.arange(len(groups))
    return [(idx[~np.isin(groups, c)], idx[np.isin(groups, c)]) for c in chunks]


def variant_arms(prior, tgt: Split, test: Split) -> tuple[dict, dict]:
    kw, g = tgt.structural(), tgt.groups
    out, lam = {}, {}
    m = ScaleProbitTransferCV(prior=prior, lams=DENSE).fit(tgt.X, tgt.y, **kw, groups=g)
    out["var_dense_grid"], lam["var_dense_grid"] = _p(m, test), m.lam_

    k = tgt.X.shape[1]
    m = ScaleProbitTransferCV(prior=prior, lams=DENSE, lam_groups=np.full(k, 1e4)).fit(tgt.X, tgt.y, **kw, groups=g)
    out["var_slopes_anchored"], lam["var_slopes_anchored"] = _p(m, test), m.lam_

    r = ScaleProbitTransferCV(prior=prior, lams=DENSE).fit(tgt.X, tgt.y, **kw, groups=g).cv_results_
    nf = sum(1 for key in r if key.startswith("split"))
    best = np.argmin(r["mean_test_log_loss"])
    thr = r["mean_test_log_loss"][best] + r["std_test_log_loss"][best] / np.sqrt(nf)
    pick = float(np.max(r["lam"][r["mean_test_log_loss"] <= thr]))
    m = ScaleProbitClassifier(prior=prior, lam=pick).fit(tgt.X, tgt.y, **kw)
    out["var_one_se"], lam["var_one_se"] = _p(m, test), pick

    scores = 0.0
    for s in range(3):
        folds = _shuffled_folds(g, s)
        cv = ScaleProbitTransferCV(prior=prior, lams=DENSE, cv=folds).fit(tgt.X, tgt.y, **kw)
        scores = scores + cv.cv_results_["mean_test_log_loss"]
    grid = cv.cv_results_["lam"]
    order = np.lexsort((-grid, np.round(scores, 12)))
    pick = float(grid[order[0]])
    m = ScaleProbitClassifier(prior=prior, lam=pick).fit(tgt.X, tgt.y, **kw)
    out["var_repeated_cv"], lam["var_repeated_cv"] = _p(m, test), pick
    return out, lam
