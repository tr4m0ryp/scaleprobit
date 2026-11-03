"""Does the one-standard-error rule still adapt when the target really differs (synthetic, known truth)?

The real-data variants showed the rule always landing on lam = inf. This checks the cells of the
synthetic map where adaptation should pay (shift 0.3 and 0.6) and one where it should not (0.1).

    python -m bench.variants.synthetic_one_se
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

from scaleprobit import ScaleProbitClassifier, ScaleProbitTransferCV, fit_source
from scaleprobit.datasets import make_transfer_pair
from scaleprobit.evaluate import log_loss_rows

OUT = Path(__file__).parents[1] / "results" / "synthetic_one_se.json"


def one_se(prior, t, kw):
    r = ScaleProbitTransferCV(prior=prior).fit(t.X, t.y, **kw).cv_results_
    nf = sum(1 for k in r if k.startswith("split"))
    b = np.argmin(r["mean_test_log_loss"])
    thr = r["mean_test_log_loss"][b] + r["std_test_log_loss"][b] / np.sqrt(nf)
    lam = float(np.max(r["lam"][r["mean_test_log_loss"] <= thr]))
    return ScaleProbitClassifier(prior=prior, lam=lam).fit(t.X, t.y, **kw), lam


def main(n_seeds=10):
    res = {"se_method": "standard error over independent draws", "n_draws": n_seeds, "cells": {}}
    for shift in (0.1, 0.3, 0.6):
        for n in (50, 200, 800):
            d, lams = {"ours_centred_cv": [], "source_only": [], "target_only_cv": []}, []
            for s in range(n_seeds):        # the same draws as bench/run_synthetic.py
                p = make_transfer_pair(n_source=20000, n_target=n, n_test=5000, shift=shift, n_features=4,
                                       random_state=10_000 * s + n + int(shift * 1000))
                src, t, te = p.source, p.target, p.test
                pr = fit_source(src.X, src.y, distance=src.distance, horizon=src.horizon, lam=1.0)
                kw, kte = {"distance": t.distance, "horizon": t.horizon}, {"distance": te.distance, "horizon": te.horizon}
                ll = lambda m: log_loss_rows(te.y, m.predict_proba(te.X, **kte)[:, 1]).mean()
                m, lam = one_se(pr, t, kw)
                lams.append(lam)
                v = ll(m)
                d["ours_centred_cv"].append(v - ll(ScaleProbitTransferCV(prior=pr).fit(t.X, t.y, **kw)))
                d["source_only"].append(v - ll(ScaleProbitClassifier(prior=pr, lam=np.inf).fit(t.X, t.y, **kw)))
                d["target_only_cv"].append(v - ll(ScaleProbitTransferCV().fit(t.X, t.y, **kw)))
            cell = {"chosen_lam": lams}
            for k, v in d.items():
                v = np.asarray(v)
                cell[f"one_se_minus_{k}"] = {"mean_diff": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(v.size))}
            res["cells"][f"shift={shift:g},n={n}"] = cell
            print(shift, n, sorted(set(lams)), {k: f"{c['mean_diff']:+.4f}({c['se']:.4f})"
                                               for k, c in cell.items() if k != "chosen_lam"}, flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
