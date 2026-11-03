"""Run the variants on one dataset's small-target sizes, paired against the default ``ours``.

    python -m bench.variants.run football|hoops
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

from scaleprobit import ScaleProbitClassifier, ScaleProbitTransferCV, fit_source
from scaleprobit.evaluate import log_loss_rows

from ..common.split import subsample_groups
from ..common.stats import arm_metrics, paired_summary
from .arms import variant_arms

OUT = Path(__file__).parents[1] / "results"
SIZES = (25, 50, 100)
REFS = ("ours_centred_cv", "source_only", "target_only_cv")


def main(name: str, n_seeds: int = 20) -> None:
    src, pool, test, meta = importlib.import_module(f"bench.data.{name}").build()
    prior = fit_source(src.X, src.y, **src.structural(), lam=1.0)
    res, t0 = {"dataset": name, "sizes": {}, "se_method": "as in the main results"}, time.time()
    for n in SIZES:
        L, M, lams = {}, {}, {}
        for seed in range(n_seeds):
            tgt = subsample_groups(pool, n, np.random.default_rng(1000 * seed + n))   # same draws as main
            kw, g = tgt.structural(), tgt.groups
            p, lam = variant_arms(prior, tgt, test)
            p["ours_centred_cv"] = ScaleProbitTransferCV(prior=prior).fit(tgt.X, tgt.y, **kw, groups=g) \
                .predict_proba(test.X, **test.structural())[:, 1]
            p["source_only"] = ScaleProbitClassifier(prior=prior, lam=np.inf).fit(tgt.X, tgt.y, **kw) \
                .predict_proba(test.X, **test.structural())[:, 1]
            p["target_only_cv"] = ScaleProbitTransferCV().fit(tgt.X, tgt.y, **kw, groups=g) \
                .predict_proba(test.X, **test.structural())[:, 1]
            for a, v in p.items():
                L.setdefault(a, []).append(log_loss_rows(test.y, v))
                M.setdefault(a, []).append(arm_metrics(test.y, v))
            for a, v in lam.items():
                lams.setdefault(a, []).append(v)
        L = {a: np.vstack(v) for a, v in L.items()}
        cell = {"chosen_lam": lams, "arms": {}}
        for a in L:
            cell["arms"][a] = {k: float(np.mean([m[k] for m in M[a]])) for k in M[a][0]}
            if a.startswith("var_"):
                cell["arms"][a]["paired_variant_minus_ref"] = {
                    r: paired_summary(L[a], L[r], test.groups, seed=n) for r in REFS}
        res["sizes"][str(n)] = cell
        print(f"[{name} variants] n={n} ({time.time() - t0:.0f}s) " + ", ".join(
            f"{a}={cell['arms'][a]['log_loss']:.4f}" for a in cell["arms"]), flush=True)
    res["wall_seconds"] = time.time() - t0
    (OUT / f"{name}_variants.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
