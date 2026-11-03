"""Learning curve over target size: every arm on S random target subsamples per size, one test set."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from scaleprobit.evaluate import log_loss_rows

from .arms_generic import SourceBooster, generic_arms
from .arms_scale import SourceFits, scale_arms
from .split import Split, subsample_groups
from .stats import arm_metrics, paired_summary

OURS = "ours_centred_cv"


def run_curve(name: str, src: Split, pool: Split, test: Split, sizes, seeds, out_path: Path, meta: dict,
              generic: bool = True) -> dict:
    t0 = time.time()
    fits = SourceFits(src)
    booster = SourceBooster(src) if generic else None
    res = {"dataset": name, "meta": meta, "n_source_rows": len(src),
           "n_source_groups": int(np.unique(src.groups).size), "n_test_rows": len(test),
           "n_test_groups": int(np.unique(test.groups).size), "prior_theta": fits.prior.theta.tolist(),
           "se_method": "block bootstrap over test groups (2000 draws) of the per-row difference averaged "
                        "over target subsamples; se_total adds the between-subsample variance",
           "sizes": {}}
    for n in sizes:
        losses, probs, lams = {}, {}, {}
        full = n >= np.unique(pool.groups).size          # the whole pool: every subsample is identical
        for seed in (seeds[:1] if full else seeds):
            tgt = subsample_groups(pool, n, np.random.default_rng(1000 * seed + n))
            p, lam = scale_arms(fits, tgt, test)
            if generic:
                p.update(generic_arms(src, booster, tgt, test))
            for arm, v in p.items():
                losses.setdefault(arm, []).append(log_loss_rows(test.y, v))
                probs.setdefault(arm, []).append(v)
            for arm, v in lam.items():
                lams.setdefault(arm, []).append(v)
        L = {a: np.vstack(v) for a, v in losses.items()}
        cell = {"n_target_groups": int(min(n, np.unique(pool.groups).size)), "chosen_lam": lams, "arms": {}}
        for arm in L:
            per = [arm_metrics(test.y, q) for q in probs[arm]]
            cell["arms"][arm] = {k: float(np.mean([m[k] for m in per])) for k in per[0]}
            cell["arms"][arm]["log_loss_per_subsample"] = [m["log_loss"] for m in per]
            if arm != OURS:
                cell["arms"][arm]["paired_ours_minus_arm"] = paired_summary(L[OURS], L[arm], test.groups, seed=n)
        if test.proba is not None:
            cell["oracle_log_loss"] = float(log_loss_rows(test.y, test.proba).mean())
        res["sizes"][str(n)] = cell
        print(f"[{name}] n={n} done at {time.time() - t0:.0f}s: "
              + ", ".join(f"{a}={cell['arms'][a]['log_loss']:.4f}" for a in cell["arms"]), flush=True)
    res["wall_seconds"] = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    return res
