"""Metrics per arm and the paired comparison ours - arm.

The paired standard error is a block bootstrap over test groups (games, or independent synthetic
draws): rows of one group are dependent, groups are independent. The per-row difference is first
averaged over the target subsamples (all evaluated on the same test rows), then whole groups are
resampled. The spread of the per-subsample mean differences is reported separately
(``sd_between_subsamples``); ``se_total`` combines both as sqrt(se_block^2 + sd_between^2 / S).
"""
from __future__ import annotations

import numpy as np

from scaleprobit.evaluate import (calibration_slope, expected_calibration_error, log_loss_rows,
                                  paired_block_bootstrap)


def arm_metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return {"log_loss": float(log_loss_rows(y, p).mean()), "brier": float(np.mean((p - y) ** 2)),
            "ece": expected_calibration_error(y, p), "calibration_slope": calibration_slope(y, p)}


def paired_summary(loss_ours: np.ndarray, loss_arm: np.ndarray, groups: np.ndarray, seed: int = 0,
                   n_boot: int = 2000) -> dict:
    """``loss_*``: (S, n) per-row log-loss for S target subsamples on one shared test set."""
    diff_s = loss_ours - loss_arm
    per_sub = diff_s.mean(axis=1)
    bb = paired_block_bootstrap(loss_ours.mean(0), loss_arm.mean(0), blocks=groups, n_boot=n_boot,
                                random_state=seed)
    S = diff_s.shape[0]
    sd_b = float(per_sub.std(ddof=1)) if S > 1 else 0.0
    se_tot = float(np.sqrt(bb["se"] ** 2 + sd_b ** 2 / S))
    return {"mean_diff": bb["mean"], "se_block": bb["se"], "ci95_block": [bb["ci_low"], bb["ci_high"]],
            "p_diff_nonneg": bb["p_nonneg"], "sd_between_subsamples": sd_b, "se_total": se_tot,
            "upper95_total": bb["mean"] + 1.96 * se_tot, "n_subsamples": S, "n_test_rows": int(diff_s.shape[1]),
            "n_test_groups": bb["n_blocks"], "subsample_wins": int(np.sum(per_sub < 0))}
