"""Probability-quality metrics."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def log_loss_rows(y, p, eps=1e-15) -> np.ndarray:
    """Per-row binary log-loss (useful as input to ``paired_block_bootstrap``)."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def expected_calibration_error(y, p, n_bins=15) -> float:
    """Row-weighted mean ``|mean(y) - mean(p)|`` over equal-width probability bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    n = np.bincount(idx, minlength=n_bins)
    sy = np.bincount(idx, weights=y, minlength=n_bins)
    sp = np.bincount(idx, weights=p, minlength=n_bins)
    return float(np.sum(np.abs(sy - sp)) / y.size)


def calibration_slope(y, p, eps=1e-12) -> float:
    """Slope ``beta`` of the logistic recalibration ``P(y) = sigmoid(alpha + beta * logit(p))``.

    1 is ideal; below 1 means the probabilities are too extreme, above 1 too timid.
    """
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    z = np.log(p / (1 - p))

    def f(t):
        q = 1 / (1 + np.exp(-(t[0] + t[1] * z)))
        q = np.clip(q, eps, 1 - eps)
        r = q - y
        return -np.sum(y * np.log(q) + (1 - y) * np.log1p(-q)), np.array([r.sum(), (r * z).sum()])

    return float(minimize(f, np.array([0.0, 1.0]), jac=True, method="BFGS").x[1])
