"""The penalised negative log-likelihood of the model and its analytic gradient.

Parameter vector ``theta = (a, w_1..w_k, b)``:

    r_i     = u_i / (s_i * exp(a + w . z_i))       (structural z-score over the fitted scale)
    eta_i   = r_i + b
    p_i     = F(clip(eta_i, -c, c))

The loss is the (weighted) binary cross-entropy plus ``0.5 * sum(pen * (theta - centre) ** 2)``.
Rows where ``eta`` is clipped, or where ``p`` hits the numerical floor, contribute a flat loss and so
a zero gradient; returning a non-zero gradient there makes L-BFGS stop after one step.
"""
from __future__ import annotations

import numpy as np

from ._link import cdf, cdf_pdf

#: Probability floor used inside the log-likelihood and in ``predict_proba``.
EPS = 1e-6


def scores(theta: np.ndarray, u: np.ndarray, log_s: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r, eta)``: the scaled structural z-score and the score with the bias added."""
    k = Z.shape[1]
    r = u * np.exp(-(log_s + theta[0] + Z @ theta[1:1 + k]))
    return r, r + theta[1 + k]


def probability(eta: np.ndarray, clip: float, df: float | None) -> np.ndarray:
    """``F(clip(eta))`` floored to ``[EPS, 1 - EPS]``."""
    return np.clip(cdf(np.clip(eta, -clip, clip), df), EPS, 1.0 - EPS)


def objective(theta, u, log_s, Z, y, weight, pen, centre, clip, df):
    """Penalised NLL and its gradient in ``theta``.

    ``weight`` holds per-row sample weights; ``pen`` the per-coordinate ridge strength; ``centre``
    the point the ridge pulls towards. The bias coordinate is handled like any other: callers fix it
    at 0 through the optimiser bounds when it is not fitted.
    """
    r, eta = scores(theta, u, log_s, Z)
    ec = np.clip(eta, -clip, clip)
    F, f = cdf_pdf(ec, df)
    p = np.clip(F, EPS, 1.0 - EPS)
    nll = -np.sum(weight * (y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    live = (np.abs(eta) < clip) & (F > EPS) & (F < 1.0 - EPS)
    g = np.where(live, -weight * f * (y - p) / (p * (1.0 - p)), 0.0)   # d nll / d eta_i
    gr = g * r
    grad = np.concatenate([[-gr.sum()], -(Z.T @ gr), [g.sum()]])
    d = theta - centre
    return nll + 0.5 * np.sum(pen * d * d), grad + pen * d
