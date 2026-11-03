"""Penalty construction and the bounded L-BFGS solve.

Two ridge layouts:

* no prior (plain variant): centre 0; strength ``lam`` on ``w`` and ``b``; the scale intercept ``a``
  is free.
* prior (source-centred variant): centre ``theta0``; strength ``lam`` on every coordinate, ``a``
  included, so the target's overall scale is pulled to the source's.

In both, feature coordinates are further multiplied by their group factor. ``lam = inf`` pins every
penalised coordinate at its centre exactly. When the bias is not fitted it is pinned at 0.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._loss import objective


def penalty(k: int, lam: float, factors: np.ndarray, centre: np.ndarray | None, fit_bias: bool):
    """Return ``(pen, centre, pinned)`` for a model with ``k`` scale features."""
    if lam < 0 or np.isnan(lam):
        raise ValueError(f"lam must be >= 0, got {lam}")
    f = np.ones(k + 2)
    f[1:1 + k] = factors
    c = np.zeros(k + 2) if centre is None else np.array(centre, dtype=float).copy()
    if centre is None:
        f[0] = 0.0
    pinned = np.zeros(k + 2, dtype=bool)
    if not fit_bias:
        f[k + 1], c[k + 1], pinned[k + 1] = 0.0, 0.0, True
    if np.isinf(lam):
        pinned |= f > 0
        pen = np.zeros(k + 2)
    else:
        pen = lam * f
    return pen, c, pinned


def solve(u, log_s, Z, y, weight, pen, centre, pinned, bounds, clip, df, max_iter, tol):
    """Minimise the penalised NLL from ``centre``. Returns ``(theta, n_iter, grad_norm)``.

    ``bounds = (scale_bound, coef_bound)`` boxes ``|a|`` and ``|w_j|, |b|``; ``None`` leaves them free.
    Pinned coordinates are held at their centre. With nothing free, or ``max_iter == 0``, the centre
    is returned unchanged.
    """
    k = Z.shape[1]
    if bounds is None:
        box = [(-np.inf, np.inf)] * (k + 2)
    else:
        box = [(-bounds[0], bounds[0])] + [(-bounds[1], bounds[1])] * (k + 1)
    box = [(c, c) if p else b for b, c, p in zip(box, centre, pinned)]
    lo = np.array([b[0] for b in box])
    hi = np.array([b[1] for b in box])
    start = np.where(pinned, centre, np.clip(centre, lo, hi))

    def fun(th):
        return objective(th, u, log_s, Z, y, weight, pen, centre, clip, df)

    if pinned.all() or max_iter == 0:
        theta, n_iter = start.copy(), 0
    else:
        res = minimize(fun, start, jac=True, method="L-BFGS-B", bounds=box,
                       options={"maxiter": int(max_iter), "gtol": tol, "ftol": tol * 1e-3})
        theta, n_iter = np.where(pinned, centre, res.x), int(res.nit)
    g = fun(theta)[1]
    free = ~pinned & (theta > lo) & (theta < hi)
    return theta, n_iter, float(np.linalg.norm(g[free])) if free.any() else 0.0
