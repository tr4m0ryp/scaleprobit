"""Synthetic threshold-crossing data with a known scale model.

Each row is a Brownian path observed at distance ``d`` from a threshold with horizon ``h`` left. The
path's volatility is ``sigma = s * exp(a + w . log(x))`` with ``log(x)`` standard normal, and the
label is whether the path ends above the threshold:

    y = 1[d + sigma * sqrt(h) * (b + eps) > 0],   eps ~ N(0, 1)

so ``P(y = 1) = Phi(d / (sigma * sqrt(h)) + b)`` exactly: the model is well specified, and the true
parameters are known (in population-standardised units).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from sklearn.utils import Bunch, check_random_state


def make_brownian_threshold(n_samples=1000, n_features=2, coef=None, scale_intercept=0.0, bias=0.0,
                            horizon_range=(1.0, 100.0), distance_spread=2.0, base_scale=False,
                            random_state=None) -> Bunch:
    """Draw ``n_samples`` rows.

    Parameters
    ----------
    coef : (n_features,) true scale coefficients on log-features (default ``0.3, -0.2, 0, ...``).
    scale_intercept, bias : true ``a`` and ``b``.
    horizon_range : uniform range of ``h``.
    distance_spread : ``d`` is drawn as ``N(0, distance_spread * sigma * sqrt(h))``, so most rows are
        undecided rather than trivially far from the threshold.
    base_scale : if True, also draw a per-row base scale ``s`` (lognormal) and return it; otherwise
        ``s = 1``.

    Returns
    -------
    Bunch with ``X`` (n, k) positive features, ``y``, ``distance``, ``horizon``, ``base_scale``
    (array or None), ``proba`` (true P(y = 1)), ``sigma`` (true scale) and ``theta`` (true ``(a, w, b)``).
    """
    rng = check_random_state(random_state)
    if coef is None:
        coef = np.zeros(n_features)
        coef[:2] = [0.3, -0.2][:n_features]
    coef = np.asarray(coef, dtype=float).reshape(-1)
    if coef.shape[0] != n_features:
        raise ValueError("coef must have n_features entries")
    logx = rng.standard_normal((n_samples, n_features))
    s = np.exp(0.5 * rng.standard_normal(n_samples)) if base_scale else np.ones(n_samples)
    h = rng.uniform(*horizon_range, size=n_samples)
    sigma = s * np.exp(scale_intercept + logx @ coef)
    d = rng.normal(0.0, distance_spread * sigma * np.sqrt(h))
    proba = ndtr(d / (sigma * np.sqrt(h)) + bias)
    y = (rng.uniform(size=n_samples) < proba).astype(int)
    return Bunch(X=np.exp(logx), y=y, distance=d, horizon=h, base_scale=s if base_scale else None,
                 proba=proba, sigma=sigma, theta=np.concatenate([[scale_intercept], coef, [bias]]))


def make_transfer_pair(n_source=20000, n_target=300, n_test=5000, shift=0.1, n_features=4,
                       random_state=None) -> Bunch:
    """A large source sample, a small target sample and a target test sample.

    The target's true ``(a, w, b)`` is the source's plus a random direction of Euclidean length
    ``shift``; ``shift = 0`` makes the two tasks identical.

    Returns a Bunch with ``source``, ``target``, ``test`` (each from ``make_brownian_threshold``) and
    ``shift``.
    """
    rng = check_random_state(random_state)
    a_s, b_s = 0.1, 0.0
    w_s = rng.uniform(-0.4, 0.4, size=n_features)
    theta_s = np.concatenate([[a_s], w_s, [b_s]])
    v = rng.standard_normal(n_features + 2)
    theta_t = theta_s + shift * v / np.linalg.norm(v)

    def draw(n, th):
        return make_brownian_threshold(n, n_features, coef=th[1:-1], scale_intercept=th[0], bias=th[-1],
                                       random_state=rng.randint(2**31 - 1))

    return Bunch(source=draw(n_source, theta_s), target=draw(n_target, theta_t), test=draw(n_test, theta_t),
                 shift=float(shift))
