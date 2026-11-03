"""Input handling: split the caller's data into scale features and the structural columns,
transform and standardise the features, and form the structural z-score ``u = d / sqrt(h)``.

Scale features may be passed as ``X`` with the distance, horizon and base scale as separate keyword
arrays, or as columns of ``X`` named by ``distance_col`` / ``horizon_col`` / ``base_scale_col``
(a column name for a DataFrame, an integer index for an array). The second form makes the estimator
usable as-is inside ``cross_val_score``, ``GridSearchCV`` and ``Pipeline``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Inputs:
    features: np.ndarray          # (n, k) raw scale features
    distance: np.ndarray          # (n,)
    horizon: np.ndarray           # (n,)
    base_scale: np.ndarray        # (n,)
    feature_names: list | None    # column names of the features when X was a DataFrame


def _is_frame(X) -> bool:
    return hasattr(X, "columns") and hasattr(X, "iloc")


def _vector(v, n: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape[0] != n:
        raise ValueError(f"{name} has {a.shape[0]} rows, X has {n}")
    return a


def split_inputs(X, distance, horizon, base_scale, distance_col=None, horizon_col=None,
                 base_scale_col=None) -> Inputs:
    """Return the scale features and the three structural arrays as float numpy arrays."""
    given = {"distance": (distance, distance_col), "horizon": (horizon, horizon_col),
             "base_scale": (base_scale, base_scale_col)}
    if X is None:
        n = len(np.asarray(distance if distance is not None else horizon).reshape(-1))
        X = np.empty((n, 0))
    frame = _is_frame(X)
    n = X.shape[0]
    taken, out = [], {}
    for name, (arr, col) in given.items():
        if col is not None:
            if arr is not None:
                raise ValueError(f"pass {name} either as an array or through {name}_col, not both")
            arr = X[col].to_numpy(float) if frame else np.asarray(X, float)[:, col]
            taken.append(col)
        out[name] = arr
    if frame:
        names = [c for c in X.columns if c not in taken]
        F = X[names].to_numpy(float) if names else np.empty((n, 0))
    else:
        A = np.asarray(X, dtype=float)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        keep = [j for j in range(A.shape[1]) if j not in {int(c) % max(A.shape[1], 1) for c in taken}]
        F, names = A[:, keep], None
    if out["distance"] is None or out["horizon"] is None:
        raise ValueError("distance and horizon are required (as arrays or through distance_col / horizon_col)")
    s = np.ones(n) if out["base_scale"] is None else _vector(out["base_scale"], n, "base_scale")
    return Inputs(F, _vector(out["distance"], n, "distance"), _vector(out["horizon"], n, "horizon"), s, names)


def transform_features(F: np.ndarray, transform) -> np.ndarray:
    """Apply the elementwise feature transform. ``"log"`` maps non-positive values to NaN."""
    if transform is None or transform == "identity":
        return np.array(F, dtype=float)
    if transform == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(np.where(F > 0, F, np.nan))
    if callable(transform):
        return np.asarray(transform(F), dtype=float)
    raise ValueError(f"transform must be 'log', 'identity', None or a callable, got {transform!r}")


def standardisation(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and standard deviations over finite values (std 1 for constant columns)."""
    if L.shape[1] == 0:
        return np.zeros(0), np.ones(0)
    with np.errstate(invalid="ignore"):
        finite = np.where(np.isfinite(L), L, np.nan)
        mean = np.nan_to_num(np.nanmean(finite, axis=0)) if L.shape[0] else np.zeros(L.shape[1])
        std = np.nan_to_num(np.nanstd(finite, axis=0)) if L.shape[0] else np.ones(L.shape[1])
    return mean, np.where(std > 1e-12, std, 1.0)


def standardise(L: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(L - mean) / std`` with missing or non-finite entries set to 0 (the mean)."""
    Z = (L - mean) / std
    return np.where(np.isfinite(Z), Z, 0.0)


def structural(inp: Inputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(u, log_s, valid)``: ``u = d / sqrt(h)``, the log base scale, and the usable rows."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u = inp.distance / np.sqrt(np.where(inp.horizon > 0, inp.horizon, np.nan))
        log_s = np.log(np.where(inp.base_scale > 0, inp.base_scale, np.nan))
    valid = np.isfinite(u) & np.isfinite(log_s)
    return np.where(valid, u, 0.0), np.where(valid, log_s, 0.0), valid
