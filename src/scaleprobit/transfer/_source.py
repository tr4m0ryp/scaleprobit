"""Fit the source model once and freeze it as a ``Prior``."""
from __future__ import annotations

from .._model import ScaleProbitClassifier
from .._prior import Prior


def fit_source(X_s, y_s, *, distance=None, horizon=None, base_scale=None, sample_weight=None, **params) -> Prior:
    """Fit the plain variant (ridge at 0) on a large source sample and return it as a ``Prior``.

    ``params`` are ``ScaleProbitClassifier`` parameters (``lam``, ``link``, ``transform`` ...). The
    returned prior carries the source parameters and the source feature standardisation; nothing of
    the target is used, so the source rows must be kept disjoint from any target test data.
    """
    if params.get("prior") is not None:
        raise ValueError("fit_source fits the plain variant; do not pass prior=")
    model = ScaleProbitClassifier(**params).fit(X_s, y_s, distance=distance, horizon=horizon,
                                                base_scale=base_scale, sample_weight=sample_weight)
    return Prior.from_model(model)
