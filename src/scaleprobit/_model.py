"""``ScaleProbitClassifier``: the probability that a quantity ends above a threshold, from its distance
to the threshold, the horizon left, and a scale learned as a log-linear function of features."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from . import core
from ._prior import Prior


class ScaleProbitClassifier(ClassifierMixin, BaseEstimator):
    """Heteroscedastic threshold-crossing probit with a ridge, optionally centred at a source fit.

    For each row, with signed distance ``d`` to the threshold, horizon ``h > 0``, base scale ``s > 0``
    and scale features ``x``::

        sigma = s * exp(a + w . z),      z = (transform(x) - mean) / std
        eta   = d / (sqrt(h) * sigma) + b
        P(y = 1) = F(clip(eta, -clip, clip))

    ``F`` is the standard normal CDF (``link="probit"``) or a Student-t CDF (``link="t"``, ``df``
    degrees of freedom). With ``a = w = b = 0`` this is the closed form ``Phi(d / (s * sqrt(h)))`` of a
    driftless Brownian motion crossing the threshold.

    Parameters
    ----------
    link : {"probit", "t"}
    df : float or None
        Degrees of freedom of the t link (default 4 when ``link="t"``).
    lam : float
        Ridge strength on the summed (not averaged) log-loss. ``np.inf`` pins every penalised
        coordinate at the ridge centre.
    lam_groups : dict or array-like or None
        Extra multiplier of ``lam`` per scale feature: a length-k array, or a dict mapping a feature
        (column name or index into the scale features) to its factor. Unlisted features keep 1.
    fit_bias : bool
        Fit the bias ``b`` (a constant shift of the score); if False it is held at 0.
    transform : "log", "identity", None or callable
        Elementwise transform of the scale features before standardisation.
    clip : float
        The score is clipped to ``[-clip, clip]``; the loss is flat (zero gradient) outside.
    prior : Prior or None
        A source fit. With it the ridge is centred at ``prior.theta`` (scale intercept included), the
        optimiser starts there, and the source standardisation is reused. Without it the ridge is
        centred at 0 and ``a`` is unpenalised.
    max_iter, tol : L-BFGS-B iteration cap and gradient tolerance.
    bounds : (float, float) or None
        Box ``|a| <= bounds[0]`` and ``|w_j|, |b| <= bounds[1]`` (standardised units).
    distance_col, horizon_col, base_scale_col : column name or index, optional
        Read the structural inputs from columns of ``X`` instead of keyword arrays.

    Attributes
    ----------
    theta_ : (k + 2,) fitted ``(a, w, b)``; ``scale_intercept_``, ``coef_``, ``bias_`` its parts.
    mean_, scale_std_ : feature standardisation used (the prior's when one is given).
    n_iter_, grad_norm_ : optimiser iterations and the gradient norm reached on free coordinates.
    n_scale_terms_ : number of scale features ``k``. ``classes_`` : the two labels.

    Rows with a non-finite ``d / sqrt(h)`` or a non-positive base scale are skipped in ``fit`` and get
    probability 0.5 in ``predict_proba``. Missing scale features are set to the (source) mean.
    """

    def __init__(self, link="probit", df=None, lam=1.0, lam_groups=None, fit_bias=True, transform="log",
                 clip=12.0, prior=None, max_iter=500, tol=1e-6, bounds=(1.5, 2.0), distance_col=None,
                 horizon_col=None, base_scale_col=None):
        self.link = link
        self.df = df
        self.lam = lam
        self.lam_groups = lam_groups
        self.fit_bias = fit_bias
        self.transform = transform
        self.clip = clip
        self.prior = prior
        self.max_iter = max_iter
        self.tol = tol
        self.bounds = bounds
        self.distance_col = distance_col
        self.horizon_col = horizon_col
        self.base_scale_col = base_scale_col

    # ----- input handling -------------------------------------------------------------------
    def _inputs(self, X, distance, horizon, base_scale) -> core.Inputs:
        return core.split_inputs(X, distance, horizon, base_scale, self.distance_col, self.horizon_col,
                                 self.base_scale_col)

    def _factors(self, k: int, names) -> np.ndarray:
        g = self.lam_groups
        if g is None:
            return np.ones(k)
        if isinstance(g, dict):
            f = np.ones(k)
            for key, v in g.items():
                j = names.index(key) if (names is not None and key in names) else key
                if not isinstance(j, (int, np.integer)):
                    raise ValueError(f"lam_groups key {key!r} is not a scale feature")
                f[j] = float(v)
            return f
        f = np.asarray(g, dtype=float).reshape(-1)
        if f.shape[0] != k:
            raise ValueError(f"lam_groups has {f.shape[0]} factors for {k} scale features")
        return f

    # ----- fitting ----------------------------------------------------------------------------
    def fit(self, X, y, *, distance=None, horizon=None, base_scale=None, sample_weight=None):
        """Fit on scale features ``X`` and binary labels ``y``."""
        inp = self._inputs(X, distance, horizon, base_scale)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] != 2:
            raise ValueError(f"y must hold exactly two classes, got {self.classes_.shape[0]}")
        y01 = (y == self.classes_[1]).astype(float)
        w = np.ones(len(y01)) if sample_weight is None else np.asarray(sample_weight, float).reshape(-1)
        return self._fit_inputs(inp, y01, w)

    def _fit_inputs(self, inp: core.Inputs, y01: np.ndarray, weight: np.ndarray):
        self.df_ = core.resolve_df(self.link, self.df)
        L = core.transform_features(inp.features, self.transform)
        k = L.shape[1]
        u, log_s, valid = core.structural(inp)
        if not valid.any():
            raise ValueError("no row has a finite distance / sqrt(horizon) and a positive base scale")
        prior = self.prior
        if prior is not None:
            if not isinstance(prior, Prior):
                raise TypeError("prior must be a scaleprobit.Prior (see fit_source / Prior.from_model)")
            prior.check_compatible(self.link, self.df_, self.transform, k)
            self.mean_, self.scale_std_ = prior.mean.copy(), prior.scale_std.copy()
            centre = prior.theta
        else:
            self.mean_, self.scale_std_ = core.standardisation(L[valid])
            centre = None
        Z = core.standardise(L, self.mean_, self.scale_std_)
        pen, c, pinned = core.penalty(k, float(self.lam), self._factors(k, inp.feature_names), centre,
                                      self.fit_bias)
        theta, self.n_iter_, self.grad_norm_ = core.solve(
            u[valid], log_s[valid], Z[valid], y01[valid], weight[valid], pen, c, pinned, self.bounds,
            float(self.clip), self.df_, self.max_iter, self.tol)
        self.theta_ = theta
        self.scale_intercept_, self.coef_, self.bias_ = float(theta[0]), theta[1:1 + k].copy(), float(theta[-1])
        self.n_scale_terms_ = self.n_features_in_ = k
        self.n_samples_fit_ = int(valid.sum())
        if inp.feature_names is not None:
            self.feature_names_in_ = np.asarray(inp.feature_names, dtype=object)
        if not hasattr(self, "classes_"):
            self.classes_ = np.array([0, 1])
        return self

    # ----- prediction -------------------------------------------------------------------------
    def _design(self, inp: core.Inputs):
        check_is_fitted(self, "theta_")
        L = core.transform_features(inp.features, self.transform)
        if L.shape[1] != self.n_scale_terms_:
            raise ValueError(f"X has {L.shape[1]} scale features, the model was fitted with {self.n_scale_terms_}")
        return core.standardise(L, self.mean_, self.scale_std_)

    def _eta(self, inp: core.Inputs):
        Z = self._design(inp)
        u, log_s, valid = core.structural(inp)
        return core.scores(self.theta_, u, log_s, Z)[1], valid

    def _proba1(self, inp: core.Inputs) -> np.ndarray:
        eta, valid = self._eta(inp)
        return np.where(valid, core.probability(eta, float(self.clip), self.df_), 0.5)

    def predict_proba(self, X, *, distance=None, horizon=None, base_scale=None) -> np.ndarray:
        """Return ``(n, 2)`` class probabilities ordered as ``classes_``."""
        p = self._proba1(self._inputs(X, distance, horizon, base_scale))
        return np.column_stack([1.0 - p, p])

    def decision_function(self, X, *, distance=None, horizon=None, base_scale=None) -> np.ndarray:
        """Return the unclipped score ``eta`` (0 on unusable rows)."""
        eta, valid = self._eta(self._inputs(X, distance, horizon, base_scale))
        return np.where(valid, eta, 0.0)

    def predict(self, X, *, distance=None, horizon=None, base_scale=None) -> np.ndarray:
        """Return the more probable class per row."""
        p = self._proba1(self._inputs(X, distance, horizon, base_scale))
        return self.classes_[(p > 0.5).astype(int)]

    def scale(self, X, *, distance=None, horizon=None, base_scale=None) -> np.ndarray:
        """Return the fitted scale ``sigma = s * exp(a + w . z)`` per row (distance is not needed)."""
        if distance is None and self.distance_col is None:
            n = np.asarray(X).shape[0] if X is not None else len(np.asarray(base_scale))
            distance, horizon = np.zeros(n), np.ones(n) if horizon is None else horizon
        inp = self._inputs(X, distance, horizon, base_scale)
        Z = self._design(inp)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(inp.base_scale > 0, inp.base_scale, np.nan)
        return s * np.exp(self.theta_[0] + Z @ self.theta_[1:1 + self.n_scale_terms_])
