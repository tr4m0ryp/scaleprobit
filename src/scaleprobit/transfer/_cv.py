"""``ScaleProbitTransferCV``: choose the ridge strength towards the source fit by cross-validation."""
from __future__ import annotations

import numpy as np

from .. import core
from .._model import ScaleProbitClassifier
from ._folds import make_folds

DEFAULT_LAMS = (0.0, 1.0, 10.0, 100.0, 1e3, 1e4, np.inf)


def _log_loss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p, core.EPS, 1.0 - core.EPS)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p))) / np.sum(w))


class ScaleProbitTransferCV(ScaleProbitClassifier):
    """``ScaleProbitClassifier`` whose ``lam`` is picked from a grid by out-of-fold log-loss.

    With a ``prior`` the grid is a path between the two reference fits: ``lam = 0`` is the
    unpenalised target fit started at the source parameters, ``lam = inf`` returns the source
    parameters unchanged. Both endpoints are always scored (``include_endpoints=True``) and labelled
    in ``cv_results_["role"]``, so the table shows directly whether adapting to the target paid.
    Without a prior the same search runs for the plain ridge centred at 0.

    Parameters (beyond ``ScaleProbitClassifier``'s, whose ``lam`` is ignored)
    ----------
    lams : sequence of float
        Candidate ridge strengths.
    cv : int, splitter or iterable of (train, validation) index pairs
        An int gives ``GroupKFold`` when ``groups`` is passed to ``fit`` (use it for blocked or
        clustered rows) and an unshuffled ``KFold`` otherwise.
    include_endpoints : bool
        Add 0 and inf to ``lams`` when missing.

    Attributes
    ----------
    lam_ : the chosen strength (ties go to the larger, more shrunk value).
    best_score_ : its mean out-of-fold log-loss.
    cv_results_ : dict of arrays: ``lam``, ``role``, ``mean_test_log_loss``, ``std_test_log_loss``,
        ``split<j>_test_log_loss``, ``rank_test_log_loss``.
    cv_oof_proba_ : (n_lams, n) out-of-fold P(class 1); each row's value comes from a model that never
        saw that row's fold.
    """

    def __init__(self, link="probit", df=None, lam=1.0, lam_groups=None, fit_bias=True, transform="log",
                 clip=12.0, prior=None, max_iter=500, tol=1e-6, bounds=(1.5, 2.0), distance_col=None,
                 horizon_col=None, base_scale_col=None, lams=DEFAULT_LAMS, cv=5, include_endpoints=True):
        super().__init__(link=link, df=df, lam=lam, lam_groups=lam_groups, fit_bias=fit_bias,
                         transform=transform, clip=clip, prior=prior, max_iter=max_iter, tol=tol, bounds=bounds,
                         distance_col=distance_col, horizon_col=horizon_col, base_scale_col=base_scale_col)
        self.lams = lams
        self.cv = cv
        self.include_endpoints = include_endpoints

    def _grid(self) -> np.ndarray:
        g = [float(v) for v in self.lams]
        if self.include_endpoints:
            g += [0.0, np.inf]
        g = np.unique(np.asarray(g, dtype=float))
        if g.size == 0 or np.any(g < 0) or np.any(np.isnan(g)):
            raise ValueError("lams must be a non-empty set of values >= 0")
        return g

    def _member(self, lam: float) -> ScaleProbitClassifier:
        params = {k: v for k, v in self.get_params(deep=False).items() if k not in ("lams", "cv", "include_endpoints")}
        params["lam"] = lam
        return ScaleProbitClassifier(**params)

    def fit(self, X, y, *, distance=None, horizon=None, base_scale=None, sample_weight=None, groups=None):
        """Score every ``lam`` by out-of-fold log-loss, keep the best, refit it on all rows."""
        inp = self._inputs(X, distance, horizon, base_scale)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] != 2:
            raise ValueError(f"y must hold exactly two classes, got {self.classes_.shape[0]}")
        y01 = (y == self.classes_[1]).astype(float)
        w = np.ones(len(y01)) if sample_weight is None else np.asarray(sample_weight, float).reshape(-1)
        folds = make_folds(self.cv, len(y01), None if groups is None else np.asarray(groups))
        grid = self._grid()
        oof = np.full((grid.size, len(y01)), 0.5)
        split_scores = np.zeros((grid.size, len(folds)))
        for i, lam in enumerate(grid):
            for j, (tr, va) in enumerate(folds):
                m = self._member(lam)._fit_inputs(_take(inp, tr), y01[tr], w[tr])
                oof[i, va] = m._proba1(_take(inp, va))
                split_scores[i, j] = _log_loss(y01[va], oof[i, va], w[va])
        mean, std = split_scores.mean(axis=1), split_scores.std(axis=1)
        order = np.lexsort((-grid, np.round(mean, 12)))       # lowest loss, ties to the larger lam
        best = int(order[0])
        rank = np.empty(grid.size, dtype=int)
        rank[order] = np.arange(1, grid.size + 1)
        src = self.prior is not None
        roles = np.array(["target-only" if v == 0 else ("source-only" if src else "scale-intercept-only")
                          if np.isinf(v) else "" for v in grid], dtype=object)
        self.cv_results_ = {"lam": grid, "role": roles, "mean_test_log_loss": mean, "std_test_log_loss": std,
                            "rank_test_log_loss": rank,
                            **{f"split{j}_test_log_loss": split_scores[:, j] for j in range(len(folds))}}
        self.cv_oof_proba_ = oof
        self.lam_, self.best_score_ = float(grid[best]), float(mean[best])
        final = self._member(self.lam_)._fit_inputs(inp, y01, w)
        for k, v in vars(final).items():
            if k.endswith("_") and not k.startswith("_") and k != "classes_":
                setattr(self, k, v)
        return self


def _take(inp: core.Inputs, idx: np.ndarray) -> core.Inputs:
    return core.Inputs(inp.features[idx], inp.distance[idx], inp.horizon[idx], inp.base_scale[idx],
                       inp.feature_names)
