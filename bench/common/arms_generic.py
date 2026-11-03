"""Generic learners as rivals: logistic regression on the structural terms and gradient boosting.

Logistic regression sees ``[u, u * z, z]`` with ``u = distance / sqrt(horizon)`` and ``z`` the
standardised log scale features (the linearised form of the scale model). Gradient boosting sees the
dataset's ``flex`` matrix. Both pick their capacity by grouped CV on the target rows only.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold

from .split import Split

LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 40,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": 2, "verbose": -1, "seed": 0}


def _lr_design(s: Split, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    u = s.distance / np.sqrt(s.horizon) / s.base_scale
    z = (np.log(s.X) - mu) / sd
    return np.column_stack([u, u[:, None] * z, z])


def _folds(groups, k=5):
    k = min(k, np.unique(groups).size)
    return list(GroupKFold(n_splits=k).split(groups, groups=groups))


def _lr(train: Split, test: Split, extra=None) -> np.ndarray:
    mu, sd = np.log(train.X).mean(0), np.log(train.X).std(0) + 1e-12
    A, B = _lr_design(train, mu, sd), _lr_design(test, mu, sd)
    if extra is not None:
        A, B = np.column_stack([A, extra[0]]), np.column_stack([B, extra[1]])
    m = LogisticRegressionCV(Cs=8, cv=_folds(train.groups), scoring="neg_log_loss", max_iter=2000)
    return m.fit(A, train.y).predict_proba(B)[:, 1]


class SourceBooster:
    """A booster fitted on the source, its rounds chosen on held-out source groups."""

    def __init__(self, src: Split, max_rounds: int = 1500):
        uniq = np.unique(src.groups)
        rng = np.random.default_rng(0)
        hold = np.isin(src.groups, rng.choice(uniq, size=max(1, uniq.size // 5), replace=False))
        dtr = lgb.Dataset(src.flex[~hold], src.y[~hold])
        dva = lgb.Dataset(src.flex[hold], src.y[hold], reference=dtr)
        self.booster = lgb.train(LGB_PARAMS, dtr, max_rounds, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(50, verbose=False)])


def _lgb_cv_rounds(train: Split, init=None, max_rounds=1000) -> int:
    d = lgb.Dataset(train.flex, train.y, free_raw_data=False)
    if init is not None:
        d.set_init_score(init.predict(train.flex, raw_score=True))
    r = lgb.cv(LGB_PARAMS, d, max_rounds, folds=_folds(train.groups), stratified=False,
               callbacks=[lgb.early_stopping(30, verbose=False)])
    return max(1, len(next(iter(r.values()))))


def generic_arms(src: Split, booster: SourceBooster, tgt: Split, test: Split) -> dict:
    out = {"logreg_target_only": _lr(tgt, test)}
    pool = Split.concat([src, tgt])
    ind = (np.concatenate([np.zeros(len(src)), np.ones(len(tgt))])[:, None], np.ones((len(test), 1)))
    out["logreg_pooled_indicator"] = _lr(pool, test, extra=ind)
    n = _lgb_cv_rounds(tgt)
    out["lgbm_target_only"] = lgb.train(LGB_PARAMS, lgb.Dataset(tgt.flex, tgt.y), n).predict(test.flex)
    b = booster.booster
    out["lgbm_source_only"] = b.predict(test.flex)
    n2 = _lgb_cv_rounds(tgt, init=b, max_rounds=300)
    d = lgb.Dataset(tgt.flex, tgt.y)
    d.set_init_score(b.predict(tgt.flex, raw_score=True))
    ft = lgb.train(LGB_PARAMS, d, n2)
    raw = b.predict(test.flex, raw_score=True) + ft.predict(test.flex, raw_score=True)
    out["lgbm_finetuned"] = 1.0 / (1.0 + np.exp(-raw))
    return out
