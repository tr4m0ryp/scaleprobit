"""One sample of a threshold-crossing task, in the form every arm consumes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    """Rows of one task.

    ``X`` holds positive scale features (log-transformed by the model), ``distance`` / ``horizon`` /
    ``base_scale`` the structural inputs, ``groups`` the dependence unit (a game, a draw) used for
    blocked CV and the block bootstrap, and ``flex`` the feature matrix handed to the generic learners
    (logistic regression, gradient boosting).
    """

    X: np.ndarray
    y: np.ndarray
    distance: np.ndarray
    horizon: np.ndarray
    base_scale: np.ndarray
    groups: np.ndarray
    flex: np.ndarray
    proba: np.ndarray | None = None      # true probabilities (synthetic data only)

    def __len__(self) -> int:
        return self.y.shape[0]

    def take(self, idx) -> "Split":
        return Split(self.X[idx], self.y[idx], self.distance[idx], self.horizon[idx], self.base_scale[idx],
                     self.groups[idx], self.flex[idx], None if self.proba is None else self.proba[idx])

    def structural(self) -> dict:
        return {"distance": self.distance, "horizon": self.horizon, "base_scale": self.base_scale}

    @staticmethod
    def concat(parts: list["Split"]) -> "Split":
        cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
        return Split(cat("X"), cat("y"), cat("distance"), cat("horizon"), cat("base_scale"), cat("groups"),
                     cat("flex"), None)


def subsample_groups(split: Split, n_groups: int, rng: np.random.Generator) -> Split:
    """Keep ``n_groups`` whole groups drawn without replacement (all of them if fewer exist)."""
    uniq = np.unique(split.groups)
    if n_groups >= uniq.size:
        return split
    keep = rng.choice(uniq, size=n_groups, replace=False)
    return split.take(np.isin(split.groups, keep))
