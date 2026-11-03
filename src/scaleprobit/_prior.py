"""A frozen source fit: the centre and the standardisation a target fit is pulled towards."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Prior:
    """Parameters of a source fit.

    Attributes
    ----------
    theta : (k + 2,) array
        ``(a, w_1..w_k, b)``: scale intercept, scale coefficients, bias. A target fit with this prior
        uses ``theta`` as both the ridge centre and the optimiser's starting point.
    mean, scale_std : (k,) arrays
        Standardisation of the transformed features, computed on the source and reused unchanged on
        the target so that ``w`` means the same thing in both fits.
    link, df, transform : the model settings the prior was fitted with; a target fit must match.
    n_samples : number of source rows (informational).
    """

    theta: np.ndarray
    mean: np.ndarray
    scale_std: np.ndarray
    link: str = "probit"
    df: float | None = None
    transform: object = "log"
    n_samples: int | None = None
    feature_names: tuple | None = field(default=None)

    def __post_init__(self):
        th = np.array(self.theta, dtype=float).reshape(-1)
        mu = np.array(self.mean, dtype=float).reshape(-1)
        sd = np.array(self.scale_std, dtype=float).reshape(-1)
        if th.shape[0] != mu.shape[0] + 2 or mu.shape != sd.shape:
            raise ValueError("theta must have len(mean) + 2 entries and mean / scale_std the same length")
        if np.any(sd <= 0) or not (np.all(np.isfinite(th)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))):
            raise ValueError("prior parameters must be finite and scale_std positive")
        for name, v in (("theta", th), ("mean", mu), ("scale_std", sd)):
            v.setflags(write=False)
            object.__setattr__(self, name, v)

    @property
    def n_features(self) -> int:
        return self.mean.shape[0]

    @classmethod
    def from_model(cls, model) -> "Prior":
        """Build a prior from any fitted ``ScaleProbitClassifier``."""
        if not hasattr(model, "theta_"):
            raise ValueError("the model is not fitted")
        names = getattr(model, "feature_names_in_", None)
        return cls(theta=model.theta_.copy(), mean=model.mean_.copy(), scale_std=model.scale_std_.copy(),
                   link=model.link, df=model.df_, transform=model.transform,
                   n_samples=getattr(model, "n_samples_fit_", None),
                   feature_names=None if names is None else tuple(names))

    def check_compatible(self, link: str, df: float | None, transform, n_features: int) -> None:
        """Raise if a target model's settings differ from the ones the prior was fitted with."""
        if (link, df) != (self.link, self.df):
            raise ValueError(f"prior was fitted with link={self.link!r}, df={self.df}; got link={link!r}, df={df}")
        if transform != self.transform:
            raise ValueError(f"prior was fitted with transform={self.transform!r}; got {transform!r}")
        if n_features != self.n_features:
            raise ValueError(f"prior has {self.n_features} scale features; X has {n_features}")
