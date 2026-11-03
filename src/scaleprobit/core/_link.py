"""Link functions: the CDF and density that map the score eta to a probability.

``df=None`` is the standard normal (probit). A finite ``df`` is the Student-t CDF with that many
degrees of freedom, which has heavier tails and so gives less extreme probabilities far from the
threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, stdtr
from scipy.stats import norm, t as student_t


def resolve_df(link: str, df: float | None) -> float | None:
    """Return the degrees of freedom implied by ``link`` and ``df`` (None means probit)."""
    if link == "probit":
        if df is not None:
            raise ValueError("df must be None when link='probit'; use link='t' for a Student-t link")
        return None
    if link == "t":
        if df is None:
            df = 4.0
        if not np.isfinite(df) or df <= 0:
            raise ValueError(f"df must be a positive finite number, got {df!r}")
        return float(df)
    raise ValueError(f"link must be 'probit' or 't', got {link!r}")


def cdf_pdf(z: np.ndarray, df: float | None) -> tuple[np.ndarray, np.ndarray]:
    """CDF and density of the link at ``z``."""
    if df is None:
        return ndtr(z), norm.pdf(z)
    return stdtr(df, z), student_t.pdf(z, df)


def cdf(z: np.ndarray, df: float | None) -> np.ndarray:
    """CDF of the link at ``z``."""
    return ndtr(z) if df is None else stdtr(df, z)
