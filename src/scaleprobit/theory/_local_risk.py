"""Local risk of the source-centred ridge in the quadratic approximation.

Around the target optimum ``theta_*`` the penalised fit is approximately

    theta(lam) - theta_* = A^{-1} (lam * Lam @ delta + g),   A = J + lam * Lam,   g ~ (0, J)

with ``J`` the total Fisher information of the ``n`` target rows, ``Lam`` the diagonal penalty
weights and ``delta = theta_0 - theta_*`` the source-target gap. Measured in the per-row Fisher metric
``J / n`` the expected excess risk splits into

    bias(lam)     = lam^2 delta' Lam A^{-1} (J / n) A^{-1} Lam delta
    variance(lam) = tr(A^{-1} J A^{-1} J) / n

At ``lam = 0`` the bias has zero slope while the variance falls with slope ``-2 tr(J^{-1} Lam) / n``,
so a small positive ``lam`` always beats the unpenalised target fit; as ``lam -> inf`` the risk tends
to ``delta' (J / n) delta`` (the source fit). This is the classic biased-regularisation argument
(Kuzborskij and Orabona 2013) restated for this likelihood.
"""
from __future__ import annotations

import numpy as np


def local_excess_risk(lam: float, J, Lam, delta, n: int) -> tuple[float, float]:
    """Return ``(bias, variance)`` of the quadratic-approximation excess risk at ``lam``."""
    J, Lam, delta = np.asarray(J, float), np.asarray(Lam, float), np.asarray(delta, float)
    if np.isinf(lam):
        return float(delta @ (J / n) @ delta), 0.0
    Ai = np.linalg.inv(J + lam * Lam)
    m = lam * Ai @ Lam @ delta
    return float(m @ (J / n) @ m), float(np.trace(Ai @ J @ Ai @ J) / n)


def risk_slope_at_zero(J, Lam, n: int) -> float:
    """Derivative of the excess risk in ``lam`` at ``lam = 0``: ``-2 tr(J^{-1} Lam) / n`` (< 0)."""
    J, Lam = np.asarray(J, float), np.asarray(Lam, float)
    return float(-2.0 * np.trace(np.linalg.solve(J, Lam)) / n)
