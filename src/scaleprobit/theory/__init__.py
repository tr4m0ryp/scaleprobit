"""The local risk comparison behind the lambda path (quadratic approximation)."""
from ._local_risk import local_excess_risk, risk_slope_at_zero

__all__ = ["local_excess_risk", "risk_slope_at_zero"]
