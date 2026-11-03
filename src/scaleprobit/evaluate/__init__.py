"""Evaluation helpers: per-row losses, calibration and paired block-bootstrap comparisons."""
from ._calibration import calibration_slope, expected_calibration_error, log_loss_rows
from ._paired import paired_block_bootstrap

__all__ = ["calibration_slope", "expected_calibration_error", "log_loss_rows", "paired_block_bootstrap"]
