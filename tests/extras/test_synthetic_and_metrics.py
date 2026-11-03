"""Synthetic generator and evaluation helpers."""
import numpy as np
from scipy.special import ndtr

from scaleprobit.datasets import make_brownian_threshold, make_transfer_pair
from scaleprobit.evaluate import (calibration_slope, expected_calibration_error, log_loss_rows,
                                  paired_block_bootstrap)


def test_generator_probability_is_the_model():
    b = make_brownian_threshold(500, 2, coef=[0.2, 0.1], scale_intercept=0.3, bias=-0.2, base_scale=True,
                                random_state=0)
    sig = b.base_scale * np.exp(0.3 + np.log(b.X) @ [0.2, 0.1])
    np.testing.assert_allclose(b.sigma, sig)
    np.testing.assert_allclose(b.proba, ndtr(b.distance / (sig * np.sqrt(b.horizon)) - 0.2))
    assert abs(b.y.mean() - b.proba.mean()) < 0.06


def test_generator_is_reproducible_and_shift_is_exact():
    a = make_transfer_pair(1000, 100, 100, shift=0.3, random_state=5)
    b = make_transfer_pair(1000, 100, 100, shift=0.3, random_state=5)
    np.testing.assert_array_equal(a.target.y, b.target.y)
    np.testing.assert_allclose(np.linalg.norm(a.target.theta - a.source.theta), 0.3)


def test_true_probabilities_are_calibrated():
    b = make_brownian_threshold(100000, 2, random_state=1)
    assert expected_calibration_error(b.y, b.proba) < 0.01
    assert abs(calibration_slope(b.y, b.proba) - 1.0) < 0.05


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=1000)
    r = paired_block_bootstrap(a, a, blocks=np.arange(1000) // 10, random_state=0)
    assert r["mean"] == 0.0 and r["se"] == 0.0 and r["n_blocks"] == 100
    r = paired_block_bootstrap(a - 0.1, a, random_state=0)
    assert r["ci_high"] < 0 and r["p_nonneg"] == 0.0
    np.testing.assert_allclose(r["mean"], -0.1)


def test_log_loss_rows():
    np.testing.assert_allclose(log_loss_rows([1, 0], [0.8, 0.8]), [-np.log(0.8), -np.log(0.2)])
