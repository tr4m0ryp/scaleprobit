"""The local risk comparison (quadratic approximation), checked numerically."""
import numpy as np

from scaleprobit.theory import local_excess_risk, risk_slope_at_zero


def _spd(rng, k, n):
    A = rng.standard_normal((k, k))
    return n * (A @ A.T / k + 0.2 * np.eye(k))


def _risk(lam, J, Lam, delta, n):
    return sum(local_excess_risk(lam, J, Lam, delta, n))


def test_slope_at_zero_matches_finite_differences_and_is_negative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        k, n = 5, int(rng.integers(50, 500))
        J, Lam = _spd(rng, k, n), np.diag(rng.uniform(0.1, 2, k))
        delta = rng.standard_normal(k) * rng.uniform(0, 3)
        h = 1e-4
        fd = (_risk(h, J, Lam, delta, n) - _risk(0.0, J, Lam, delta, n)) / h
        slope = risk_slope_at_zero(J, Lam, n)
        assert slope < 0
        np.testing.assert_allclose(fd, slope, rtol=1e-2, atol=1e-7)


def test_some_positive_lam_beats_target_only():
    rng = np.random.default_rng(1)
    for _ in range(20):
        k, n = 4, 200
        J, Lam = _spd(rng, k, n), np.eye(4)
        delta = rng.standard_normal(k) * rng.uniform(0.1, 5)
        assert min(_risk(lam, J, Lam, delta, n) for lam in np.logspace(-3, 4, 60)) < _risk(0.0, J, Lam, delta, n)


def test_endpoints():
    rng = np.random.default_rng(2)
    k, n = 3, 100
    J, Lam, delta = _spd(rng, k, n), np.eye(k), rng.standard_normal(k)
    b0, v0 = local_excess_risk(0.0, J, Lam, delta, n)
    assert b0 == 0.0
    np.testing.assert_allclose(v0, k / n)                              # tr(I) / n
    np.testing.assert_allclose(_risk(1e12, J, Lam, delta, n), _risk(np.inf, J, Lam, delta, n), rtol=1e-6)
    np.testing.assert_allclose(local_excess_risk(np.inf, J, Lam, delta, n)[0], delta @ J @ delta / n)


def test_source_only_wins_when_the_gap_is_small_against_the_variance():
    rng = np.random.default_rng(3)
    k, n = 4, 100
    J, Lam = _spd(rng, k, n), np.eye(k)
    tiny = 1e-3 * rng.standard_normal(k)
    grid = np.r_[0.0, np.logspace(-2, 6, 50)]
    risks = [_risk(lam, J, Lam, tiny, n) for lam in grid]
    assert np.argmin(risks) == len(grid) - 1
    assert _risk(np.inf, J, Lam, tiny, n) < _risk(0.0, J, Lam, tiny, n)
