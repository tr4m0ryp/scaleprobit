"""The analytic gradient matches finite differences, including in the clipped (flat) region."""
import numpy as np
import pytest

from scaleprobit.core import objective


def _problem(rng, n=400, k=3):
    u = rng.normal(0, 3, n)
    log_s = rng.normal(0, 0.3, n)
    Z = rng.standard_normal((n, k))
    y = (rng.uniform(size=n) < 0.5).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    return u, log_s, Z, y, w


def _fd(f, th, h=1e-6):
    g = np.zeros_like(th)
    for j in range(th.size):
        e = np.zeros_like(th)
        e[j] = h
        g[j] = (f(th + e) - f(th - e)) / (2 * h)
    return g


@pytest.mark.parametrize("df", [None, 4.0])
@pytest.mark.parametrize("clip", [12.0, 1.5])
def test_gradient_matches_finite_differences(rng, df, clip):
    u, log_s, Z, y, w = _problem(rng)
    k = Z.shape[1]
    pen = np.r_[0.7, rng.uniform(0, 3, k), 2.0]
    centre = rng.normal(0, 0.2, k + 2)
    th = rng.normal(0, 0.3, k + 2)

    def f(t):
        return objective(t, u, log_s, Z, y, w, pen, centre, clip, df)[0]

    g = objective(th, u, log_s, Z, y, w, pen, centre, clip, df)[1]
    np.testing.assert_allclose(g, _fd(f, th), rtol=1e-5, atol=1e-5)


def test_gradient_is_zero_where_the_loss_is_flat(rng):
    # every row clipped: the data term is constant, so only the ridge contributes
    n, k = 50, 2
    u = np.full(n, 100.0)
    log_s = np.zeros(n)
    Z = rng.standard_normal((n, k)) * 0.01
    y = np.ones(n)
    th = np.zeros(k + 2)
    pen = np.zeros(k + 2)
    val, g = objective(th, u, log_s, Z, y, np.ones(n), pen, th, clip=3.0, df=None)
    assert np.all(g == 0.0)
    eps = 1e-3
    for j in range(k + 2):
        e = np.zeros(k + 2)
        e[j] = eps
        assert objective(th + e, u, log_s, Z, y, np.ones(n), pen, th, 3.0, None)[0] == val
