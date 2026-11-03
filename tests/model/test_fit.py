"""Fitting properties: recovery, an independent reference fit, weights, bias, invalid rows."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from scaleprobit import ScaleProbitClassifier
from scaleprobit.datasets import make_brownian_threshold

from ..conftest import kw


def test_recovers_true_parameters():
    b = make_brownian_threshold(60000, 3, coef=[0.4, -0.3, 0.1], scale_intercept=0.2, bias=0.1,
                                random_state=3)
    m = ScaleProbitClassifier(lam=0.0, bounds=None).fit(b.X, b.y, **kw(b))
    np.testing.assert_allclose(m.theta_, b.theta, atol=0.05)
    assert m.grad_norm_ < 1e-6 * len(b.y)


def test_matches_an_independent_reference_fit(data):
    """Our optimum equals a direct minimisation of the textbook likelihood with numeric gradients."""
    lam = 3.0
    m = ScaleProbitClassifier(lam=lam, bounds=None, tol=1e-10).fit(data.X, data.y, **kw(data))
    L = np.log(data.X)
    Z = (L - L.mean(0)) / L.std(0)
    u = data.distance / np.sqrt(data.horizon)

    def nll(t):
        sig = data.base_scale * np.exp(t[0] + Z @ t[1:4])
        p = np.clip(ndtr(u / sig + t[4]), 1e-6, 1 - 1e-6)
        return -np.sum(data.y * np.log(p) + (1 - data.y) * np.log(1 - p)) + 0.5 * lam * np.sum(t[1:] ** 2)

    ref = minimize(nll, np.zeros(5), method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 40000, "maxfev": 40000}).x
    np.testing.assert_allclose(m.theta_, ref, atol=2e-4)
    assert nll(m.theta_) <= nll(ref) + 1e-6


def test_integer_weights_equal_row_duplication(data):
    idx = np.arange(600)
    w = np.r_[np.full(300, 2.0), np.ones(300)]
    dup = np.r_[np.arange(300), idx]
    a = ScaleProbitClassifier(lam=1.0, tol=1e-10).fit(data.X[idx], data.y[idx], sample_weight=w, **kw(data, idx))
    b = ScaleProbitClassifier(lam=1.0, tol=1e-10).fit(data.X[dup], data.y[dup], **kw(data, dup))
    np.testing.assert_allclose(a.theta_, b.theta_, atol=1e-6)


def test_doubling_weights_equals_halving_lam(data):
    a = ScaleProbitClassifier(lam=4.0, tol=1e-10).fit(data.X, data.y, sample_weight=np.full(len(data.y), 2.0),
                                                      **kw(data))
    b = ScaleProbitClassifier(lam=2.0, tol=1e-10).fit(data.X, data.y, **kw(data))
    np.testing.assert_allclose(a.theta_, b.theta_, atol=1e-6)


def test_no_bias_keeps_b_exactly_zero(data):
    m = ScaleProbitClassifier(fit_bias=False).fit(data.X, data.y, **kw(data))
    assert m.bias_ == 0.0 and m.theta_[-1] == 0.0


def test_invalid_rows_are_skipped_and_predicted_at_one_half(data):
    n = 1000
    d, h = data.distance[:n].copy(), data.horizon[:n].copy()
    bad = np.zeros(n, bool)
    bad[[3, 50, 400]] = True
    d[3], h[50], h[400] = np.nan, 0.0, -1.0
    a = ScaleProbitClassifier().fit(data.X[:n], data.y[:n], distance=d, horizon=h)
    b = ScaleProbitClassifier().fit(data.X[:n][~bad], data.y[:n][~bad], distance=d[~bad], horizon=h[~bad])
    np.testing.assert_array_equal(a.theta_, b.theta_)
    p = a.predict_proba(data.X[:n], distance=d, horizon=h)[:, 1]
    assert np.all(p[bad] == 0.5)


def test_missing_features_are_imputed_at_the_mean(data):
    m = ScaleProbitClassifier().fit(data.X, data.y, **kw(data))
    X = data.X[:5].copy()
    X[0, 1] = np.nan
    X[1, 1] = -1.0                              # not loggable
    sig = m.scale(X, base_scale=data.base_scale[:5])
    Xm = X.copy()
    Xm[:2, 1] = np.exp(m.mean_[1])
    np.testing.assert_allclose(sig, m.scale(Xm, base_scale=data.base_scale[:5]), rtol=1e-12)


def test_scale_is_the_fitted_sigma(data):
    m = ScaleProbitClassifier().fit(data.X, data.y, **kw(data))
    Z = (np.log(data.X) - m.mean_) / m.scale_std_
    ref = data.base_scale * np.exp(m.scale_intercept_ + Z @ m.coef_)
    np.testing.assert_allclose(m.scale(data.X, base_scale=data.base_scale), ref, rtol=1e-12)
    eta = data.distance / (np.sqrt(data.horizon) * ref) + m.bias_
    np.testing.assert_allclose(m.decision_function(data.X, **kw(data)), eta, rtol=1e-10)


def test_labels_of_any_two_values(data):
    y = np.where(data.y == 1, "yes", "no")
    a = ScaleProbitClassifier().fit(data.X, y, **kw(data))
    b = ScaleProbitClassifier().fit(data.X, data.y, **kw(data))
    np.testing.assert_array_equal(a.theta_, b.theta_)
    assert set(a.predict(data.X, **kw(data))) <= {"yes", "no"}


def test_lam_groups_scale_the_ridge(data):
    big = ScaleProbitClassifier(lam=1.0, lam_groups={2: 1e8}).fit(data.X, data.y, **kw(data))
    arr = ScaleProbitClassifier(lam=1.0, lam_groups=[1, 1, 1e8]).fit(data.X, data.y, **kw(data))
    assert abs(big.coef_[2]) < 1e-5
    np.testing.assert_array_equal(big.theta_, arr.theta_)
