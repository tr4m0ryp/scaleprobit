"""Source-centred ridge: endpoints, start at the prior, frozen standardisation, equivalences."""
import numpy as np
import pytest

from scaleprobit import Prior, ScaleProbitClassifier, ScaleProbitTransferCV, fit_source

from ..conftest import kw


@pytest.fixture(scope="module")
def prior(pair):
    s = pair.source
    return fit_source(s.X, s.y, **kw(s), lam=1.0)


def test_fit_source_equals_a_plain_fit(pair, prior):
    s = pair.source
    m = ScaleProbitClassifier(lam=1.0).fit(s.X, s.y, **kw(s))
    ref = Prior.from_model(m)
    for f in ("theta", "mean", "scale_std"):
        np.testing.assert_array_equal(getattr(prior, f), getattr(ref, f))
    assert prior.n_samples == len(s.y)


def test_infinite_lam_returns_the_source_fit_exactly(pair, prior):
    t = pair.target
    m = ScaleProbitClassifier(prior=prior, lam=np.inf).fit(t.X, t.y, **kw(t))
    np.testing.assert_array_equal(m.theta_, prior.theta)
    assert m.n_iter_ == 0
    src = ScaleProbitClassifier(lam=1.0).fit(pair.source.X, pair.source.y, **kw(pair.source))
    np.testing.assert_array_equal(m.predict_proba(t.X, **kw(t)), src.predict_proba(t.X, **kw(t)))


def test_optimiser_starts_at_the_prior(pair, prior):
    t = pair.target
    m = ScaleProbitClassifier(prior=prior, lam=0.0, max_iter=0).fit(t.X, t.y, **kw(t))
    np.testing.assert_array_equal(m.theta_, prior.theta)


def test_source_standardisation_is_frozen(pair, prior):
    t = pair.target
    m = ScaleProbitClassifier(prior=prior, lam=10.0).fit(t.X * 3.0, t.y, **kw(t))
    np.testing.assert_array_equal(m.mean_, prior.mean)
    np.testing.assert_array_equal(m.scale_std_, prior.scale_std)


def test_large_finite_lam_converges_to_the_infinite_endpoint(pair, prior):
    t = pair.target
    m = ScaleProbitClassifier(prior=prior, lam=1e10).fit(t.X, t.y, **kw(t))
    np.testing.assert_allclose(m.theta_, prior.theta, atol=1e-6)


def test_lam_path_moves_monotonically_towards_the_prior(pair, prior):
    t = pair.target
    dist = [np.linalg.norm(ScaleProbitClassifier(prior=prior, lam=lam, tol=1e-10).fit(t.X, t.y, **kw(t)).theta_
                           - prior.theta) for lam in (0.0, 1.0, 10.0, 100.0, 1e3, 1e4)]
    assert all(b <= a + 1e-6 for a, b in zip(dist, dist[1:]))
    assert dist[0] > 10 * dist[-1]


def test_scale_intercept_is_pulled_to_the_source(pair, prior):
    """With a prior, a is penalised towards theta0[0]; without, it is free."""
    t = pair.target
    with_prior = ScaleProbitClassifier(prior=prior, lam=1e8).fit(t.X, t.y, **kw(t))
    without = ScaleProbitClassifier(lam=1e8).fit(t.X, t.y, **kw(t))
    assert abs(with_prior.scale_intercept_ - prior.theta[0]) < 1e-6
    assert np.all(np.abs(without.theta_[1:]) < 1e-6) and abs(without.scale_intercept_) > 1e-3


def test_no_bias_with_a_prior_keeps_b_zero(pair):
    s, t = pair.source, pair.target
    pr = fit_source(s.X, s.y, **kw(s))
    assert pr.theta[-1] != 0.0
    m = ScaleProbitClassifier(prior=pr, fit_bias=False, lam=10.0).fit(t.X, t.y, **kw(t))
    assert m.bias_ == 0.0
    m_inf = ScaleProbitClassifier(prior=pr, fit_bias=False, lam=np.inf).fit(t.X, t.y, **kw(t))
    np.testing.assert_array_equal(m_inf.theta_[:-1], pr.theta[:-1])
    assert m_inf.bias_ == 0.0


def test_prior_must_match_the_model(pair, prior):
    t = pair.target
    with pytest.raises(ValueError):
        ScaleProbitClassifier(prior=prior, link="t").fit(t.X, t.y, **kw(t))
    with pytest.raises(ValueError):
        ScaleProbitClassifier(prior=prior, transform="identity").fit(t.X, t.y, **kw(t))
    with pytest.raises(ValueError):
        ScaleProbitClassifier(prior=prior).fit(t.X[:, :2], t.y, **kw(t))
    with pytest.raises(ValueError):
        fit_source(t.X, t.y, **kw(t), prior=prior)


def test_prior_is_immutable(prior):
    with pytest.raises(ValueError):
        prior.theta[0] = 1.0


def test_transfer_cv_single_lam_equals_the_classifier(pair, prior):
    """With one candidate the search is a no-op: the final model equals the plain estimator bit for bit."""
    t = pair.target
    for pr in (None, prior):
        cv = ScaleProbitTransferCV(prior=pr, lams=(10.0,), include_endpoints=False).fit(t.X, t.y, **kw(t))
        ref = ScaleProbitClassifier(prior=pr, lam=10.0).fit(t.X, t.y, **kw(t))
        np.testing.assert_array_equal(cv.theta_, ref.theta_)
        assert cv.lam_ == 10.0
