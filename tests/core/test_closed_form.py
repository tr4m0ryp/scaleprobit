"""Reduction remarks: theta = 0 is the closed form Phi(d / (s sqrt h)); the t link tends to probit."""
import numpy as np
from scipy.special import ndtr

from scaleprobit import ScaleProbitClassifier
from scaleprobit.core import EPS

from ..conftest import kw


def test_zero_theta_is_the_closed_form(data):
    m = ScaleProbitClassifier(max_iter=0).fit(data.X, data.y, **kw(data))
    assert np.all(m.theta_ == 0.0)
    p = m.predict_proba(data.X, **kw(data))[:, 1]
    ref = np.clip(ndtr(data.distance / (data.base_scale * np.sqrt(data.horizon))), EPS, 1 - EPS)
    np.testing.assert_allclose(p, ref, rtol=1e-13, atol=0)  # equal up to float rounding


def test_infinite_lam_without_prior_fits_only_the_scale_intercept(data):
    m = ScaleProbitClassifier(lam=np.inf).fit(data.X, data.y, **kw(data))
    assert np.all(m.coef_ == 0.0) and m.bias_ == 0.0
    assert m.scale_intercept_ != 0.0


def test_t_link_tends_to_probit(data):
    a = ScaleProbitClassifier(lam=1.0).fit(data.X, data.y, **kw(data))
    b = ScaleProbitClassifier(link="t", df=1e7, lam=1.0).fit(data.X, data.y, **kw(data))
    np.testing.assert_allclose(a.theta_, b.theta_, atol=1e-4)


def test_t_link_has_heavier_tails():
    from scaleprobit.core import cdf
    z = np.array([-4.0, -2.0, 2.0, 4.0])
    pt, pn = cdf(z, 4.0), cdf(z, None)
    assert np.all(pt[:2] > pn[:2]) and np.all(pt[2:] < pn[2:])
