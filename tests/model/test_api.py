"""scikit-learn integration: clone, pickle, column inputs, row-wise prediction, CV utilities."""
import pickle

import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from scaleprobit import ScaleProbitClassifier

from ..conftest import kw


def test_clone_and_params():
    m = ScaleProbitClassifier(lam=3.0, link="t", df=5)
    c = clone(m)
    assert c.get_params() == m.get_params()


def test_pickle_round_trip(data):
    m = ScaleProbitClassifier().fit(data.X, data.y, **kw(data))
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m.predict_proba(data.X, **kw(data)), m2.predict_proba(data.X, **kw(data)))


def test_prediction_is_row_wise(data):
    """No batch statistic enters a prediction: each row alone gives the same probability."""
    m = ScaleProbitClassifier().fit(data.X, data.y, **kw(data))
    full = m.predict_proba(data.X[:20], **kw(data, slice(0, 20)))[:, 1]
    single = [m.predict_proba(data.X[i:i + 1], **kw(data, slice(i, i + 1)))[0, 1] for i in range(20)]
    np.testing.assert_array_equal(full, single)


def _stacked(data):
    return np.column_stack([data.X, data.distance, data.horizon, data.base_scale])


def test_column_inputs_equal_keyword_inputs(data):
    a = ScaleProbitClassifier().fit(data.X, data.y, **kw(data))
    b = ScaleProbitClassifier(distance_col=3, horizon_col=4, base_scale_col=-1).fit(_stacked(data), data.y)
    np.testing.assert_array_equal(a.theta_, b.theta_)
    np.testing.assert_array_equal(a.predict_proba(data.X, **kw(data)), b.predict_proba(_stacked(data)))


def test_dataframe_columns_by_name(data):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_stacked(data), columns=["f1", "f2", "f3", "dist", "hor", "s"])
    a = ScaleProbitClassifier().fit(data.X, data.y, **kw(data))
    b = ScaleProbitClassifier(distance_col="dist", horizon_col="hor", base_scale_col="s",
                              lam_groups={"f3": 1.0}).fit(df, data.y)
    np.testing.assert_array_equal(a.theta_, b.theta_)
    assert list(b.feature_names_in_) == ["f1", "f2", "f3"]


def test_cross_val_score_with_column_inputs(data):
    m = ScaleProbitClassifier(distance_col=3, horizon_col=4, base_scale_col=5)
    s = cross_val_score(m, _stacked(data), data.y, cv=3, scoring="neg_log_loss")
    assert s.shape == (3,) and np.all(np.isfinite(s))


def test_metadata_routing_of_structural_arrays(data):
    from sklearn.model_selection import cross_validate
    with sklearn.config_context(enable_metadata_routing=True):
        m = ScaleProbitClassifier().set_fit_request(distance=True, horizon=True, base_scale=True)
        r = cross_validate(m, data.X, data.y, cv=3, params=kw(data), return_estimator=True,
                           return_indices=True, scoring=lambda est, X, y: 0.0)
    tr = r["indices"]["train"][0]
    ref = ScaleProbitClassifier().fit(data.X[tr], data.y[tr], **kw(data, tr))
    np.testing.assert_array_equal(r["estimator"][0].theta_, ref.theta_)


def test_errors():
    m = ScaleProbitClassifier()
    with pytest.raises(ValueError):
        m.fit(np.ones((5, 1)), np.ones(5), distance=np.ones(5), horizon=np.ones(5))
    with pytest.raises(ValueError):
        m.fit(np.ones((4, 1)), [0, 1, 0, 1])
    with pytest.raises(ValueError):
        ScaleProbitClassifier(link="logit").fit(np.ones((4, 1)), [0, 1, 0, 1], distance=np.ones(4),
                                                horizon=np.ones(4))
