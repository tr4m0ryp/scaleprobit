"""Lambda search: endpoints as reference rows, out-of-fold honesty, groups, the choice rule."""
import numpy as np
import pytest

from scaleprobit import ScaleProbitClassifier, ScaleProbitTransferCV, fit_source
from scaleprobit.evaluate import log_loss_rows
from scaleprobit.transfer import make_folds

from ..conftest import kw


@pytest.fixture(scope="module")
def prior(pair):
    return fit_source(pair.source.X, pair.source.y, **kw(pair.source))


def test_endpoints_are_reference_rows(pair, prior):
    t = pair.target
    m = ScaleProbitTransferCV(prior=prior, lams=(1.0, 100.0)).fit(t.X, t.y, **kw(t))
    r = m.cv_results_
    np.testing.assert_array_equal(r["lam"], [0.0, 1.0, 100.0, np.inf])
    assert list(r["role"]) == ["target-only", "", "", "source-only"]
    assert r["rank_test_log_loss"][list(r["lam"]).index(m.lam_)] == 1
    assert m.best_score_ == r["mean_test_log_loss"].min()


def test_oof_rows_never_see_their_own_fold(pair, prior):
    t = pair.target
    folds = make_folds(5, len(t.y), None)
    base = ScaleProbitTransferCV(prior=prior, lams=(10.0,)).fit(t.X, t.y, **kw(t))
    va = folds[2][1]
    y2 = t.y.copy()
    y2[va] = 1 - y2[va]
    flip = ScaleProbitTransferCV(prior=prior, lams=(10.0,)).fit(t.X, y2, **kw(t))
    np.testing.assert_array_equal(base.cv_oof_proba_[:, va], flip.cv_oof_proba_[:, va])
    rest = np.setdiff1d(np.arange(len(t.y)), va)
    assert not np.array_equal(base.cv_oof_proba_[1, rest], flip.cv_oof_proba_[1, rest])


def test_plain_variant_standardises_on_the_training_fold_only(pair):
    """Changing one validation row's features changes only that row's out-of-fold prediction."""
    t = pair.target
    folds = make_folds(5, len(t.y), None)
    va = folds[1][1]
    X2 = t.X.copy()
    X2[va[0]] *= 50.0
    a = ScaleProbitTransferCV(lams=(1.0,)).fit(t.X, t.y, **kw(t))
    b = ScaleProbitTransferCV(lams=(1.0,)).fit(X2, t.y, **kw(t))
    np.testing.assert_array_equal(a.cv_oof_proba_[:, va[1:]], b.cv_oof_proba_[:, va[1:]])


def test_oof_equals_manual_fold_fits(pair, prior):
    t = pair.target
    m = ScaleProbitTransferCV(prior=prior, lams=(30.0,), include_endpoints=False, cv=4).fit(t.X, t.y, **kw(t))
    for tr, va in make_folds(4, len(t.y), None):
        f = ScaleProbitClassifier(prior=prior, lam=30.0).fit(t.X[tr], t.y[tr], **kw(t, tr))
        np.testing.assert_array_equal(m.cv_oof_proba_[0, va], f.predict_proba(t.X[va], **kw(t, va))[:, 1])


def test_groups_are_never_split(pair, prior):
    t = pair.target
    groups = np.arange(len(t.y)) // 10
    for tr, va in make_folds(5, len(t.y), groups):
        assert not set(groups[tr]) & set(groups[va])
    m = ScaleProbitTransferCV(prior=prior, lams=(10.0,)).fit(t.X, t.y, groups=groups, **kw(t))
    assert len([k for k in m.cv_results_ if k.startswith("split")]) == 5


def test_final_model_is_the_classifier_at_the_chosen_lam(pair, prior):
    t = pair.target
    m = ScaleProbitTransferCV(prior=prior).fit(t.X, t.y, **kw(t))
    ref = ScaleProbitClassifier(prior=prior, lam=m.lam_).fit(t.X, t.y, **kw(t))
    np.testing.assert_array_equal(m.theta_, ref.theta_)
    np.testing.assert_array_equal(m.predict_proba(t.X, **kw(t)), ref.predict_proba(t.X, **kw(t)))


def test_ties_go_to_the_larger_lam(pair, prior):
    t = pair.target
    m = ScaleProbitTransferCV(prior=prior, lams=(1e12,)).fit(t.X, t.y, **kw(t))
    r = m.cv_results_
    i12, iinf = list(r["lam"]).index(1e12), list(r["lam"]).index(np.inf)
    assert abs(r["mean_test_log_loss"][i12] - r["mean_test_log_loss"][iinf]) < 1e-12
    if m.lam_ in (1e12, np.inf):
        assert m.lam_ == np.inf


def test_identical_tasks_choose_strong_shrinkage_and_beat_target_only():
    from scaleprobit.datasets import make_transfer_pair
    gains = []
    for seed in range(4):
        p = make_transfer_pair(n_source=20000, n_target=150, n_test=4000, shift=0.0, n_features=4, random_state=seed)
        pr = fit_source(p.source.X, p.source.y, **kw(p.source))
        m = ScaleProbitTransferCV(prior=pr).fit(p.target.X, p.target.y, **kw(p.target))
        plain = ScaleProbitTransferCV().fit(p.target.X, p.target.y, **kw(p.target))
        te = p.test
        ll = lambda est: log_loss_rows(te.y, est.predict_proba(te.X, **kw(te))[:, 1]).mean()
        gains.append(ll(plain) - ll(m))
        assert m.lam_ > 0.0                    # never the unpenalised target fit
    assert np.mean(gains) > 0
