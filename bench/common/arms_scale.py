"""Arms built on scaleprobit: the closed form, target-only, source-only, pooled, ours and ablations."""
from __future__ import annotations

import numpy as np

from scaleprobit import Prior, ScaleProbitClassifier, ScaleProbitTransferCV, fit_source

from .split import Split

FIXED_LAMS = (1.0, 100.0, 1e4)


class SourceFits:
    """Everything fitted on the source once per run (never sees a target row)."""

    def __init__(self, src: Split, source_lam: float = 1.0):
        self.src = src
        self.prior = fit_source(src.X, src.y, **src.structural(), lam=source_lam)
        self.prior_t4 = fit_source(src.X, src.y, **src.structural(), lam=source_lam, link="t", df=4)


def _p(model, test: Split) -> np.ndarray:
    return model.predict_proba(test.X, **test.structural())[:, 1]


def _with_indicator(s: Split, value: float) -> np.ndarray:
    return np.column_stack([s.X, np.full(len(s), value)])


def scale_arms(fits: SourceFits, tgt: Split, test: Split, ablations: bool = True) -> tuple[dict, dict]:
    """Return ``({arm: p_test}, {arm: chosen lam})`` for one target subsample."""
    kw, g = tgt.structural(), tgt.groups
    out, lam = {}, {}

    out["closed_form"] = _p(ScaleProbitClassifier(max_iter=0).fit(tgt.X, tgt.y, **kw), test)

    m = ScaleProbitTransferCV().fit(tgt.X, tgt.y, **kw, groups=g)
    out["target_only_cv"], lam["target_only_cv"] = _p(m, test), m.lam_

    out["source_only"] = _p(ScaleProbitClassifier(prior=fits.prior, lam=np.inf).fit(tgt.X, tgt.y, **kw), test)

    ours = ScaleProbitTransferCV(prior=fits.prior).fit(tgt.X, tgt.y, **kw, groups=g)
    out["ours_centred_cv"], lam["ours_centred_cv"] = _p(ours, test), ours.lam_

    src = fits.src
    pool = Split.concat([src, tgt])
    Xp = np.concatenate([_with_indicator(src, 1.0), _with_indicator(tgt, np.e)])
    pm = ScaleProbitClassifier(lam=1.0).fit(Xp, pool.y, **pool.structural())
    out["pooled_indicator"] = pm.predict_proba(_with_indicator(test, np.e), **test.structural())[:, 1]

    if not ablations:
        return out, lam
    for v in FIXED_LAMS:
        out[f"abl_ours_lam{v:g}"] = _p(ScaleProbitClassifier(prior=fits.prior, lam=v).fit(tgt.X, tgt.y, **kw), test)
    t4 = ScaleProbitTransferCV(prior=fits.prior_t4, link="t", df=4).fit(tgt.X, tgt.y, **kw, groups=g)
    out["abl_ours_t4_link"], lam["abl_ours_t4_link"] = _p(t4, test), t4.lam_
    tstd = ScaleProbitClassifier(lam=0.0, max_iter=0).fit(tgt.X, tgt.y, **kw)      # target standardisation
    pr = fits.prior
    alt = Prior(theta=pr.theta, mean=tstd.mean_, scale_std=tstd.scale_std_, link=pr.link, df=pr.df,
                transform=pr.transform)
    ts = ScaleProbitTransferCV(prior=alt).fit(tgt.X, tgt.y, **kw, groups=g)
    out["abl_ours_target_standardisation"], lam["abl_ours_target_standardisation"] = _p(ts, test), ts.lam_
    return out, lam
