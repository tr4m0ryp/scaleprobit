"""Quickstart: transfer a learned scale from a large source sample to a small target sample.

Synthetic data with known truth: 20,000 source rows, 300 target rows and 10,000 target test rows;
the target's true parameters differ from the source's by a vector of length 0.1. Five independent
draws are averaged, because a single 300-row target sample is noisy enough to mislead any choice
made on it.

Run:  python examples/quickstart.py
"""
import numpy as np

from scaleprobit import ScaleProbitClassifier, ScaleProbitTransferCV, fit_source
from scaleprobit.datasets import make_transfer_pair
from scaleprobit.evaluate import expected_calibration_error, log_loss_rows


def structural(b):
    return {"distance": b.distance, "horizon": b.horizon}


def arms(prior, tgt):
    kw = structural(tgt)
    return {
        "closed form (theta = 0)": ScaleProbitClassifier(max_iter=0).fit(tgt.X, tgt.y, **kw),
        "target only (CV ridge at 0)": ScaleProbitTransferCV().fit(tgt.X, tgt.y, **kw),
        "source only (lam = inf)": ScaleProbitClassifier(prior=prior, lam=np.inf).fit(tgt.X, tgt.y, **kw),
        "source-centred ridge (CV)": ScaleProbitTransferCV(prior=prior).fit(tgt.X, tgt.y, **kw),
    }


def main(n_draws=5):
    loss, ece, chosen = {}, {}, []
    for seed in range(n_draws):
        data = make_transfer_pair(n_source=20000, n_target=300, n_test=10000, shift=0.1, n_features=4,
                                  random_state=seed)
        src, tgt, test = data.source, data.target, data.test
        prior = fit_source(src.X, src.y, **structural(src), lam=1.0)
        fitted = arms(prior, tgt)
        ours = fitted["source-centred ridge (CV)"]
        chosen.append(ours.lam_)
        if seed == 0:
            print("lambda path on the first draw (out-of-fold log-loss on the 300 target rows):")
            r = ours.cv_results_
            for lam, role, ll in zip(r["lam"], r["role"], r["mean_test_log_loss"]):
                print(f"  lam {lam:>8g}  {ll:.4f}  {role}")
            print("fitted scale, first five test rows:", np.round(ours.scale(test.X[:5]), 3))
            print("true scale,   first five test rows:", np.round(test.sigma[:5], 3), "\n")
        for name, model in fitted.items():
            p = model.predict_proba(test.X, **structural(test))[:, 1]
            loss.setdefault(name, []).append(log_loss_rows(test.y, p).mean())
            ece.setdefault(name, []).append(expected_calibration_error(test.y, p))
        loss.setdefault("true probabilities", []).append(log_loss_rows(test.y, test.proba).mean())
        ece.setdefault("true probabilities", []).append(expected_calibration_error(test.y, test.proba))

    print(f"test log-loss and ECE, mean over {n_draws} draws (chosen lam per draw: {chosen})")
    for name in loss:
        print(f"  {name:30s} {np.mean(loss[name]):.4f}  {np.mean(ece[name]):.4f}")


if __name__ == "__main__":
    main()
