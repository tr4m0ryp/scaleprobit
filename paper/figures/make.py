"""Paper figures, vector PDF, from the benchmark results and the full-season refit.

    python paper/figures/make.py [--results DIR] [--preds DIR]

fig1_paired.pdf       ours - rival in held-out log-loss per target size, 95% interval (+-1.96 se_total)
fig2_reliability.pdf  reliability diagram at the full target season, ECE in the legend
fig3_regime.pdf       synthetic regime map: ours - target-only and ours - source-only over shift x n
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Okabe-Ito, colourblind-safe
C = {"target_only_cv": "#0072B2", "source_only": "#D55E00", "pooled_indicator": "#009E73",
     "ours_centred_cv": "#000000", "closed_form": "#CC79A7"}
LABEL = {"target_only_cv": "target-only", "source_only": "source-only", "pooled_indicator": "pooled",
         "ours_centred_cv": "source-centred (ours)", "closed_form": "closed form"}
MARK = {"target_only_cv": "o", "source_only": "s", "pooled_indicator": "^"}
TITLE = {"football": "Football, era shift", "hoops": "Basketball, men's to women's league"}
SHORT = {"target_only_cv": "target-only", "source_only": "source-only", "ours_centred_cv": "ours"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})


def fig1(results: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=False)
    for ax, name in zip(axes, ("football", "hoops")):
        d = json.loads((results / f"{name}.json").read_text())["sizes"]
        keys = list(d)
        xs = np.arange(len(keys))
        labels = [str(d[k]["n_target_groups"]) + ("\n(all)" if k == "1000000" else "") for k in keys]
        for j, arm in enumerate(("target_only_cv", "source_only", "pooled_indicator")):
            m = np.array([d[k]["arms"][arm]["paired_ours_minus_arm"]["mean_diff"] for k in keys])
            se = np.array([d[k]["arms"][arm]["paired_ours_minus_arm"]["se_total"] for k in keys])
            ax.errorbar(xs + (j - 1) * 0.18, m, yerr=1.96 * se, fmt=MARK[arm], ms=3.5, lw=0.9, capsize=1.8,
                        color=C[arm], label=f"ours $-$ {LABEL[arm]}")
        ax.axhline(0, color="0.4", lw=0.6, ls="--")
        ax.set_xticks(xs, labels)
        ax.set_xlabel("target training games")
        ax.set_title(TITLE[name])
        ax.grid(axis="y", lw=0.3, color="0.85")
    axes[0].set_ylabel("$\\Delta$ log-loss (nats per row)\n< 0: ours better")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "fig1_paired.pdf")
    plt.close(fig)


def _reliability(y, p, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(idx, minlength=bins)
    py = np.bincount(idx, weights=y, minlength=bins)
    pp = np.bincount(idx, weights=p, minlength=bins)
    ok = n >= 30
    ece = float(np.sum(np.abs(py - pp)) / y.size)
    return pp[ok] / n[ok], py[ok] / n[ok], ece


def fig2(preds: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
    for ax, name in zip(axes, ("football", "hoops")):
        z = np.load(preds / f"preds_{name}_full.npz")
        y = z["y"].astype(float)
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls="--")
        for arm in ("target_only_cv", "source_only", "ours_centred_cv"):
            mp, my, ece = _reliability(y, z[arm])
            ax.plot(mp, my, marker=MARK.get(arm, "D"), ms=3, lw=0.9, color=C[arm],
                    label=f"{SHORT[arm]}: {ece:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("predicted P(home win)")
        ax.set_title(f"{name.replace('hoops', 'basketball').capitalize()}, full target season")
        ax.legend(loc="upper left", frameon=False, handlelength=1.5, title="ECE (15 bins)", title_fontsize=7)
        ax.grid(lw=0.3, color="0.88")
    axes[0].set_ylabel("observed home-win rate")
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "fig2_reliability.pdf")
    plt.close(fig)


def fig3(results: Path) -> None:
    d = json.loads((results / "synthetic.json").read_text())
    shifts, sizes = d["shifts"], d["sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    for ax, arm in zip(axes, ("target_only_cv", "source_only")):
        M = np.zeros((len(shifts), len(sizes)))
        S = np.zeros_like(M)
        for i, sh in enumerate(shifts):
            for j, n in enumerate(sizes):
                pr = d["cells"][f"shift={sh:g},n={n}"]["arms"][arm]["paired_ours_minus_arm"]
                M[i, j], S[i, j] = pr["mean_diff"], pr["se"]
        v = 0.05
        im = ax.imshow(np.clip(M, -v, v), cmap="PuOr_r", vmin=-v, vmax=v, origin="lower", aspect="auto")
        for i in range(len(shifts)):
            for j in range(len(sizes)):
                sig = abs(M[i, j]) > 1.96 * S[i, j]
                ax.text(j, i, f"{M[i, j] * 1000:+.1f}" + ("" if sig else "\n(tie)"), ha="center", va="center",
                        fontsize=6, color="white" if abs(M[i, j]) > 0.03 else "black",
                        fontweight="bold" if sig else "normal")
        ax.set_xticks(range(len(sizes)), [str(n) for n in sizes])
        ax.set_yticks(range(len(shifts)), [f"{s:g}" for s in shifts])
        ax.set_xlabel("target training rows")
        ax.set_title(f"ours $-$ {LABEL[arm]}")
    axes[0].set_ylabel("source-target shift")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("$\\Delta$ log-loss (nats); cell text in $10^{-3}$", fontsize=7)
    fig.savefig(HERE / "fig3_regime.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "bench" / "results")
    ap.add_argument("--preds", type=Path, default=REPO / "paper" / "results")
    a = ap.parse_args()
    fig1(a.results); fig2(a.preds); fig3(a.results)
    print("wrote", sorted(p.name for p in HERE.glob("*.pdf")))
