"""Write results/numbers.tex: one \\newcommand per number the paper may cite.

    python paper/results/make_numbers.py [--results DIR] [--preds DIR] [--out FILE]

--results  the benchmark result files (default: <repo>/bench/results, the public evidence)
--preds    per-row full-season predictions from refit_full.py (default: this directory)

Every macro is preceded by a % comment naming its file and JSON path. Two kinds of SE appear:
  *Se      = se_total, the benchmark's own reported SE (block bootstrap over test games of the per-row
             difference averaged over the 20 subsamples, plus the between-subsample variance / 20);
             this is the one the benchmark used for win / tie / loss.
  *SeBlock = se_block, the block-bootstrap part alone (test-game sampling only).
  *SeGame  = recomputed here from refit_full.py's per-row predictions at the full season: a paired
             block bootstrap over test games (2,000 draws, seed 0). One subsample only, so it should
             match se_block / se_total at "all" (where the between-subsample term is 0).
Macro names: n<Data><Size><Quantity><Arm>; Data Fb football, Hp basketball, Syn synthetic; Size A=25,
B=50, C=100, D=200, E=400, Full=all games; synthetic n A=50 B=100 C=200 D=400 E=800 and shift
Sa=0, Sb=0.1, Sc=0.3, Sd=0.6.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from numfmt import Out, f4, sgn4, f3, intc  # noqa: E402

ARMS = {"closed_form": "CF", "target_only_cv": "TO", "source_only": "SO", "pooled_indicator": "Pool",
        "ours_centred_cv": "Ours", "abl_ours_lam1": "LamOne", "abl_ours_lam100": "LamHund",
        "abl_ours_lam10000": "LamTenk", "abl_ours_t4_link": "Tfour",
        "abl_ours_target_standardisation": "Tstd", "logreg_target_only": "LRt",
        "logreg_pooled_indicator": "LRp", "lgbm_target_only": "GBt", "lgbm_source_only": "GBs",
        "lgbm_finetuned": "GBft"}
SIZE = {"25": "A", "50": "B", "100": "C", "200": "D", "400": "E", "1000000": "Full"}
SYN_N = {"50": "A", "100": "B", "200": "C", "400": "D", "800": "E"}
SYN_S = {"0": "Sa", "0.1": "Sb", "0.3": "Sc", "0.6": "Sd"}
VARS = {"var_dense_grid": "Dense", "var_slopes_anchored": "Anch", "var_one_se": "OneSe",
        "var_repeated_cv": "RepCv"}
METRICS = {"log_loss": "LL", "brier": "Brier", "ece": "ECE", "calibration_slope": "Slope"}
RIVALS = ("TO", "SO", "Pool")


def real(o: Out, path: Path, tag: str) -> dict:
    d = json.loads(path.read_text())
    fn = path.name
    o.c(f"{fn}: n_source_rows / n_source_groups / n_test_rows / n_test_groups")
    o.m(f"n{tag}SrcRows", intc(d["n_source_rows"])); o.m(f"n{tag}SrcGames", intc(d["n_source_groups"]))
    o.m(f"n{tag}TestRows", intc(d["n_test_rows"])); o.m(f"n{tag}TestGames", intc(d["n_test_groups"]))
    o.c(f"{fn}: meta.rows_per_game, wall_seconds"); o.m(f"n{tag}RowsPerGame", str(d["meta"]["rows_per_game"]))
    o.m(f"n{tag}WallS", intc(round(d["wall_seconds"])))
    o.c(f"{fn}: len(prior_theta) = number of model parameters (a, w_1..w_k, b)")
    o.m(f"n{tag}NParams", str(len(d["prior_theta"])))
    prereg_hits = 0
    for n, cell in d["sizes"].items():
        s = SIZE[n]
        o.c(f"{fn}: sizes.{n}.n_target_groups"); o.m(f"n{tag}{s}Games", intc(cell["n_target_groups"]))
        for arm, v in cell["arms"].items():
            a = ARMS[arm]
            for k, mk in METRICS.items():
                o.c(f"{fn}: sizes.{n}.arms.{arm}.{k} (mean over subsamples)")
                o.m(f"n{tag}{s}{mk}{a}", f4(v[k]))
            pr = v.get("paired_ours_minus_arm")
            if pr is None:
                continue
            o.c(f"{fn}: sizes.{n}.arms.{arm}.paired_ours_minus_arm.mean_diff / se_total / se_block / "
                f"upper95_total / p_diff_nonneg / subsample_wins (ours - arm, log-loss nats)")
            o.m(f"n{tag}{s}Diff{a}", sgn4(pr["mean_diff"])); o.m(f"n{tag}{s}Se{a}", f4(pr["se_total"]))
            o.m(f"n{tag}{s}SeBlock{a}", f4(pr["se_block"])); o.m(f"n{tag}{s}Up{a}", sgn4(pr["upper95_total"]))
            o.m(f"n{tag}{s}P{a}", f3(pr["p_diff_nonneg"])); o.m(f"n{tag}{s}Wins{a}", str(pr["subsample_wins"]))
        lam = np.asarray(cell["chosen_lam"]["ours_centred_cv"], float)
        o.c(f"{fn}: sizes.{n}.chosen_lam.ours_centred_cv: subsamples with lam = inf (source-only) / = 0 / count")
        o.m(f"n{tag}{s}LamInf", str(int(np.isinf(lam).sum()))); o.m(f"n{tag}{s}LamZero", str(int((lam == 0).sum())))
        o.m(f"n{tag}{s}NSub", str(lam.size))
        if n != "1000000" and int(n) <= 400:
            ok = all(cell["arms"][r]["paired_ours_minus_arm"]["upper95_total"] < 0
                     for r in ("target_only_cv", "source_only", "pooled_indicator"))
            prereg_hits += ok
    o.c(f"{fn}: sizes with n <= 400 where ours beats target-only, source-only and pooled (all upper95_total < 0)")
    o.m(f"n{tag}PreregHits", str(prereg_hits))
    return d


def variants(o: Out, path: Path, tag: str) -> None:
    d = json.loads(path.read_text())
    for n, cell in d["sizes"].items():
        s = SIZE[n]
        for arm, v in cell["arms"].items():
            if arm not in VARS:
                continue
            for ref, pr in v["paired_variant_minus_ref"].items():
                r = ARMS[ref]
                o.c(f"{path.name}: sizes.{n}.arms.{arm}.paired_variant_minus_ref.{ref}.mean_diff / se_total")
                o.m(f"n{tag}{s}{VARS[arm]}Minus{r}", sgn4(pr["mean_diff"]))
                o.m(f"n{tag}{s}{VARS[arm]}Minus{r}Se", f4(pr["se_total"]))
        if "var_one_se" in cell["chosen_lam"]:
            lam = np.asarray(cell["chosen_lam"]["var_one_se"], float)
            o.c(f"{path.name}: sizes.{n}.chosen_lam.var_one_se: subsamples with lam = inf")
            o.m(f"n{tag}{s}OneSeLamInf", str(int(np.isinf(lam).sum())))


def synthetic(o: Out, path: Path, one_se: Path) -> None:
    d = json.loads(path.read_text())
    fn = path.name
    o.c(f"{fn}: n_seeds, wall_seconds"); o.m("nSynSeeds", str(d["n_seeds"])); o.m("nSynWallS", intc(round(d["wall_seconds"])))
    wins = {r: [0, 0, 0] for r in RIVALS}       # win / tie / loss beyond 1.96 SE
    allthree = 0
    for key, cell in d["cells"].items():
        sh, n = key.replace("shift=", "").split(",n=")
        t = f"Syn{SYN_S[sh]}{SYN_N[n]}"
        beats = []
        for arm in ("target_only_cv", "source_only", "pooled_indicator", "closed_form"):
            pr = cell["arms"][arm]["paired_ours_minus_arm"]
            a = ARMS[arm]
            o.c(f"{fn}: cells.{key}.arms.{arm}.paired_ours_minus_arm.mean_diff / se (SE over 20 independent draws)")
            o.m(f"n{t}Diff{a}", sgn4(pr["mean_diff"])); o.m(f"n{t}Se{a}", f4(pr["se"]))
            if a in wins:
                hi, lo = pr["mean_diff"] + 1.96 * pr["se"], pr["mean_diff"] - 1.96 * pr["se"]
                wins[a][0 if hi < 0 else (2 if lo > 0 else 1)] += 1
                beats.append(hi < 0)
        allthree += all(beats)
        o.c(f"{fn}: cells.{key}.arms.ours_centred_cv.log_loss / oracle.log_loss")
        o.m(f"n{t}LLOurs", f4(cell["arms"]["ours_centred_cv"]["log_loss"]))
        o.m(f"n{t}LLOracle", f4(cell["arms"]["oracle"]["log_loss"]))
    o.c(f"{fn}: number of cells")
    o.m("nSynCells", str(len(d["cells"])))
    for a, (w, t_, l) in wins.items():
        o.c(f"{fn}: cells where ours vs {a} is a win / tie / loss (95% interval from se excludes 0 below / covers / above)")
        o.m(f"nSynWins{a}", str(w)); o.m(f"nSynTies{a}", str(t_)); o.m(f"nSynLoss{a}", str(l))
    o.c(f"{fn}: cells where ours beats target-only, source-only and pooled at once")
    o.m("nSynAllThree", str(allthree))
    e = json.loads(one_se.read_text())
    for key, cell in e["cells"].items():
        sh, n = key.replace("shift=", "").split(",n=")
        t = f"Syn{SYN_S[sh]}{SYN_N[n]}"
        pr = cell["one_se_minus_ours_centred_cv"]
        o.c(f"{one_se.name}: cells.{key}.one_se_minus_ours_centred_cv.mean_diff / se ({e['n_draws']} draws)")
        o.m(f"n{t}OneSeMinusOurs", sgn4(pr["mean_diff"])); o.m(f"n{t}OneSeMinusOursSe", f4(pr["se"]))


def per_game(o: Out, preds: Path, tag: str) -> None:
    """Paired per-game block-bootstrap SE and calibration from the full-season refit."""
    if not preds.exists():
        o.c(f"{preds.name} missing: run refit_full.py; per-game SEs not written")
        return
    from scaleprobit.evaluate import expected_calibration_error, paired_block_bootstrap
    z = np.load(preds)
    y, g = z["y"].astype(float), z["game"]
    ll = lambda p: -(y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1)))
    ours = ll(z["ours_centred_cv"])
    for arm in ("target_only_cv", "source_only", "pooled_indicator", "closed_form"):
        bb = paired_block_bootstrap(ours, ll(z[arm]), blocks=g, n_boot=2000, random_state=0)
        a = ARMS[arm]
        o.c(f"{preds.name}: paired block bootstrap over test games, ours - {arm} (recomputed here, 2000 draws)")
        o.m(f"n{tag}FullRefitDiff{a}", sgn4(bb["mean"])); o.m(f"n{tag}FullSeGame{a}", f4(bb["se"]))
        o.m(f"n{tag}FullCiLo{a}", sgn4(bb["ci_low"])); o.m(f"n{tag}FullCiHi{a}", sgn4(bb["ci_high"]))
    for arm in ("target_only_cv", "source_only", "ours_centred_cv", "pooled_indicator"):
        o.c(f"{preds.name}: ECE (package default bins) of {arm}, full-season refit")
        o.m(f"n{tag}FullRefitECE{ARMS[arm]}", f4(expected_calibration_error(y, z[arm])))
    o.c(f"{preds.name}: lam chosen by grouped CV at the full season (ours / target-only)")
    o.m(f"n{tag}FullLamOurs", "\\infty" if np.isinf(z["lam_ours"]) else f"{float(z['lam_ours']):g}")
    o.m(f"n{tag}FullLamTO", "\\infty" if np.isinf(z["lam_target"]) else f"{float(z['lam_target']):g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "bench" / "results")
    ap.add_argument("--preds", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=HERE / "numbers.tex")
    a = ap.parse_args()
    o = Out()
    o.c("Generated by results/make_numbers.py; do not edit by hand. Log-loss in nats; Diff = ours - arm.")
    o.c("Se = se_total (benchmark's reported SE, used for win/tie/loss); SeBlock = block part only;")
    o.c("SeGame = recomputed here from per-row full-season predictions (paired block bootstrap over games).")
    o.c("bench/common/stats.py: n_boot = 2000 block-bootstrap draws; subsamples per size = 20 (seeds 0-19)")
    o.m("nBootDraws", "2{,}000"); o.m("nSubsamples", "20")
    for name, tag in (("football", "Fb"), ("hoops", "Hp")):
        real(o, a.results / f"{name}.json", tag)
        variants(o, a.results / f"{name}_variants.json", tag)
        per_game(o, a.preds / f"preds_{name}_full.npz", tag)
    synthetic(o, a.results / "synthetic.json", a.results / "synthetic_one_se.json")
    priv = (a.results / "PRIVATE_LINE.txt").read_text().strip()
    o.c("PRIVATE_LINE.txt: the one permitted sentence about the private application, verbatim")
    o.m("privateline", priv.replace("%", "\\%").replace("->", "$\\to$").replace("~", "$\\sim$"))
    a.out.write_text(o.text())
    print(f"wrote {a.out} ({o.count} macros)")


if __name__ == "__main__":
    main()
