# Benchmark scope, cuts and outcomes

Budget: <= 2.5 h wall on 2 threads, CPU only, on a shared 8-core machine. Measured wall:
football 862 s, hoops 1,396 s, synthetic 2,633 s, variants 721 + 890 + 262 s; total **6,764 s (1.88 h)**.

Reproduce (data are cached in `$SCALEPROBIT_DATA`, default `~/projects/oss/_data`, never in the repo):

```
python -m bench.run_football
python -m bench.run_hoops
python -m bench.run_synthetic
python -m bench.variants.run football
python -m bench.variants.run hoops
python -m bench.variants.synthetic_one_se
python -m bench.summarize
```

## What ran

| file | task | source | target pool | test |
|---|---|---|---|---|
| `football.json` | American football in-game home win, era shift | seasons 2009-2017, 2,398 games, 47,960 rows | season 2021, 284 games | seasons 2022-2024, 852 games, 17,040 rows |
| `hoops.json` | basketball in-game home win, men's league -> women's league | 2019-2024, 7,565 games, 151,300 rows | 2019-2022, 817 games | 2023-2025, 838 games, 16,760 rows |
| `synthetic.json` | `make_transfer_pair`, known truth, 4 features | 20,000 rows per draw | n rows | 5,000 rows per draw |

Rows: 20 random game states per game (fixed seed). Target sizes (games): football 25 / 50 / 100 / 200 / all;
hoops 25 / 50 / 100 / 200 / 400 / all; synthetic rows 50 / 100 / 200 / 400 / 800 x shift 0 / 0.1 / 0.3 / 0.6.
**20 random target subsamples per size** (seeds 0-19; one fit at "all", where every subsample is the same).

**Paired SE.** Real data: rows of one game are dependent, so the SE of ours - arm is a block bootstrap
over test games (2,000 draws) of the per-row difference averaged over the 20 subsamples (`se_block`);
`se_total` adds the between-subsample variance (sqrt(se_block^2 + sd_between^2 / 20)) and is the one
used for win / tie / loss. Synthetic: draws are independent, so the SE is the plain standard error of
the 20 per-draw mean differences.

## Cuts (none silent)

- **Third real dataset cut.** The plan listed an optional third public dataset; dropped for budget.
- **Basketball without a pregame line.** The target league's files carry a placeholder spread, so no
  Stern drift term is used on either side; the bias absorbs the home edge.
- **Football source shortened** to 2009-2017 (spec: 1999-2017) to bound download and fit time; the
  test is 2022-2024 as in the spec.
- **R arms cut:** `glmx::hetglm` (arm 6) and `glmtrans` (arm 9) need R, which is not installed. Our
  plain variant's agreement with a textbook likelihood is covered by the package tests instead.
- **Ablations cut:** Platt on / off; a free (unpenalised) scale intercept; `lam_x` on groups (no grouped
  features in these tasks). Kept: lam fixed at 1 / 100 / 1e4, t(4) link, target standardisation.
- **Sizes and subsamples:** 800 target rows / 400 games is the top of the small-n curve (spec: up to
  800 and "all"); 20 subsamples as in the spec, but the synthetic map uses 20 draws per cell, not a
  finer shift grid.
- **Gradient boosting** uses one fixed parameter set (15 leaves, rate 0.05); only its rounds are chosen
  by grouped CV. The fine-tuned arm continues the source booster on the target (rounds by CV).
- **Pooled arms** concatenate source and target with a domain indicator in the scale (scaleprobit) or
  as a column (logistic regression); no target up-weighting was tuned.
- Per-row predictions are not stored (they would exceed 5 MB); the JSON files hold every metric, the
  per-subsample log-loss and the chosen lam of every fit.

## Results (log-loss, ours = source-centred ridge, lam by grouped 5-fold CV over 0, 1, 10, ..., 1e4, inf)

**Against the plain baseline (target-only, ridge at 0, CV):**
- **football:** ours wins beyond se_total at every size from 50 games: 50 games -0.0265 (0.0077),
  100 -0.0116 (0.0024), 200 -0.0098 (0.0022), all 284 games -0.0056 (0.0014). At 25 games -0.0218 (0.0127), a tie.
- **hoops:** ties at every size (-0.0065 (0.0051) at 25 games down to -0.0004 (0.0006) at all 817 games).
  The target league is close to Brownian (the closed form alone scores 0.4732 against 0.4715 for ours) and
  20 states per game give target-only enough rows from 50 games on.
- **synthetic:** ours wins beyond SE in 15 of 20 cells (e.g. shift 0.1, 200 rows: -0.0158 (0.0044)); ties at
  shift 0.3 with 400 rows and at shift 0.6 with n >= 100 rows, where the target needs its own fit and target-only catches up.

**Against source-only (lam = inf) and pooling, the pre-registered rivals: not beaten on real data.**
- **football:** source-only beats ours at 100 games and above (+0.0086 (0.0036) at 100, +0.0058 (0.0025)
  at all 284); pooled beats ours at 100 and above too. CV on one season picks a finite lam that fits
  season-specific deviations which do not carry to the next three seasons.
- **hoops:** ties with source-only and pooling from 50 games; at 25 games ours loses to source-only by
  +0.0324 (0.0145): CV on 25 games picked lam = 0 or 10 in some subsamples.
- **gradient boosting fitted on the source only** has the lowest log-loss on both real tasks (football
  0.4628, hoops 0.4699); ours never beats it beyond SE and loses to it at hoops 50 games.
- **synthetic regime map:** ours beats source-only beyond SE at shift 0.3 with n >= 400 rows and at shift
  0.6 with n >= 200 rows (up to -0.0489 (0.0028)); loses at shift 0.1 with 50 / 100 rows (+0.0050 (0.0023),
  +0.0148 (0.0068)) and, by under 0.001, at shift 0 with 400 / 800 rows; ties elsewhere.
  This is the spec's section 6 picture: the endpoint that wins depends on the shift, and grouped CV on a
  small target chooses between them noisily.

**Pre-registered success (all three paired upper 95 % bounds < 0 vs target-only, source-only and pooled at
n <= 400 on football or hoops): not met.**

## Variants tried ("what would have to be true", spec section 6)

Run on the same target subsamples as the main bench (n = 25 / 50 / 100 games, 20 subsamples), paired
against the default; numbers are variant - reference, se_total in brackets.

1. **Penalty sweep, 13-point grid 1..1e6** (true if the default grid's decade jumps cost accuracy).
   Ties the default everywhere (football +0.0003 (0.0093) / +0.0011 (0.0018) / -0.0019 (0.0018); hoops
   +0.0057 / -0.0004 / +0.0007). **Not kept.**
2. **Slopes anchored, 10,000x stronger pull on w** (true if the shift is a level shift in a and b only).
   Worse at 25 games (football +0.0214 (0.0170), hoops -0.0078 (0.0068)), ties after. **Not kept.**
3. **One-standard-error rule: largest lam within 1 SE of the best CV loss** (true if CV on a few dozen
   groups picks too small a lam by noise). On both real tasks it picks lam = inf in every subsample and so
   equals source-only: beats the default at football 50 games -0.0053 (0.0027) and 100 games -0.0086 (0.0036),
   hoops 25 games -0.0324 (0.0145); ties elsewhere. On the synthetic map (`synthetic_one_se.json`) it
   **loses** where adaptation is real: shift 0.3 n = 800 +0.0077 (0.0014), shift 0.6 n = 50 / 200
   +0.0128 (0.0051) / +0.0138 (0.0065). **Not adopted as the default**; it is a setting for targets of
   about 100 groups or fewer when the shift is believed small, and then it is source-only by another name.
4. **Repeated grouped CV, 3 shuffled 5-fold splits averaged** (true if the selection noise is
   fold-assignment noise). Helps at 25 games (football -0.0183 (0.0134), hoops -0.0120 (0.0094), ties),
   still loses to source-only (football +0.0051 (0.0023) at 25 games, hoops +0.0204 (0.0082)). **Not kept.**

## Honest reading

The source-centred ridge reliably beats the plain target-only fit (football, synthetic) or ties it
(hoops), and its lambda path contains the right endpoint. It does not beat the best endpoint on
these public tasks: with small targets, grouped CV cannot tell a real shift from noise, and on the
football era shift the target season itself is a poor guide to later seasons. The private result's
own conclusion (lam* effectively inf, the gain came from the source fit) is reproduced, not surpassed.
