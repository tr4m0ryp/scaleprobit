"""Professional American football, an era shift: older seasons as the source, one recent season as the
target, the following seasons as the test.

Rows are game states; y = the home team wins; horizon = seconds of regulation left; distance = the
home margin plus the pregame point spread scaled by the share of the game left (Stern's drift
term), so a favourite's expected remaining edge counts as distance. Overtime states and ties are
dropped. Scale features: the pregame over/under total (a dispersion anchor set before the game),
horizon + 30 s, 1 + |margin| and 1 + the timeouts left on both sides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fetch import fetch
from .sample import sample_states, to_split

FEATURES = ("total_line", "h_plus", "abs_margin", "timeouts")
FLEX = ("distance", "horizon", "margin", "spread_line", "total_line", "abs_margin", "timeouts", "u")
COLS = ["game_id", "season_type", "qtr", "game_seconds_remaining", "total_home_score", "total_away_score",
        "result", "spread_line", "total_line", "home_timeouts_remaining", "away_timeouts_remaining"]


def load(seasons) -> pd.DataFrame:
    frames = []
    for y in seasons:
        d = pd.read_parquet(fetch("nfl_pbp", y), columns=COLS)
        frames.append(d.assign(season=y))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["game_seconds_remaining", "total_home_score", "total_away_score", "result",
                         "spread_line", "total_line"])
    h = d.game_seconds_remaining.astype(float)
    d = d[(d.qtr <= 4) & (d.result != 0) & (h >= 5) & (h <= 3540)].copy()
    h = d.game_seconds_remaining.astype(float)
    m = (d.total_home_score - d.total_away_score).astype(float)
    dist = m + d.spread_line.astype(float) * h / 3600.0
    to = d.home_timeouts_remaining.fillna(0) + d.away_timeouts_remaining.fillna(0)
    d = d.assign(game=d.game_id.astype(str), y=(d.result > 0).astype(int), distance=dist, horizon=h, margin=m,
                 h_plus=h + 30.0, abs_margin=1.0 + m.abs(), timeouts=1.0 + to.astype(float),
                 total_line=d.total_line.astype(float), u=dist / np.sqrt(h))
    return d


def build(source_seasons=range(2009, 2018), target_season=2021, test_seasons=range(2022, 2025)):
    """Return (source, target_pool, test, meta)."""
    full = load(source_seasons)
    src = sample_states(full)
    pool = sample_states(load([target_season]), seed=1)
    test = sample_states(load(test_seasons), seed=2)
    g = full.groupby("game").agg(r=("result", "first"), s=("spread_line", "first"))
    base_scale = float(np.std(g.r - g.s) / np.sqrt(3600.0))   # closed-form scale, from the source only
    meta = {"source": f"seasons {source_seasons.start}-{source_seasons.stop - 1} (nflverse play-by-play)",
            "target_pool": f"season {target_season}", "test": f"seasons {test_seasons.start}-{test_seasons.stop - 1}",
            "rows_per_game": 20, "features": list(FEATURES), "flex": list(FLEX), "base_scale": base_scale}
    mk = lambda d: to_split(d, FEATURES, FLEX, base_scale)
    return mk(src), mk(pool), mk(test), meta
