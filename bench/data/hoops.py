"""Professional basketball: a men's league as the source, a women's league as the target.

Rows are game states; y = the home team wins; distance = the home margin; horizon = seconds of
regulation left (the women's game is 40 minutes, the men's 48). Overtime states are dropped; the label
still counts an overtime result. No pregame line is used (it is missing for the target league).
Scale features (positive, log-transformed by the model): horizon + 30 s, the scoring rate so far
(shrunk towards a typical pace), and 1 + |margin|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fetch import fetch
from .sample import sample_states, to_split

FEATURES = ("h_plus", "pace", "abs_margin")
FLEX = ("distance", "horizon", "pace", "abs_margin", "u")
COLS = ["game_id", "season_type", "period_number", "home_score", "away_score", "end_game_seconds_remaining",
        "game_play_number"]


def load_league(source: str, seasons, length_s: float) -> pd.DataFrame:
    frames = []
    for y in seasons:
        d = pd.read_parquet(fetch(source, y), columns=COLS)
        d = d[d.season_type.isin([2, 3])].sort_values(["game_id", "game_play_number"])
        fin = d.groupby("game_id").agg(fh=("home_score", "last"), fa=("away_score", "last"))
        fin = fin[fin.fh != fin.fa]
        d = d[(d.period_number <= 4) & d.game_id.isin(fin.index)].join(fin, on="game_id")
        d = d.assign(game=f"{source}-{y}-" + d.game_id.astype(str), season=y)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    h = d.end_game_seconds_remaining.astype(float)
    d = d[(h >= 5) & (h <= length_s - 60)].copy()
    h = d.end_game_seconds_remaining.astype(float)
    m = (d.home_score - d.away_score).astype(float)
    elapsed = length_s - h
    pts = (d.home_score + d.away_score).astype(float)
    d = d.assign(y=(d.fh > d.fa).astype(int), distance=m, horizon=h, h_plus=h + 30.0,
                 pace=(pts + 4.0 * 10) / (elapsed / 60.0 + 10), abs_margin=1.0 + m.abs(), u=m / np.sqrt(h))
    return d


def build(base_scale: float | None = None):
    """Return (source, target_pool, test, meta)."""
    src = sample_states(load_league("nba_espn", range(2019, 2025), 2880.0))
    wn = load_league("wnba_espn", range(2019, 2026), 2400.0)
    pool = sample_states(wn[wn.season <= 2022], seed=1)
    test = sample_states(wn[wn.season >= 2023], seed=2)
    if base_scale is None:      # closed-form scale per sqrt-second, from the source's final margins only
        g = src.groupby("game").agg(fh=("fh", "first"), fa=("fa", "first"))
        base_scale = float(np.std(g.fh - g.fa) / np.sqrt(2880.0))
    meta = {"source": "men's league seasons 2019-2024 (ESPN play-by-play via sportsdataverse)",
            "target_pool": "women's league seasons 2019-2022", "test": "women's league seasons 2023-2025",
            "rows_per_game": 20, "features": list(FEATURES), "flex": list(FLEX), "base_scale": base_scale}
    mk = lambda d: to_split(d, FEATURES, FLEX, base_scale)
    return mk(src), mk(pool), mk(test), meta
