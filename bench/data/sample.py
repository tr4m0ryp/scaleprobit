"""Turn per-play game states into benchmark rows: a few random states per game, one label per game."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.split import Split

ROWS_PER_GAME = 20


def sample_states(df: pd.DataFrame, rows_per_game: int = ROWS_PER_GAME, seed: int = 0) -> pd.DataFrame:
    """Draw up to ``rows_per_game`` distinct states per game (fixed seed)."""
    rng = np.random.default_rng(seed)
    key = rng.random(len(df))
    df = df.assign(_k=key).sort_values(["game", "_k"])
    return df.groupby("game", sort=False).head(rows_per_game).drop(columns="_k").reset_index(drop=True)


def to_split(df: pd.DataFrame, feature_cols, flex_cols, base_scale: float) -> Split:
    """``df`` needs ``game``, ``y``, ``distance``, ``horizon`` and the named columns."""
    n = len(df)
    return Split(X=df[list(feature_cols)].to_numpy(float), y=df["y"].to_numpy(int),
                 distance=df["distance"].to_numpy(float), horizon=df["horizon"].to_numpy(float),
                 base_scale=np.full(n, base_scale), groups=df["game"].to_numpy(),
                 flex=df[list(flex_cols)].to_numpy(float))
