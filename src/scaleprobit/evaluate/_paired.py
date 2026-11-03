"""Paired comparison of two models' per-row losses with a block bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state


def paired_block_bootstrap(loss_a, loss_b, blocks=None, n_boot=2000, random_state=None) -> dict:
    """Mean of ``loss_a - loss_b`` and its uncertainty, resampling whole blocks.

    Rows in the same block (a game, a month, a patient) are dependent, so the block is the unit of
    resampling. Each block's summed difference and row count are drawn together with replacement.

    Returns ``{"mean", "se", "ci_low", "ci_high", "p_nonneg", "n_blocks"}`` where ``p_nonneg`` is the
    bootstrap share of draws with a mean difference ``>= 0`` (A no better than B) and the interval is
    the 2.5 / 97.5 percentile one.
    """
    diff = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    blocks = np.arange(diff.size) if blocks is None else np.asarray(blocks)
    _, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    rng = check_random_state(random_state)
    draw = rng.randint(0, sums.size, size=(n_boot, sums.size))
    means = sums[draw].sum(1) / counts[draw].sum(1)
    return {"mean": float(diff.mean()), "se": float(means.std(ddof=1)),
            "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)),
            "p_nonneg": float(np.mean(means >= 0)), "n_blocks": int(sums.size)}
