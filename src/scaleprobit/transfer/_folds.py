"""Fold construction for the lambda search."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold


def make_folds(cv, n: int, groups) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(train, validation)`` index pairs.

    ``cv`` may be an int (``GroupKFold`` when ``groups`` is given, else unshuffled ``KFold``, so that
    contiguous rows such as consecutive time blocks stay together), a splitter with ``split``, or an
    iterable of index pairs. Rows that share a group never sit on both sides of a split.
    """
    idx = np.arange(n)
    if isinstance(cv, (int, np.integer)):
        splitter = GroupKFold(n_splits=int(cv)) if groups is not None else KFold(n_splits=int(cv))
        return [(np.asarray(a), np.asarray(b)) for a, b in splitter.split(idx, groups=groups)]
    if hasattr(cv, "split"):
        return [(np.asarray(a), np.asarray(b)) for a, b in cv.split(idx, groups=groups)]
    return [(np.asarray(a), np.asarray(b)) for a, b in cv]
