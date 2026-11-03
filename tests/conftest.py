import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pytest

from scaleprobit.datasets import make_brownian_threshold, make_transfer_pair


@pytest.fixture(scope="session")
def data():
    return make_brownian_threshold(4000, n_features=3, coef=[0.4, -0.3, 0.1], scale_intercept=0.2,
                                   bias=0.1, base_scale=True, random_state=1)


@pytest.fixture(scope="session")
def pair():
    return make_transfer_pair(n_source=20000, n_target=300, n_test=4000, shift=0.05, n_features=3,
                              random_state=2)


def kw(b, idx=None):
    """The structural keyword arguments of a synthetic bunch, optionally restricted to rows idx."""
    sel = (lambda a: a) if idx is None else (lambda a: a[idx])
    out = {"distance": sel(b.distance), "horizon": sel(b.horizon)}
    if b.base_scale is not None:
        out["base_scale"] = sel(b.base_scale)
    return out


@pytest.fixture
def rng():
    return np.random.default_rng(0)
