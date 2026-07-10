"""Shared fixtures. Synthetic data keeps unit tests fast, deterministic, offline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260710)


@pytest.fixture(scope="session")
def month_index() -> pd.DatetimeIndex:
    return pd.date_range("2016-01-31", periods=96, freq="ME")


@pytest.fixture(scope="session")
def synthetic_factors(month_index, rng) -> pd.DataFrame:
    """Plausible monthly factor returns (decimal) for regression tests."""
    n = len(month_index)
    df = pd.DataFrame({
        "Mkt-RF": rng.normal(0.006, 0.043, n),
        "SMB": rng.normal(0.001, 0.03, n),
        "HML": rng.normal(0.0, 0.03, n),
        "RMW": rng.normal(0.002, 0.02, n),
        "CMA": rng.normal(0.001, 0.02, n),
        "MOM": rng.normal(0.003, 0.045, n),
        "RF": np.full(n, 0.0015),
    }, index=month_index)
    return df
