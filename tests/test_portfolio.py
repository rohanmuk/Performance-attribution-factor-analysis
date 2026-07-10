"""Portfolio weight dynamics: drift, rebalance, and the sector aggregation identity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attribution.config import Holding, PortfolioConfig
from attribution.portfolio import simulate_portfolio


def _make_cfg(rebalance: str) -> PortfolioConfig:
    holdings = [
        Holding("AAA", "Tech", 0.4),
        Holding("BBB", "Tech", 0.2),
        Holding("CCC", "Fin", 0.4),
    ]
    return PortfolioConfig("t", "2016-01-01", "2018-12-31", rebalance, holdings)


@pytest.fixture()
def returns(rng):
    idx = pd.date_range("2016-01-31", periods=24, freq="ME")
    return pd.DataFrame(
        rng.normal(0.01, 0.05, (len(idx), 3)), index=idx, columns=["AAA", "BBB", "CCC"]
    )


def test_begin_weights_sum_to_one(returns):
    panel = simulate_portfolio(returns, _make_cfg("Q"))
    row_sums = panel.holding_weights.sum(axis=1)
    assert np.allclose(row_sums.to_numpy(), 1.0, atol=1e-12)


def test_sector_weights_sum_to_one(returns):
    panel = simulate_portfolio(returns, _make_cfg("Q"))
    assert np.allclose(panel.sector_weights.sum(axis=1).to_numpy(), 1.0, atol=1e-12)


def test_sector_identity_matches_total_return(returns):
    """sum_i wp_i * Rp_i must equal the total portfolio return each month."""
    panel = simulate_portfolio(returns, _make_cfg("Q"))
    reconstructed = (panel.sector_weights * panel.sector_returns).sum(axis=1)
    assert np.allclose(reconstructed.to_numpy(), panel.returns.to_numpy(), atol=1e-12)


def test_quarterly_rebalance_resets_to_target(returns):
    """On calendar-quarter starts the beginning weights snap back to target."""
    panel = simulate_portfolio(returns, _make_cfg("Q"))
    target = pd.Series({"AAA": 0.4, "BBB": 0.2, "CCC": 0.4})
    for date, row in panel.holding_weights.iterrows():
        if date.month in (1, 4, 7, 10):
            assert np.allclose(row[target.index].to_numpy(), target.to_numpy(), atol=1e-12)


def test_monthly_rebalance_is_always_target(returns):
    panel = simulate_portfolio(returns, _make_cfg("M"))
    target = np.array([0.4, 0.2, 0.4])
    for _, row in panel.holding_weights.iterrows():
        assert np.allclose(row[["AAA", "BBB", "CCC"]].to_numpy(), target, atol=1e-12)


def test_drift_between_rebalances(returns):
    """Between rebalances weights should move (not equal target every month)."""
    panel = simulate_portfolio(returns, _make_cfg("Q"))
    target = np.array([0.4, 0.2, 0.4])
    drifted = [
        not np.allclose(row[["AAA", "BBB", "CCC"]].to_numpy(), target, atol=1e-9)
        for date, row in panel.holding_weights.iterrows()
        if date.month not in (1, 4, 7, 10)
    ]
    assert any(drifted)
