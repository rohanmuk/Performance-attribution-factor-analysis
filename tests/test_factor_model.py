"""Factor-model correctness.

The hand-rolled OLS + Newey-West is validated against statsmodels (an independent
implementation used ONLY in tests), plus the classic sanity check that SPY regressed
on the market factor gives beta ~= 1 and alpha ~= 0 on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attribution.factor_model import (
    FACTOR_SETS,
    build_excess_return,
    newey_west_lag,
    regress,
    rolling_betas,
)

sm = pytest.importorskip("statsmodels.api")


@pytest.fixture()
def synthetic_y(synthetic_factors, rng):
    """A dependent series with known-ish loadings plus noise."""
    f = synthetic_factors
    y = (0.001
         + 0.95 * f["Mkt-RF"]
         + 0.30 * f["SMB"]
         - 0.20 * f["HML"]
         + 0.15 * f["MOM"]
         + rng.normal(0, 0.01, len(f)))
    return y.rename("y")


def test_ols_matches_statsmodels(synthetic_y, synthetic_factors):
    res = regress(synthetic_y, synthetic_factors, "Carhart6")
    names = FACTOR_SETS["Carhart6"]
    X = sm.add_constant(synthetic_factors[names])
    sm_res = sm.OLS(synthetic_y, X).fit()
    # Coefficients (point estimates) must match to high precision.
    assert res.params["alpha"] == pytest.approx(sm_res.params["const"], abs=1e-10)
    for n in names:
        assert res.params[n] == pytest.approx(sm_res.params[n], abs=1e-10)
    assert res.r2 == pytest.approx(sm_res.rsquared, abs=1e-10)


def test_newey_west_matches_statsmodels(synthetic_y, synthetic_factors):
    res = regress(synthetic_y, synthetic_factors, "FF5")
    names = FACTOR_SETS["FF5"]
    lag = newey_west_lag(res.nobs)
    X = sm.add_constant(synthetic_factors[names])
    sm_res = sm.OLS(synthetic_y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
    )
    # Newey-West standard errors must match statsmodels HAC (no small-sample corr).
    assert res.se["alpha"] == pytest.approx(sm_res.bse["const"], rel=1e-6)
    for n in names:
        assert res.se[n] == pytest.approx(sm_res.bse[n], rel=1e-6)


def test_contributions_sum_to_mean(synthetic_y, synthetic_factors):
    res = regress(synthetic_y, synthetic_factors, "Carhart6")
    # alpha + sum(beta_k * mean f_k) == mean(y), annualized on both sides.
    contrib_total = res.contributions().sum()
    assert contrib_total == pytest.approx(synthetic_y.mean() * 12, abs=1e-10)


def test_rolling_betas_shape(synthetic_y, synthetic_factors):
    roll = rolling_betas(synthetic_y, synthetic_factors, "FF3", window=36)
    assert list(roll.columns) == FACTOR_SETS["FF3"]
    assert len(roll) == len(synthetic_factors) - 36 + 1


@pytest.mark.network
def test_spy_on_market_gives_beta_one_alpha_zero():
    """Real-data sanity check: SPY excess ~ CAPM => beta ~= 1, alpha ~= 0."""
    from attribution.data.factors import load_factors
    from attribution.data.prices import get_daily_prices, to_monthly_returns

    try:
        px = get_daily_prices(["SPY"], "2015-01-01", "2024-12-31")
        factors = load_factors("2015-01-01", "2024-12-31")
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"network/data unavailable: {exc}")

    spy_ret = to_monthly_returns(px)["SPY"].rename("SPY")
    spy_excess = build_excess_return(spy_ret, factors["RF"])
    res = regress(spy_excess, factors, "CAPM")

    assert res.params["Mkt-RF"] == pytest.approx(1.0, abs=0.05)
    assert abs(res.alpha_monthly) < 0.003          # < ~3.6% annualized
    assert abs(res.alpha_tstat) < 2.5              # not statistically distinguishable
    assert res.r2 > 0.95                           # SPY basically IS the market
