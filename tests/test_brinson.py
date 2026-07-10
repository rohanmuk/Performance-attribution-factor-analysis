"""Brinson-Fachler correctness: single-period + multi-period (Carino) reconciliation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attribution.brinson import _carino_coefficient, run_brinson, reconciliation_report
from attribution.config import BenchmarkConfig, Holding, PortfolioConfig
from attribution.portfolio import PortfolioPanel, simulate_portfolio


# ---------------------------------------------------------------------------
# Tiny hand-computable example
# ---------------------------------------------------------------------------
def test_single_period_known_example():
    """Two sectors, one period, numbers worked out by hand."""
    # Sector A: wp=0.7, wb=0.6, Rp=0.10, Rb=0.08
    # Sector B: wp=0.3, wb=0.4, Rp=0.02, Rb=0.03
    wp = pd.Series({"A": 0.7, "B": 0.3})
    wb = pd.Series({"A": 0.6, "B": 0.4})
    Rp = pd.Series({"A": 0.10, "B": 0.02})
    Rb = pd.Series({"A": 0.08, "B": 0.03})

    Rb_total = float((wb * Rb).sum())          # 0.6*0.08 + 0.4*0.03 = 0.060
    Rp_total = float((wp * Rp).sum())          # 0.7*0.10 + 0.3*0.02 = 0.076

    alloc = (wp - wb) * (Rb - Rb_total)
    selc = wb * (Rp - Rb)
    inter = (wp - wb) * (Rp - Rb)

    # Hand values:
    # A: alloc=(.1)(.08-.06)=.002 ; sel=.6(.02)=.012 ; inter=.1(.02)=.002
    # B: alloc=(-.1)(.03-.06)=.003; sel=.4(-.01)=-.004; inter=(-.1)(-.01)=.001
    assert alloc["A"] == pytest.approx(0.002)
    assert alloc["B"] == pytest.approx(0.003)
    assert selc["A"] == pytest.approx(0.012)
    assert selc["B"] == pytest.approx(-0.004)
    assert inter["A"] == pytest.approx(0.002)
    assert inter["B"] == pytest.approx(0.001)

    total = float((alloc + selc + inter).sum())
    assert total == pytest.approx(Rp_total - Rb_total, abs=1e-12)  # 0.016


# ---------------------------------------------------------------------------
# Fixtures: a small synthetic 3-sector portfolio + benchmark
# ---------------------------------------------------------------------------
@pytest.fixture()
def small_setup(rng):
    sectors = ["Tech", "Fin", "Energy"]
    tickers = {"Tech": ["T1", "T2"], "Fin": ["F1"], "Energy": ["E1", "E2"]}
    holdings = []
    for s, ts in tickers.items():
        for t in ts:
            holdings.append(Holding(ticker=t, sector=s, target_weight=1.0 / 5))
    pcfg = PortfolioConfig(
        name="small", start="2016-01-01", end="2019-12-31",
        rebalance="Q", holdings=holdings,
    )
    idx = pd.date_range("2016-01-31", periods=36, freq="ME")
    all_t = [h.ticker for h in holdings]
    holding_ret = pd.DataFrame(
        rng.normal(0.01, 0.05, (len(idx), len(all_t))), index=idx, columns=all_t
    )
    panel = simulate_portfolio(holding_ret, pcfg)

    bcfg = BenchmarkConfig(
        market_proxy="SPY",
        sector_etfs={"Tech": "XLK", "Fin": "XLF", "Energy": "XLE"},
        sector_weights={"Tech": 0.5, "Fin": 0.3, "Energy": 0.2},
        snapshot_date="2019-12-31", source="test",
    )
    bench_sector_ret = pd.DataFrame(
        rng.normal(0.008, 0.04, (len(idx), 3)), index=idx, columns=sectors
    )
    return panel, bench_sector_ret, bcfg


def test_single_period_reconciles_each_month(small_setup):
    panel, bench_sector_ret, bcfg = small_setup
    result = run_brinson(panel, bench_sector_ret, bcfg)
    pt = result.period_totals
    effects_sum = pt[["allocation", "selection", "interaction"]].sum(axis=1)
    # Each month's effects sum to that month's active return.
    assert np.allclose(effects_sum.to_numpy(), pt["active"].to_numpy(), atol=1e-12)


def test_carino_linked_reconciles_to_cumulative(small_setup):
    panel, bench_sector_ret, bcfg = small_setup
    result = run_brinson(panel, bench_sector_ret, bcfg)

    rp = result.period_totals["Rp"].to_numpy()
    rb = result.period_totals["Rb"].to_numpy()
    cum_active = np.prod(1 + rp) - 1 - (np.prod(1 + rb) - 1)

    linked_sum = result.linked_totals[["allocation", "selection", "interaction"]].sum()
    assert linked_sum == pytest.approx(cum_active, abs=1e-12)
    assert result.linked_totals["active"] == pytest.approx(cum_active, abs=1e-12)

    recon = reconciliation_report(result)
    assert recon["max_single_period_error"] < 1e-12
    assert recon["linked_reconciliation_error"] < 1e-12


def test_carino_coefficient_limit():
    """When Rp == Rb the coefficient reduces to 1/(1+Rp)."""
    assert _carino_coefficient(0.05, 0.05) == pytest.approx(1 / 1.05)
    # Off the limit, matches the log-difference form.
    k = _carino_coefficient(0.10, 0.02)
    assert k == pytest.approx((np.log1p(0.10) - np.log1p(0.02)) / (0.10 - 0.02))


def test_naive_sum_does_not_reconcile(small_setup):
    """Sanity: naive summation generally does NOT equal the compounded active."""
    panel, bench_sector_ret, bcfg = small_setup
    result = run_brinson(panel, bench_sector_ret, bcfg)
    naive = result.period_totals["active"].sum()
    rp = result.period_totals["Rp"].to_numpy()
    rb = result.period_totals["Rb"].to_numpy()
    compounded = np.prod(1 + rp) - 1 - (np.prod(1 + rb) - 1)
    # With 36 volatile months the geometric gap should be clearly non-zero.
    assert abs(naive - compounded) > 1e-6
