"""End-to-end data pipeline shared by the CLI, the notebook, and the tests.

Loads prices + factors, simulates the portfolio, reconstructs the benchmark, and
returns a single bundle of aligned monthly series so the Brinson and factor modules
describe the *same* active return.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import (
    BenchmarkConfig,
    PortfolioConfig,
    load_benchmark_config,
    load_portfolio_config,
)
from .data.benchmark import (
    benchmark_sector_returns,
    reconstructed_benchmark_return,
    tracking_gap_vs_market,
)
from .data.factors import load_factors
from .data.prices import get_daily_prices, to_monthly_returns
from .logging_setup import get_logger
from .portfolio import PortfolioPanel, simulate_portfolio

logger = get_logger(__name__)


@dataclass
class DataBundle:
    """All aligned inputs the analysis modules need."""

    portfolio_cfg: PortfolioConfig
    benchmark_cfg: BenchmarkConfig
    panel: PortfolioPanel
    bench_sector_returns: pd.DataFrame
    portfolio_return: pd.Series      # total monthly portfolio return
    benchmark_return: pd.Series      # reconstructed benchmark return
    active_return: pd.Series         # portfolio_return - benchmark_return
    market_return: pd.Series         # market proxy (SPY) monthly return
    factors: pd.DataFrame            # Mkt-RF, SMB, HML, RMW, CMA, MOM, RF (decimal)
    tracking: pd.DataFrame           # reconstructed vs market proxy diagnostic
    sample_end: str


def load_bundle(
    portfolio_path: str | None = None,
    benchmark_path: str | None = None,
    use_cache: bool = True,
) -> DataBundle:
    """Build the full DataBundle from real yfinance + Ken French data."""
    pcfg = load_portfolio_config(portfolio_path)
    bcfg = load_benchmark_config(benchmark_path)

    all_tickers = sorted(set(pcfg.tickers) | set(bcfg.etf_tickers) | {bcfg.market_proxy})
    daily = get_daily_prices(all_tickers, pcfg.start, pcfg.end, use_cache=use_cache)
    monthly = to_monthly_returns(daily)

    factors = load_factors(pcfg.start, pcfg.end, use_cache=use_cache)

    # Cap the sample to the last month both prices and factors cover.
    sample_end = min(monthly.index.max(), factors.index.max())
    monthly = monthly.loc[:sample_end]
    factors = factors.loc[:sample_end]
    logger.info("Analysis sample ends %s (prices ∩ factors).", sample_end.date())

    # Portfolio simulation over full holding history.
    holding_returns = monthly[pcfg.tickers]
    panel = simulate_portfolio(holding_returns, pcfg)

    # Benchmark reconstruction.
    bench_sector_ret = benchmark_sector_returns(monthly, bcfg)
    benchmark_return = reconstructed_benchmark_return(bench_sector_ret, bcfg).dropna()

    market_return = monthly[bcfg.market_proxy].rename("market_return")
    tracking = tracking_gap_vs_market(benchmark_return, market_return)

    # Active return where both portfolio and reconstructed benchmark exist.
    aligned = pd.concat(
        {"p": panel.returns, "b": benchmark_return}, axis=1
    ).dropna()
    active_return = (aligned["p"] - aligned["b"]).rename("active_return")

    return DataBundle(
        portfolio_cfg=pcfg,
        benchmark_cfg=bcfg,
        panel=panel,
        bench_sector_returns=bench_sector_ret,
        portfolio_return=panel.returns.rename("portfolio_return"),
        benchmark_return=benchmark_return.rename("benchmark_return"),
        active_return=active_return,
        market_return=market_return,
        factors=factors,
        tracking=tracking,
        sample_end=str(sample_end.date()),
    )
