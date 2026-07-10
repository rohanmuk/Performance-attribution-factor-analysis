"""Reconstructed benchmark: SPDR sector-ETF returns weighted by an S&P 500 snapshot.

The Brinson benchmark return each period is

    Rb(t) = sum_i  wb_i * Rb_i(t)

where ``Rb_i`` is the monthly return of the SPDR sector ETF proxying sector ``i``
and ``wb_i`` is the (static) snapshot weight for that sector. This is a documented
approximation of the true S&P 500 sector structure (see configs/benchmark.yml).
"""
from __future__ import annotations

import pandas as pd

from ..config import BenchmarkConfig
from ..logging_setup import get_logger

logger = get_logger(__name__)


def benchmark_sector_returns(
    monthly_returns: pd.DataFrame, cfg: BenchmarkConfig
) -> pd.DataFrame:
    """Map ETF monthly returns to a sector-named DataFrame (columns = GICS sectors)."""
    cols = {}
    for sector, etf in cfg.sector_etfs.items():
        if etf not in monthly_returns.columns:
            raise KeyError(f"Missing monthly returns for sector ETF {etf} ({sector}).")
        cols[sector] = monthly_returns[etf]
    out = pd.DataFrame(cols)
    return out[list(cfg.sector_etfs.keys())]


def benchmark_weights_series(cfg: BenchmarkConfig) -> pd.Series:
    """Return the normalized benchmark sector weights as a Series (sums to 1)."""
    w = pd.Series(cfg.sector_weights, name="benchmark_weight")
    return w / w.sum()


def reconstructed_benchmark_return(
    sector_returns: pd.DataFrame, cfg: BenchmarkConfig
) -> pd.Series:
    """Compute the static-weight reconstructed benchmark return per period."""
    w = benchmark_weights_series(cfg).reindex(sector_returns.columns)
    rb = sector_returns.mul(w, axis=1).sum(axis=1)
    rb.name = "benchmark_return"
    return rb


def tracking_gap_vs_market(
    reconstructed: pd.Series, market_monthly_return: pd.Series
) -> pd.DataFrame:
    """Diagnostic: reconstructed benchmark vs. the actual market proxy (e.g. SPY).

    Quantifies how far the static-weight sector reconstruction drifts from the
    true index return — a key stated caveat.
    """
    df = pd.concat(
        {"reconstructed": reconstructed, "market_proxy": market_monthly_return},
        axis=1,
    ).dropna()
    df["diff"] = df["reconstructed"] - df["market_proxy"]
    logger.info(
        "Reconstructed-vs-market tracking: mean abs monthly diff = %.4f%%, "
        "annualized tracking error = %.2f%%",
        df["diff"].abs().mean() * 100,
        df["diff"].std() * (12 ** 0.5) * 100,
    )
    return df
