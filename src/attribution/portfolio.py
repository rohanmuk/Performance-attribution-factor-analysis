"""Portfolio weight dynamics and sector aggregation.

Given monthly holding returns and target weights, we simulate beginning-of-period
weights that DRIFT with prices between rebalances and RESET to target on the
rebalance schedule, then aggregate holdings to GICS-sector weights and returns for
Brinson attribution.

Key identity (verified in tests):
    sum_i wp_i(t) * Rp_i(t) == portfolio_return(t)
where wp_i is the sector weight and Rp_i the within-sector portfolio return.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .config import PortfolioConfig
from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioPanel:
    """Container for the simulated portfolio series."""

    returns: pd.Series            # total monthly portfolio return
    holding_weights: pd.DataFrame # beginning-of-period holding weights (cols=tickers)
    sector_weights: pd.DataFrame  # beginning-of-period sector weights wp (cols=sectors)
    sector_returns: pd.DataFrame  # within-sector portfolio returns Rp (cols=sectors)


def _is_rebalance_month(month: int, freq: str) -> bool:
    freq = freq.upper()
    if freq == "M":
        return True
    if freq == "Q":
        return month in (1, 4, 7, 10)  # calendar-quarter starts
    if freq == "A":
        return month == 1
    if freq in ("NONE", "N", ""):
        return False
    raise ValueError(f"Unknown rebalance frequency: {freq!r}")


def simulate_portfolio(
    holding_returns: pd.DataFrame,
    cfg: PortfolioConfig,
) -> PortfolioPanel:
    """Simulate drifting weights and aggregate to sector level.

    Parameters
    ----------
    holding_returns
        Monthly simple returns, columns = tickers (must cover cfg.tickers).
    cfg
        Portfolio configuration (targets, sector map, rebalance frequency).
    """
    tickers = cfg.tickers
    returns = holding_returns[tickers].dropna(how="any").sort_index()
    if returns.empty:
        raise ValueError("No complete-history months available for the holdings.")

    target = pd.Series(cfg.target_weights, index=tickers, dtype=float)
    target = target / target.sum()  # defensive re-normalization
    sector_of = cfg.ticker_sector
    sectors = cfg.sectors

    begin_weight_rows: Dict[pd.Timestamp, pd.Series] = {}
    port_ret_rows: Dict[pd.Timestamp, float] = {}

    w = target.copy()
    for i, (date, r) in enumerate(returns.iterrows()):
        rebal = (i == 0) or _is_rebalance_month(date.month, cfg.rebalance)
        w_begin = target.copy() if rebal else w
        begin_weight_rows[date] = w_begin
        port_ret_rows[date] = float((w_begin * r).sum())
        # Drift to end-of-period weights (buy-and-hold within the period).
        w_end = w_begin * (1.0 + r)
        w = w_end / w_end.sum()

    holding_weights = pd.DataFrame(begin_weight_rows).T[tickers]
    holding_weights.index.name = "date"
    port_returns = pd.Series(port_ret_rows, name="portfolio_return").sort_index()
    port_returns.index.name = "date"

    # Aggregate to sectors.
    sec_w = _aggregate_sector_weights(holding_weights, sector_of, sectors)
    sec_r = _aggregate_sector_returns(holding_weights, returns, sector_of, sectors, sec_w)

    logger.info(
        "Simulated portfolio: %d months (%s to %s), %d sectors.",
        len(port_returns), port_returns.index.min().date(),
        port_returns.index.max().date(), len(sectors),
    )
    return PortfolioPanel(port_returns, holding_weights, sec_w, sec_r)


def _aggregate_sector_weights(
    holding_weights: pd.DataFrame, sector_of: Dict[str, str], sectors: List[str]
) -> pd.DataFrame:
    """Sum holding weights within each sector -> wp_i(t)."""
    frames = {}
    for s in sectors:
        cols = [t for t in holding_weights.columns if sector_of[t] == s]
        frames[s] = holding_weights[cols].sum(axis=1)
    out = pd.DataFrame(frames)[sectors]
    out.index.name = "date"
    return out


def _aggregate_sector_returns(
    holding_weights: pd.DataFrame,
    returns: pd.DataFrame,
    sector_of: Dict[str, str],
    sectors: List[str],
    sector_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Within-sector weighted holding return -> Rp_i(t).

    Rp_i = sum_{j in i} w_j * r_j / sum_{j in i} w_j. If a sector's weight is ~0
    (fully drifted away, not possible here but guarded), fall back to 0.
    """
    frames = {}
    for s in sectors:
        cols = [t for t in holding_weights.columns if sector_of[t] == s]
        contrib = (holding_weights[cols] * returns[cols]).sum(axis=1)
        wp = sector_weights[s]
        frames[s] = (contrib / wp.where(wp != 0.0)).fillna(0.0)
    out = pd.DataFrame(frames)[sectors]
    out.index.name = "date"
    return out
