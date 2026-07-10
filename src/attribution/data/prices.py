"""Price download and monthly-return construction (yfinance -> parquet cache).

We download auto-adjusted daily closes (adjusted for splits and dividends), cache
them to ``data/prices.parquet``, then resample to month-end and convert to simple
monthly returns. Simple (not log) returns are used because Brinson attribution and
weight arithmetic are defined on simple returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf

from ..config import DATA_DIR
from ..logging_setup import get_logger

logger = get_logger(__name__)

_PRICES_CACHE = DATA_DIR / "prices.parquet"


def _download_adjusted_close(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download auto-adjusted daily closes for ``tickers`` as a wide DataFrame."""
    logger.info("Downloading daily prices for %d tickers from yfinance...", len(tickers))
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # split & dividend adjusted
        progress=False,
        group_by="column",
    )
    # With multiple tickers yfinance returns a column MultiIndex (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:  # single ticker
        close = raw[["Close"]].copy()
        close.columns = tickers
    close = close.reindex(columns=tickers)
    missing = [t for t in tickers if close[t].isna().all()]
    if missing:
        raise RuntimeError(f"yfinance returned no data for: {missing}")
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    return close.sort_index()


def get_daily_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Return daily adjusted closes, downloading and caching on a cache miss.

    The cache is keyed on the union of tickers and the requested date span: if the
    cached frame already covers every ticker and the full window, it is reused.
    """
    tickers = sorted(set(t.upper() for t in tickers))
    cache_path = cache_path or _PRICES_CACHE

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index)
        have_tickers = set(cached.columns)
        covers_span = (
            cached.index.min() <= pd.Timestamp(start)
            and cached.index.max() >= pd.Timestamp(end) - pd.offsets.BDay(5)
        )
        if set(tickers).issubset(have_tickers) and covers_span:
            logger.info("Using cached prices (%s)", cache_path.name)
            return cached.loc[str(start):str(end), tickers].sort_index()

    close = _download_adjusted_close(list(tickers), start, end)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache_path)
    logger.info("Cached prices -> %s (%d rows)", cache_path, len(close))
    return close


def to_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily adjusted closes to simple month-end returns.

    Resample to the last observation each calendar month, then percent-change.
    The first month (all-NaN from pct_change) is dropped.
    """
    monthly_px = daily_prices.resample("ME").last()
    monthly_ret = monthly_px.pct_change().dropna(how="all")
    monthly_ret.index.name = "date"
    return monthly_ret
