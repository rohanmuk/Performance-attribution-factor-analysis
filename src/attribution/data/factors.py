"""Fama-French 5 factors + momentum + risk-free from the Ken French Data Library.

Primary path: ``pandas-datareader`` (no API key). Fallback: download the CSV zip
files directly from Dartmouth and parse the monthly block by hand. Both paths yield
the same tidy monthly frame:

    columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'RF']  (decimal, not %)
    index   = month-end Timestamps

Ken French publishes returns in PERCENT; we divide by 100 so everything downstream
is in decimal, consistent with the price returns.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import List

import pandas as pd
import requests

from ..config import DATA_DIR
from ..logging_setup import get_logger

logger = get_logger(__name__)

_FACTORS_CACHE = DATA_DIR / "factors.parquet"

_FF5_DATASET = "F-F_Research_Data_5_Factors_2x3"
_MOM_DATASET = "F-F_Momentum_Factor"

_KF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FF5_ZIP = _KF_BASE + "F-F_Research_Data_5_Factors_2x3_CSV.zip"
_MOM_ZIP = _KF_BASE + "F-F_Momentum_Factor_CSV.zip"

# Canonical column order after standardization.
FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]


def _period_index_to_month_end(idx: pd.PeriodIndex | pd.Index) -> pd.DatetimeIndex:
    """Convert a monthly PeriodIndex (or parseable index) to month-end Timestamps."""
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp(how="end").normalize()
    return pd.to_datetime(idx)


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and rename momentum to 'MOM'."""
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.rename(columns={"Mom": "MOM", "MOM   ": "MOM"})
    return df


# ---------------------------------------------------------------------------
# Primary path: pandas-datareader
# ---------------------------------------------------------------------------
def _load_via_datareader(start: str, end: str) -> pd.DataFrame:
    import pandas_datareader.data as web  # imported lazily so fallback works if missing

    logger.info("Fetching Ken French factors via pandas-datareader...")
    ff5 = web.DataReader(_FF5_DATASET, "famafrench", start, end)[0]
    mom = web.DataReader(_MOM_DATASET, "famafrench", start, end)[0]

    ff5 = _standardize(ff5)
    mom = _standardize(mom)

    ff5.index = _period_index_to_month_end(ff5.index)
    mom.index = _period_index_to_month_end(mom.index)

    out = ff5.join(mom[["MOM"]], how="inner")
    out = out[FACTOR_COLUMNS] / 100.0  # percent -> decimal
    return out


# ---------------------------------------------------------------------------
# Fallback path: direct CSV zip download + manual parse
# ---------------------------------------------------------------------------
def _parse_kf_csv(text: str, value_names: List[str]) -> pd.DataFrame:
    """Parse the MONTHLY block of a Ken French CSV.

    Monthly rows are ``YYYYMM,val,val,...``; the file later contains an annual
    block with 4-digit years plus explanatory footers. We keep only rows whose
    first field is exactly 6 digits.
    """
    rows = []
    monthly = re.compile(r"^\s*(\d{6})\s*,(.*)$")
    for line in text.splitlines():
        m = monthly.match(line)
        if not m:
            continue
        period = m.group(1)
        values = [v.strip() for v in m.group(2).split(",") if v.strip() != ""]
        if len(values) < len(value_names):
            continue
        rows.append([period] + values[: len(value_names)])
    if not rows:
        raise RuntimeError("Could not parse any monthly rows from Ken French CSV.")

    df = pd.DataFrame(rows, columns=["period"] + value_names)
    df = df.set_index("period")
    df.index = pd.to_datetime(df.index, format="%Y%m").to_period("M")
    df.index = _period_index_to_month_end(df.index)
    return df.astype(float)


def _download_zip_csv(url: str) -> str:
    """Download a Ken French CSV zip and return the decoded text of the CSV inside."""
    logger.info("Downloading %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            return io.TextIOWrapper(fh, encoding="latin-1").read()


def _load_via_csv(start: str, end: str) -> pd.DataFrame:
    logger.info("Falling back to direct Ken French CSV download...")
    ff5_txt = _download_zip_csv(_FF5_ZIP)
    mom_txt = _download_zip_csv(_MOM_ZIP)

    ff5 = _parse_kf_csv(ff5_txt, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _parse_kf_csv(mom_txt, ["MOM"])

    out = ff5.join(mom[["MOM"]], how="inner")
    out = out[FACTOR_COLUMNS] / 100.0
    return out.loc[str(start):str(end)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_factors(
    start: str,
    end: str,
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Return monthly Fama-French 5 + momentum + RF in decimal, month-end indexed.

    Tries pandas-datareader first, then a direct CSV download. Results are cached to
    ``data/factors.parquet``.
    """
    cache_path = cache_path or _FACTORS_CACHE
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index)
        if (
            cached.index.min() <= pd.Timestamp(start)
            and cached.index.max() >= pd.Timestamp(end) - pd.offsets.MonthEnd(2)
        ):
            logger.info("Using cached factors (%s)", cache_path.name)
            return cached.loc[str(start):str(end)]

    try:
        out = _load_via_datareader(start, end)
    except Exception as exc:  # noqa: BLE001 - any failure should trigger the fallback
        logger.warning("pandas-datareader failed (%s); using CSV fallback.", exc)
        out = _load_via_csv(start, end)

    out = out.dropna(how="any").sort_index()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path)
    logger.info("Cached factors -> %s (%d months)", cache_path, len(out))
    return out.loc[str(start):str(end)]
