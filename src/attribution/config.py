"""Configuration models and YAML loaders.

Typed dataclasses wrap the two YAML config files so the rest of the toolkit works
with validated Python objects rather than raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from .logging_setup import get_logger

logger = get_logger(__name__)

# Repository layout anchors (…/src/attribution/config.py -> repo root).
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"


@dataclass(frozen=True)
class Holding:
    """A single portfolio constituent."""

    ticker: str
    sector: str
    target_weight: float


@dataclass(frozen=True)
class PortfolioConfig:
    """Active portfolio definition."""

    name: str
    start: str
    end: str
    rebalance: str
    holdings: List[Holding]

    @property
    def tickers(self) -> List[str]:
        return [h.ticker for h in self.holdings]

    @property
    def target_weights(self) -> Dict[str, float]:
        return {h.ticker: h.target_weight for h in self.holdings}

    @property
    def ticker_sector(self) -> Dict[str, str]:
        return {h.ticker: h.sector for h in self.holdings}

    @property
    def sectors(self) -> List[str]:
        # Preserve first-seen order for stable, readable outputs.
        seen: List[str] = []
        for h in self.holdings:
            if h.sector not in seen:
                seen.append(h.sector)
        return seen


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark definition for Brinson attribution."""

    market_proxy: str
    sector_etfs: Dict[str, str]
    sector_weights: Dict[str, float]  # normalized, sums to 1.0
    snapshot_date: str
    source: str

    @property
    def etf_tickers(self) -> List[str]:
        return list(self.sector_etfs.values())


def load_portfolio_config(path: Path | str | None = None) -> PortfolioConfig:
    """Load and validate the portfolio YAML."""
    path = Path(path) if path else CONFIG_DIR / "portfolio.yml"
    raw = yaml.safe_load(Path(path).read_text())

    holdings = [
        Holding(
            ticker=str(h["ticker"]).upper(),
            sector=str(h["sector"]),
            target_weight=float(h["target_weight"]),
        )
        for h in raw["holdings"]
    ]
    total = sum(h.target_weight for h in holdings)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Portfolio target weights sum to {total:.6f}, expected 1.0. "
            "Fix configs/portfolio.yml."
        )
    cfg = PortfolioConfig(
        name=str(raw["name"]),
        start=str(raw["start"]),
        end=str(raw["end"]),
        rebalance=str(raw.get("rebalance", "Q")).upper(),
        holdings=holdings,
    )
    logger.info(
        "Loaded portfolio '%s': %d holdings across %d sectors, rebalance=%s",
        cfg.name, len(cfg.holdings), len(cfg.sectors), cfg.rebalance,
    )
    return cfg


def load_benchmark_config(path: Path | str | None = None) -> BenchmarkConfig:
    """Load the benchmark YAML and normalize the sector-weight snapshot to sum to 1."""
    path = Path(path) if path else CONFIG_DIR / "benchmark.yml"
    raw = yaml.safe_load(Path(path).read_text())

    weights_pct = {str(k): float(v) for k, v in raw["sector_weights_pct"].items()}
    total_pct = sum(weights_pct.values())
    if total_pct <= 0:
        raise ValueError("Benchmark sector weights must be positive.")
    # Normalize so the reconstructed benchmark weights are a proper distribution.
    weights = {k: v / total_pct for k, v in weights_pct.items()}

    sector_etfs = {str(k): str(v).upper() for k, v in raw["sector_etfs"].items()}

    # Every weighted sector must have an ETF proxy.
    missing = set(weights) - set(sector_etfs)
    if missing:
        raise ValueError(f"Sectors missing an ETF mapping: {sorted(missing)}")

    cfg = BenchmarkConfig(
        market_proxy=str(raw["market_proxy"]).upper(),
        sector_etfs=sector_etfs,
        sector_weights=weights,
        snapshot_date=str(raw.get("sector_weights_snapshot_date", "")),
        source=str(raw.get("sector_weights_source", "")),
    )
    logger.info(
        "Loaded benchmark: %d sectors, snapshot %s (weights normalized from %.1f%%)",
        len(cfg.sector_weights), cfg.snapshot_date, total_pct,
    )
    return cfg
