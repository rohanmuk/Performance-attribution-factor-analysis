"""Brinson-Fachler holdings-based attribution with Carino multi-period linking.

SINGLE-PERIOD Brinson-Fachler, per sector i (all returns simple, one period):

    Rb        = sum_i wb_i * Rb_i                     (total benchmark return)
    Allocation_i  = (wp_i - wb_i) * (Rb_i - Rb)
    Selection_i   = wb_i * (Rp_i - Rb_i)
    Interaction_i = (wp_i - wb_i) * (Rp_i - Rb_i)

    sum_i (Allocation_i + Selection_i + Interaction_i) == Rp - Rb   (exact)

Brinson-Fachler (BF) vs. Brinson-Hood-Beebower (BHB):
    BHB allocation_i = (wp_i - wb_i) * Rb_i          (vs. zero)
    BF  allocation_i = (wp_i - wb_i) * (Rb_i - Rb)   (vs. the TOTAL benchmark)
They differ by Rb * sum_i(wp_i - wb_i) = Rb * 0 = 0, so the grand total reconciles
either way. BF is preferred because its allocation term measures the value of
over/under-weighting a sector relative to the whole benchmark, so a sector that
merely tracks the index contributes ~0 allocation regardless of its own level.

MULTI-PERIOD linking — Carino (2009):
Naive summation of single-period effects does NOT equal the compounded active
return because returns compound geometrically. Carino rescales each period's
effects by k_t / k, where

    k_t = [ln(1+Rp_t) - ln(1+Rb_t)] / (Rp_t - Rb_t)     (limit 1/(1+Rp_t))
    k   = [ln(1+Rp) - ln(1+Rb)] / (Rp - Rb)             (cumulative; same limit)

Because k_t*(Rp_t-Rb_t) = ln(1+Rp_t)-ln(1+Rb_t) telescopes, the linked effects sum
EXACTLY to the compounded active return Rp - Rb. Menchero (2000) is an alternative
optimized-linking scheme with a similar guarantee; Carino is used here for its
clean closed form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .data.benchmark import benchmark_weights_series
from .logging_setup import get_logger
from .portfolio import PortfolioPanel

logger = get_logger(__name__)

EFFECTS = ["allocation", "selection", "interaction"]


@dataclass
class BrinsonResult:
    """Full Brinson attribution output."""

    allocation: pd.DataFrame      # date x sector
    selection: pd.DataFrame       # date x sector
    interaction: pd.DataFrame     # date x sector
    period_totals: pd.DataFrame   # date x [allocation, selection, interaction, active, Rp, Rb]
    linked_by_sector: pd.DataFrame  # sector x [allocation, selection, interaction, total]
    linked_totals: pd.Series      # [allocation, selection, interaction, active]
    cumulative: pd.DataFrame      # date x [allocation, selection, interaction, active]
    carino_scale: pd.Series       # date -> k_t / k
    wp: pd.DataFrame              # portfolio sector weights used
    wb: pd.Series                 # benchmark sector weights used
    Rp_sector: pd.DataFrame       # portfolio sector returns used
    Rb_sector: pd.DataFrame       # benchmark sector returns used


def _carino_coefficient(rp: float, rb: float, eps: float = 1e-10) -> float:
    """Carino smoothing coefficient for one (portfolio, benchmark) return pair."""
    if abs(rp - rb) < eps:
        return 1.0 / (1.0 + rp)
    return (np.log1p(rp) - np.log1p(rb)) / (rp - rb)


def run_brinson(
    panel: PortfolioPanel,
    bench_sector_returns: pd.DataFrame,
    cfg: BenchmarkConfig,
) -> BrinsonResult:
    """Run single-period Brinson-Fachler for every month, then Carino-link.

    The analysis window is the intersection of months where the portfolio sectors
    and ALL benchmark sector ETFs have returns (e.g. XLC/XLRE inception truncates
    the early sample).
    """
    sectors = list(panel.sector_weights.columns)
    wb = benchmark_weights_series(cfg).reindex(sectors)
    if wb.isna().any():
        raise ValueError(f"Benchmark weights missing for sectors: "
                         f"{wb.index[wb.isna()].tolist()}")

    Rb_sector = bench_sector_returns[sectors]

    # Common, fully-populated month index.
    idx = (
        panel.sector_weights.dropna(how="any").index
        .intersection(panel.sector_returns.dropna(how="any").index)
        .intersection(Rb_sector.dropna(how="any").index)
    )
    idx = idx.sort_values()
    if len(idx) == 0:
        raise ValueError("No overlapping months across portfolio and benchmark ETFs.")

    wp = panel.sector_weights.loc[idx, sectors]
    Rp = panel.sector_returns.loc[idx, sectors]
    Rb = Rb_sector.loc[idx, sectors]

    # Total benchmark and portfolio returns per period.
    Rb_total = (Rb * wb).sum(axis=1)                 # sum_i wb_i Rb_i
    Rp_total = (wp * Rp).sum(axis=1)                 # sum_i wp_i Rp_i

    active_wp_minus_wb = wp.subtract(wb, axis=1)     # (wp_i - wb_i)

    # Single-period Brinson-Fachler effects (date x sector).
    allocation = active_wp_minus_wb.mul(Rb.subtract(Rb_total, axis=0))
    selection = Rp.subtract(Rb).mul(wb, axis=1)
    interaction = active_wp_minus_wb.mul(Rp.subtract(Rb))

    period_totals = pd.DataFrame({
        "allocation": allocation.sum(axis=1),
        "selection": selection.sum(axis=1),
        "interaction": interaction.sum(axis=1),
        "active": Rp_total - Rb_total,
        "Rp": Rp_total,
        "Rb": Rb_total,
    })

    # ----- Carino linking -----
    rp_arr = period_totals["Rp"].to_numpy()
    rb_arr = period_totals["Rb"].to_numpy()
    k_t = np.array([_carino_coefficient(rp, rb) for rp, rb in zip(rp_arr, rb_arr)])

    Rp_cum = float(np.prod(1.0 + rp_arr) - 1.0)
    Rb_cum = float(np.prod(1.0 + rb_arr) - 1.0)
    k = _carino_coefficient(Rp_cum, Rb_cum)
    scale = pd.Series(k_t / k, index=idx, name="carino_scale")

    # Linked effect per sector = sum_t scale_t * effect_{t,sector}.
    linked_alloc = allocation.mul(scale, axis=0).sum(axis=0)
    linked_sel = selection.mul(scale, axis=0).sum(axis=0)
    linked_inter = interaction.mul(scale, axis=0).sum(axis=0)
    linked_by_sector = pd.DataFrame({
        "allocation": linked_alloc,
        "selection": linked_sel,
        "interaction": linked_inter,
    })
    linked_by_sector["total"] = linked_by_sector.sum(axis=1)

    linked_totals = pd.Series({
        "allocation": linked_alloc.sum(),
        "selection": linked_sel.sum(),
        "interaction": linked_inter.sum(),
        "active": Rp_cum - Rb_cum,
    })

    # Cumulative (linked) attribution path — reconciles to linked_totals at the end.
    scaled_totals = period_totals[EFFECTS].mul(scale, axis=0)
    cumulative = scaled_totals.cumsum()
    cumulative["active"] = scaled_totals.sum(axis=1).cumsum()

    logger.info(
        "Brinson: %d months (%s to %s). Cumulative active = %.2f%% "
        "(alloc %.2f%%, sel %.2f%%, inter %.2f%%).",
        len(idx), idx.min().date(), idx.max().date(),
        linked_totals["active"] * 100, linked_totals["allocation"] * 100,
        linked_totals["selection"] * 100, linked_totals["interaction"] * 100,
    )

    return BrinsonResult(
        allocation=allocation,
        selection=selection,
        interaction=interaction,
        period_totals=period_totals,
        linked_by_sector=linked_by_sector,
        linked_totals=linked_totals,
        cumulative=cumulative,
        carino_scale=scale,
        wp=wp,
        wb=wb,
        Rp_sector=Rp,
        Rb_sector=Rb,
    )


def reconciliation_report(result: BrinsonResult) -> Dict[str, float]:
    """Return max per-period reconciliation error and the linked-vs-compounded gap."""
    per_period_sum = (
        result.period_totals[["allocation", "selection", "interaction"]].sum(axis=1)
    )
    per_period_err = float((per_period_sum - result.period_totals["active"]).abs().max())

    linked_effects_sum = result.linked_totals[EFFECTS].sum()
    linked_err = float(abs(linked_effects_sum - result.linked_totals["active"]))
    return {
        "max_single_period_error": per_period_err,
        "linked_reconciliation_error": linked_err,
    }
