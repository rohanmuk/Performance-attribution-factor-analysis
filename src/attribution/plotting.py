"""Matplotlib charts for both attribution lenses. All figures saved as PNG."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .brinson import BrinsonResult  # noqa: E402
from .config import REPORTS_DIR  # noqa: E402
from .factor_model import RegressionResult  # noqa: E402
from .logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

# A small, consistent palette.
C_ALLOC = "#2f6fb0"
C_SELECT = "#e2803b"
C_INTER = "#6aa84f"
C_ACTIVE = "#8250c4"
C_GRID = "#d9d9d9"

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def _save(fig, name: str, out_dir: Path | None) -> Path:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart -> %s", path)
    return path


def brinson_stacked_bar(result: BrinsonResult, out_dir: Path | None = None) -> Path:
    """Stacked allocation/selection/interaction by sector (linked, in %)."""
    df = (result.linked_by_sector[["allocation", "selection", "interaction"]] * 100)
    df = df.sort_values("allocation")
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom_pos = np.zeros(len(df))
    bottom_neg = np.zeros(len(df))
    for col, color in [("allocation", C_ALLOC), ("selection", C_SELECT),
                       ("interaction", C_INTER)]:
        vals = df[col].to_numpy()
        base = np.where(vals >= 0, bottom_pos, bottom_neg)
        ax.barh(df.index, vals, left=base, color=color, label=col.capitalize())
        bottom_pos = bottom_pos + np.where(vals >= 0, vals, 0)
        bottom_neg = bottom_neg + np.where(vals < 0, vals, 0)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Contribution to active return (%)")
    ax.set_title("Brinson-Fachler attribution by sector (Carino-linked)")
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, "brinson_by_sector.png", out_dir)


def brinson_waterfall(result: BrinsonResult, out_dir: Path | None = None) -> Path:
    """Waterfall from benchmark return to portfolio return via total effects."""
    t = result.linked_totals * 100
    rb = result.period_totals["Rb"]
    rb_cum = (np.prod(1 + rb) - 1) * 100
    steps = [
        ("Benchmark", rb_cum, "#888888"),
        ("Allocation", t["allocation"], C_ALLOC),
        ("Selection", t["selection"], C_SELECT),
        ("Interaction", t["interaction"], C_INTER),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    running = 0.0
    for i, (label, val, color) in enumerate(steps):
        if i == 0:
            ax.bar(label, val, color=color)
            running = val
        else:
            ax.bar(label, val, bottom=running, color=color)
            running += val
    ax.bar("Portfolio", running, color="#333333")
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("From benchmark to portfolio: linked active-return waterfall")
    for i, (label, val, _) in enumerate(steps):
        ax.annotate(f"{val:+.1f}", (i, val), ha="center", va="bottom", fontsize=8)
    return _save(fig, "brinson_waterfall.png", out_dir)


def brinson_cumulative(result: BrinsonResult, out_dir: Path | None = None) -> Path:
    """Cumulative (linked) attribution over time."""
    cum = result.cumulative * 100
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(cum.index, cum["allocation"], color=C_ALLOC, label="Allocation")
    ax.plot(cum.index, cum["selection"], color=C_SELECT, label="Selection")
    ax.plot(cum.index, cum["interaction"], color=C_INTER, label="Interaction")
    ax.plot(cum.index, cum["active"], color=C_ACTIVE, linewidth=2.0,
            label="Total active")
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_ylabel("Cumulative contribution (%)")
    ax.set_title("Cumulative Brinson attribution over time")
    ax.legend(loc="best", frameon=False, ncol=2)
    return _save(fig, "brinson_cumulative.png", out_dir)


def factor_contributions_bar(
    result: RegressionResult, title: str, out_dir: Path | None = None,
    name: str = "factor_contributions.png",
) -> Path:
    """Annualized factor-contribution decomposition (beta x premium) + alpha."""
    contrib = result.contributions() * 100
    colors = ["#8250c4" if k == "alpha" else "#2f6fb0" for k in contrib.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(contrib.index, contrib.to_numpy(), color=colors)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_ylabel("Annualized contribution (%)")
    ax.set_title(title)
    for i, v in enumerate(contrib.to_numpy()):
        ax.annotate(f"{v:+.2f}", (i, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    return _save(fig, name, out_dir)


def rolling_betas_line(
    betas: pd.DataFrame, title: str, out_dir: Path | None = None,
    name: str = "rolling_betas.png",
) -> Path:
    """Rolling factor betas over time."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col in betas.columns:
        ax.plot(betas.index, betas[col], label=col)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_ylabel("Beta")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, ncol=3)
    return _save(fig, name, out_dir)


def all_brinson_charts(result: BrinsonResult, out_dir: Path | None = None) -> Dict[str, Path]:
    return {
        "by_sector": brinson_stacked_bar(result, out_dir),
        "waterfall": brinson_waterfall(result, out_dir),
        "cumulative": brinson_cumulative(result, out_dir),
    }
