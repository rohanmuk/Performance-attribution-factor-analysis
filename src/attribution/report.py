"""Combined markdown + HTML report tying the two attribution lenses together."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .brinson import BrinsonResult, reconciliation_report, run_brinson
from .config import REPORTS_DIR
from .factor_model import (
    RegressionResult,
    build_excess_return,
    regress,
    regress_all_models,
    rolling_betas,
)
from .logging_setup import get_logger
from .pipeline import DataBundle
from . import plotting

logger = get_logger(__name__)


@dataclass
class _Doc:
    """Accumulates parallel markdown and HTML output."""

    md: List[str]
    html: List[str]

    def h(self, level: int, text: str) -> None:
        self.md.append(f"{'#' * level} {text}\n")
        self.html.append(f"<h{level}>{text}</h{level}>")

    def p(self, text: str) -> None:
        self.md.append(text + "\n")
        self.html.append(f"<p>{text}</p>")

    def table(self, df: pd.DataFrame, floatfmt: str = ".4f") -> None:
        self.md.append(df.to_markdown(floatfmt=floatfmt) + "\n")
        self.html.append(df.to_html(border=0, float_format=lambda x: f"{x:{floatfmt}}"))

    def img(self, path: Path, alt: str) -> None:
        rel = path.name
        self.md.append(f"![{alt}]({rel})\n")
        self.html.append(f'<img src="{rel}" alt="{alt}" style="max-width:100%;">')


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _brinson_sector_table(result: BrinsonResult) -> pd.DataFrame:
    df = result.linked_by_sector.copy() * 100
    df["portfolio_wt_avg"] = result.wp.mean() * 100
    df["benchmark_wt"] = result.wb * 100
    df = df[["portfolio_wt_avg", "benchmark_wt", "allocation", "selection",
             "interaction", "total"]]
    df = df.rename(columns={
        "portfolio_wt_avg": "Avg Port Wt %", "benchmark_wt": "Bench Wt %",
        "allocation": "Allocation %", "selection": "Selection %",
        "interaction": "Interaction %", "total": "Total %",
    })
    df = df.sort_values("Total %", ascending=False)
    df.loc["TOTAL"] = df.sum(numeric_only=True)
    df.loc["TOTAL", "Avg Port Wt %"] = result.wp.mean().sum() * 100
    df.loc["TOTAL", "Bench Wt %"] = result.wb.sum() * 100
    return df


def _active_models_table(active: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    results = regress_all_models(active, factors)
    rows = {}
    for name, res in results.items():
        row = {"alpha (ann) %": res.alpha_annual * 100, "alpha t": res.alpha_tstat}
        for f in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]:
            row[f] = res.params.get(f, float("nan"))
        row["R2"] = res.r2
        row["N"] = res.nobs
        rows[name] = row
    return pd.DataFrame(rows).T


def _coef_table(res: RegressionResult) -> pd.DataFrame:
    df = pd.DataFrame({
        "beta": res.params,
        "NW std err": res.se,
        "t-stat": res.tstats,
        "p-value": res.pvalues,
    })
    return df


def _narrative(bundle: DataBundle, brinson: BrinsonResult,
               port_c6: RegressionResult, active_c6: RegressionResult) -> List[str]:
    """Auto-generate the reconciliation narrative connecting the two lenses."""
    lines = []
    lt = brinson.linked_totals
    lines.append(
        f"Over the sample the portfolio's cumulative active return was "
        f"**{_pct(lt['active'])}**, decomposed by Brinson-Fachler into allocation "
        f"**{_pct(lt['allocation'])}**, selection **{_pct(lt['selection'])}**, and "
        f"interaction **{_pct(lt['interaction'])}**."
    )
    # Biggest allocation and selection bets.
    by_sec = brinson.linked_by_sector
    top_alloc = by_sec["allocation"].idxmax()
    bot_alloc = by_sec["allocation"].idxmin()
    top_sel = by_sec["selection"].idxmax()
    avg_active_wt = (brinson.wp.mean() - brinson.wb)
    lines.append(
        f"The largest positive allocation effect came from **{top_alloc}** "
        f"(avg active weight {_pct(avg_active_wt.get(top_alloc, 0))}), while "
        f"**{bot_alloc}** was the biggest allocation drag. The strongest stock "
        f"selection contribution was in **{top_sel}**."
    )
    # Connect to factor tilts.
    b = port_c6.betas
    tilt_bits = []
    if b.get("HML", 0) < -0.05:
        tilt_bits.append(f"a growth tilt (HML β = {b['HML']:+.2f})")
    elif b.get("HML", 0) > 0.05:
        tilt_bits.append(f"a value tilt (HML β = {b['HML']:+.2f})")
    if b.get("SMB", 0) < -0.05:
        tilt_bits.append(f"a large-cap tilt (SMB β = {b['SMB']:+.2f})")
    elif b.get("SMB", 0) > 0.05:
        tilt_bits.append(f"a small-cap tilt (SMB β = {b['SMB']:+.2f})")
    if b.get("MOM", 0) > 0.05:
        tilt_bits.append(f"positive momentum exposure (MOM β = {b['MOM']:+.2f})")
    if not tilt_bits:
        tilt_str = "broadly market-like factor exposures"
    elif len(tilt_bits) == 1:
        tilt_str = tilt_bits[0]
    else:
        tilt_str = ", ".join(tilt_bits[:-1]) + " and " + tilt_bits[-1]
    lines.append(
        f"The factor lens tells a consistent story: the portfolio's market beta is "
        f"{b.get('Mkt-RF', float('nan')):.2f} with {tilt_str}. A sector overweight in "
        f"a growth-heavy sector (e.g. Information Technology / Communication Services) "
        f"in the Brinson allocation term shows up here as the negative HML (growth) "
        f"loading — the same active bet, viewed through holdings vs. through returns."
    )
    lines.append(
        f"On the **active** return specifically, the Carhart-6 annualized alpha is "
        f"**{_pct(active_c6.alpha_annual)}** (t = {active_c6.alpha_tstat:.2f}); the "
        f"portion of active return explained by factor tilts vs. genuine skill "
        f"(alpha/selection) is what an allocator most wants to separate — which is "
        f"exactly what these two lenses jointly deliver."
    )
    return lines


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Active Return Attribution Report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
 h1 {{ border-bottom: 3px solid #2f6fb0; padding-bottom: .3rem; }}
 h2 {{ border-bottom: 1px solid #ddd; padding-bottom: .2rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; margin: 1rem 0; font-size: .9rem; }}
 th, td {{ padding: 4px 10px; border-bottom: 1px solid #eee; text-align: right; }}
 th {{ background: #f5f7fa; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ margin: 1rem 0; }}
 code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
</style></head><body>
{body}
<hr><p style="color:#888;font-size:.8rem;">Generated {ts} · Active Return Attribution Toolkit</p>
</body></html>"""


def build_report(
    bundle: DataBundle,
    out_dir: Path | None = None,
) -> Dict[str, Path]:
    """Run both lenses, generate charts, and write report.md + report.html."""
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run analyses ----
    brinson = run_brinson(bundle.panel, bundle.bench_sector_returns, bundle.benchmark_cfg)
    recon = reconciliation_report(brinson)

    port_excess = build_excess_return(bundle.portfolio_return, bundle.factors["RF"])
    active = bundle.active_return

    port_c6 = regress(port_excess, bundle.factors, "Carhart6")
    active_c6 = regress(active, bundle.factors, "Carhart6")
    roll = rolling_betas(port_excess, bundle.factors, model="Carhart6", window=36)

    # ---- Charts ----
    bcharts = plotting.all_brinson_charts(brinson, out_dir)
    contrib_chart = plotting.factor_contributions_bar(
        port_c6, "Portfolio factor-contribution decomposition (Carhart-6, annualized)",
        out_dir,
    )
    roll_chart = plotting.rolling_betas_line(
        roll, "Rolling 36-month factor betas (portfolio excess, Carhart-6)", out_dir,
    )

    # ---- Document ----
    doc = _Doc(md=[], html=[])
    doc.h(1, "Active Return Attribution Report")
    doc.p(f"**Portfolio:** {bundle.portfolio_cfg.name}  ")
    doc.p(f"**Sample:** {bundle.active_return.index.min().date()} to "
          f"{bundle.active_return.index.max().date()} "
          f"({len(bundle.active_return)} months, monthly frequency)")

    doc.h(2, "Executive summary")
    for line in _narrative(bundle, brinson, port_c6, active_c6):
        doc.p(line)

    doc.h(2, "Module A — Brinson-Fachler attribution (holdings-based)")
    doc.p(f"Single-period effects reconcile to the active return each month with "
          f"max error `{recon['max_single_period_error']:.2e}`; Carino-linked effects "
          f"reconcile to the cumulative active return with error "
          f"`{recon['linked_reconciliation_error']:.2e}`.")
    doc.table(_brinson_sector_table(brinson), floatfmt=".2f")
    doc.img(bcharts["by_sector"], "Brinson attribution by sector")
    doc.img(bcharts["waterfall"], "Active-return waterfall")
    doc.img(bcharts["cumulative"], "Cumulative attribution over time")

    doc.h(2, "Module B — Factor attribution (returns-based)")
    doc.p("Active return (portfolio − benchmark) regressed on each factor model "
          "(betas; annualized alpha with Newey-West t-stat):")
    doc.table(_active_models_table(active, bundle.factors), floatfmt=".3f")
    doc.p("Full Carhart-6 regression of **portfolio excess** return "
          "(Newey-West HAC standard errors):")
    doc.table(_coef_table(port_c6), floatfmt=".4f")
    doc.img(contrib_chart, "Factor contributions")
    doc.img(roll_chart, "Rolling betas")

    doc.h(2, "Reconciliation — one story, two lenses")
    doc.p("The Brinson allocation term and the factor betas are two projections of "
          "the same active bets. Sector over/underweights (holdings view) manifest as "
          "style-factor tilts (returns view); selection and Carhart-6 alpha both isolate "
          "the value added beyond those systematic exposures.")

    doc.h(2, "Data sources & caveats")
    bcfg = bundle.benchmark_cfg
    doc.p(f"- **Prices:** yfinance auto-adjusted monthly returns.")
    doc.p(f"- **Factors / RF:** Ken French Data Library (FF5 + momentum), monthly, decimal.")
    doc.p(f"- **Benchmark:** reconstructed from 11 SPDR sector ETFs weighted by a static "
          f"S&P 500 sector snapshot ({bcfg.snapshot_date}, source: {bcfg.source}). "
          f"Static weights and ETF proxying are stated approximations; reconstructed-vs-"
          f"SPY annualized tracking error ≈ "
          f"{bundle.tracking['diff'].std() * (12 ** 0.5) * 100:.2f}%.")

    body = "\n".join(doc.html)
    html = HTML_TEMPLATE.format(body=body, ts=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"))

    md_path = out_dir / "report.md"
    html_path = out_dir / "report.html"
    md_path.write_text("\n".join(doc.md))
    html_path.write_text(html)
    logger.info("Wrote report -> %s and %s", md_path, html_path)
    return {"md": md_path, "html": html_path, **bcharts,
            "contributions": contrib_chart, "rolling": roll_chart}
