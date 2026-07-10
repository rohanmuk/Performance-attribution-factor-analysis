"""Command-line interface: ``python -m attribution {brinson,factors,report}``."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .brinson import reconciliation_report, run_brinson
from .factor_model import (
    build_excess_return,
    regress,
    regress_all_models,
    rolling_betas,
)
from .logging_setup import get_logger
from .pipeline import load_bundle
from . import plotting, report

logger = get_logger(__name__)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--portfolio", default=None, help="Path to portfolio.yml")
    p.add_argument("--benchmark", default=None, help="Path to benchmark.yml")
    p.add_argument("--no-cache", action="store_true", help="Force re-download of data")


def cmd_brinson(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.portfolio, args.benchmark, use_cache=not args.no_cache)
    result = run_brinson(bundle.panel, bundle.bench_sector_returns, bundle.benchmark_cfg)
    recon = reconciliation_report(result)

    print("\n=== Brinson-Fachler linked attribution by sector (%) ===")
    print((result.linked_by_sector * 100).round(3).to_string())
    print("\n=== Linked totals (%) ===")
    print((result.linked_totals * 100).round(3).to_string())
    print(f"\nReconciliation: max single-period error = "
          f"{recon['max_single_period_error']:.2e}, "
          f"linked error = {recon['linked_reconciliation_error']:.2e}")

    charts = plotting.all_brinson_charts(result)
    print("\nCharts:", ", ".join(str(p) for p in charts.values()))
    return 0


def cmd_factors(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.portfolio, args.benchmark, use_cache=not args.no_cache)
    port_excess = build_excess_return(bundle.portfolio_return, bundle.factors["RF"])
    active = bundle.active_return

    print("\n=== Portfolio excess return — all models ===")
    for name, res in regress_all_models(port_excess, bundle.factors).items():
        print(f"\n[{name}]  n={res.nobs} lag={res.nw_lag}  "
              f"alpha(ann)={res.alpha_annual*100:+.2f}% (t={res.alpha_tstat:.2f})  "
              f"R2={res.r2:.3f}")
        tbl = pd.DataFrame({"beta": res.params, "NW_se": res.se, "t": res.tstats})
        print(tbl.round(4).to_string())

    print("\n=== Active return — Carhart-6 ===")
    a6 = regress(active, bundle.factors, "Carhart6")
    print(f"alpha(ann)={a6.alpha_annual*100:+.2f}% (t={a6.alpha_tstat:.2f})  R2={a6.r2:.3f}")
    print(pd.DataFrame({"beta": a6.params, "NW_se": a6.se, "t": a6.tstats}).round(4).to_string())

    c = plotting.factor_contributions_bar(
        regress(port_excess, bundle.factors, "Carhart6"),
        "Portfolio factor-contribution decomposition (Carhart-6, annualized)",
    )
    roll = rolling_betas(port_excess, bundle.factors, "Carhart6", 36)
    r = plotting.rolling_betas_line(roll, "Rolling 36-month factor betas (Carhart-6)")
    print("\nCharts:", c, r)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.portfolio, args.benchmark, use_cache=not args.no_cache)
    paths = report.build_report(bundle)
    print("\nReport written:")
    for k, v in paths.items():
        print(f"  {k:14s} {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="attribution",
        description="Active return attribution: Brinson-Fachler + factor models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_b = sub.add_parser("brinson", help="Holdings-based Brinson-Fachler attribution")
    _add_common(p_b)
    p_b.set_defaults(func=cmd_brinson)

    p_f = sub.add_parser("factors", help="Returns-based factor attribution")
    _add_common(p_f)
    p_f.set_defaults(func=cmd_factors)

    p_r = sub.add_parser("report", help="Build combined markdown/HTML report")
    _add_common(p_r)
    p_r.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("Command failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
