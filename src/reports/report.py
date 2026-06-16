"""Backtest report generation (Markdown + lightweight HTML).

Also writes the machine-readable artifacts required by the spec: a trade log
CSV and a daily-PnL CSV. The report includes the disclaimer and the benchmark
comparison so results are never read in isolation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DISCLAIMER = (
    "This system is for research/education only and is NOT investment advice. "
    "Backtested results do not guarantee future performance. Live trading is "
    "disabled by default."
)


def _metrics_table_md(metrics: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    return "| metric | value |\n| --- | --- |\n" + rows


def _bench_table_md(bench: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in bench.items())
    return "| benchmark | total_return_pct |\n| --- | --- |\n" + rows


def _dist_table_md(title: str, dist: dict) -> str:
    if not dist:
        return f"_{title}: none_\n"
    rows = "\n".join(f"| {k} | {v} |" for k, v in dist.items())
    return f"| {title} | count |\n| --- | --- |\n" + rows + "\n"


def write_reports(
    out_dir: Path,
    metrics: dict,
    benchmark: dict,
    trades: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    title: str = "Backtest Report",
    counters: dict | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    trades_csv = out_dir / "trades.csv"
    trades.to_csv(trades_csv, index=False)
    paths["trades_csv"] = trades_csv

    daily_csv = out_dir / "daily_pnl.csv"
    daily_pnl.to_csv(daily_csv, index=False)
    paths["daily_pnl_csv"] = daily_csv

    md = f"""# {title}

> {DISCLAIMER}

## Performance metrics

{_metrics_table_md(metrics)}

## Benchmark comparison

{_bench_table_md(benchmark)}

## Agent / Risk

{_dist_table_md("agent action", (counters or {}).get("action_distribution", {}))}
{_dist_table_md("risk rejection reason", (counters or {}).get("risk_rejection_reasons", {}))}

## Trades

Total trades: {len(trades)} (full log in `trades.csv`)

## Daily PnL

Trading days: {len(daily_pnl)} (full series in `daily_pnl.csv`)
"""
    md_path = out_dir / "report.md"
    md_path.write_text(md)
    paths["report_md"] = md_path

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:2rem;}}
table{{border-collapse:collapse;margin:1rem 0;}}td,th{{border:1px solid #ccc;padding:4px 10px;}}
.disclaimer{{background:#fff3cd;border:1px solid #ffe69c;padding:10px;border-radius:6px;}}</style>
</head><body>
<h1>{title}</h1>
<p class="disclaimer">{DISCLAIMER}</p>
<h2>Performance metrics</h2>
{pd.DataFrame(metrics.items(), columns=["metric", "value"]).to_html(index=False)}
<h2>Benchmark comparison</h2>
{pd.DataFrame(benchmark.items(), columns=["benchmark", "total_return_pct"]).to_html(index=False)}
<h2>Trades</h2><p>Total trades: {len(trades)}</p>
{trades.head(50).to_html(index=False) if not trades.empty else "<p>No trades.</p>"}
</body></html>"""
    html_path = out_dir / "report.html"
    html_path.write_text(html)
    paths["report_html"] = html_path

    return paths
