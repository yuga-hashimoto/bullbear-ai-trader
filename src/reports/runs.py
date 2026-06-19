"""Per-run report storage.

Each backtest writes a self-contained directory under ``<reports_dir>/runs/``:

    reports/runs/<run_id>/
        config.yaml          resolved config snapshot
        summary.json         overview (run_id, period, headline metrics)
        metrics.json         full metrics dict
        trades.csv           closed-trade log
        daily_pnl.csv        per-day PnL
        equity_curve.csv     equity time series
        agent_signals.jsonl  every agent signal (+ accepted / rejection_reason)
        risk_decisions.jsonl  every risk decision (+ risk_state)
        report.md / report.html

``<reports_dir>/latest.json`` always points at the most recent run; a
``latest`` symlink is created too when the OS allows it (best-effort).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config.settings import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


def new_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S_%f")


def runs_root(cfg: Config) -> Path:
    return cfg.path("reports_dir") / "runs"


def run_dir(cfg: Config, run_id: str) -> Path:
    return runs_root(cfg) / run_id


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def save_run(
    cfg: Config,
    run_id: str,
    result,
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
    overview: dict[str, Any],
    report_paths: dict[str, Path] | None = None,
) -> Path:
    """Persist all artifacts for ``run_id`` and update the ``latest`` pointers."""
    d = run_dir(cfg, run_id)
    d.mkdir(parents=True, exist_ok=True)

    with open(d / "config.yaml", "w") as fh:
        yaml.safe_dump(cfg.raw, fh, sort_keys=False, allow_unicode=True)

    with open(d / "metrics.json", "w") as fh:
        json.dump({"metrics": metrics, "benchmark": benchmark,
                   "counters": getattr(result, "counters", {})}, fh, indent=2, default=str)

    with open(d / "summary.json", "w") as fh:
        json.dump(overview, fh, indent=2, default=str)

    result.trades_frame.to_csv(d / "trades.csv", index=False)
    result.daily_pnl.to_csv(d / "daily_pnl.csv", index=False)

    eq = result.equity_curve
    pd.DataFrame({"timestamp": eq.index, "equity": eq.to_numpy()}).to_csv(
        d / "equity_curve.csv", index=False
    )

    _write_jsonl(result.agent_signals, d / "agent_signals.jsonl")
    _write_jsonl(result.risk_decisions, d / "risk_decisions.jsonl")

    # report.md / report.html may be written by the report module; copy refs.
    if report_paths:
        for key in ("report_md", "report_html"):
            src = report_paths.get(key)
            if src and Path(src).exists() and Path(src).resolve() != (d / Path(src).name).resolve():
                (d / Path(src).name).write_text(Path(src).read_text())

    _update_latest(cfg, run_id)
    log.info("run saved: %s", d)
    return d


def _update_latest(cfg: Config, run_id: str) -> None:
    reports = cfg.path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / "latest.json", "w") as fh:
        json.dump({"run_id": run_id}, fh, indent=2)
    # Best-effort symlink (ignored on platforms / filesystems that disallow it).
    link = reports / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("runs") / run_id, target_is_directory=True)
    except OSError:
        pass


def resolve_run_id(cfg: Config, run_id: str | None) -> str:
    if run_id and run_id != "latest":
        return run_id
    latest_json = cfg.path("reports_dir") / "latest.json"
    if latest_json.exists():
        with open(latest_json) as fh:
            return json.load(fh)["run_id"]
    runs = list_runs(cfg)
    if not runs:
        raise FileNotFoundError("no runs found; run a backtest first")
    return runs[-1]


def list_runs(cfg: Config) -> list[str]:
    root = runs_root(cfg)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
