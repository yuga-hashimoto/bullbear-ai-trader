"""Run data loader for the dashboard.

Reads a single run directory into DataFrames / dicts, tolerating missing or
empty files so the dashboard never crashes on a NO_TRADE-only or
rejected-only run. Works off a plain ``reports_dir`` path (no Config needed) so
the Streamlit app can be launched standalone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_REPORTS_DIR = "reports"


def runs_root(reports_dir: str | Path) -> Path:
    return Path(reports_dir) / "runs"


def list_run_ids(reports_dir: str | Path) -> list[str]:
    root = runs_root(reports_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def resolve_run_id(reports_dir: str | Path, run_id: str | None) -> str:
    if run_id and run_id != "latest":
        return run_id
    latest = Path(reports_dir) / "latest.json"
    if latest.exists():
        return json.loads(latest.read_text())["run_id"]
    ids = list_run_ids(reports_dir)
    if not ids:
        raise FileNotFoundError(f"no runs under {reports_dir}")
    return ids[-1]


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass(frozen=True)
class RunData:
    run_id: str
    path: Path
    summary: dict[str, Any]
    metrics: dict[str, Any]
    benchmark: dict[str, Any]
    counters: dict[str, Any]
    trades: pd.DataFrame
    daily_pnl: pd.DataFrame
    equity: pd.DataFrame
    agent_signals: pd.DataFrame
    risk_decisions: pd.DataFrame


def load_runtime(reports_dir: str | Path) -> dict[str, Any]:
    """Load PaperRunner runtime artifacts (robust to missing files)."""
    d = Path(reports_dir) / "runtime"
    events = _read_jsonl(d / "paper_events.jsonl")
    errors = _read_jsonl(d / "errors.jsonl")
    return {
        "dir": d,
        "exists": d.exists(),
        "heartbeat": _read_json(d / "heartbeat.json") or {},
        "latest_signal": _read_json(d / "latest_signal.json") or {},
        "latest_risk_decision": _read_json(d / "latest_risk_decision.json") or {},
        "current_positions": _read_json(d / "current_positions.json") or [],
        "daily_state": _read_json(d / "daily_state.json") or {},
        "events": events.tail(50) if not events.empty else events,
        "errors": errors.tail(50) if not errors.empty else errors,
    }


def load_evolution(reports_dir: str | Path) -> dict[str, Any]:
    """Load Champion/Challenger registry + evolution artifacts (robust)."""
    import yaml

    reg = Path(reports_dir) / "registry"
    evo = Path(reports_dir) / "evolution"

    champion: dict[str, Any] = {}
    cpath = reg / "champion.yaml"
    if cpath.exists():
        try:
            champion = yaml.safe_load(cpath.read_text()) or {}
        except yaml.YAMLError:
            champion = {}
    challengers = _read_json(reg / "challengers.json")

    def _jsonl(p: Path):
        df = _read_jsonl(p)
        return df

    return {
        "champion": champion,
        "previous_champions": _jsonl(reg / "previous_champions.jsonl"),
        "challengers": challengers if isinstance(challengers, list) else [],
        "promotions": _jsonl(reg / "promotions.jsonl"),
        "rollbacks": _jsonl(reg / "rollbacks.jsonl"),
        "allocations": _jsonl(reg / "allocations.jsonl"),
        "status": _read_json(evo / "evolution_status.json") or {},
        "events": _jsonl(evo / "evolution_events.jsonl"),
        "shadow_pnl": _jsonl(evo / "shadow_pnl.jsonl"),
        "drift": _jsonl(evo / "drift_alerts.jsonl"),
        "mutations": _jsonl(evo / "mutations.jsonl"),
    }


def load_run(reports_dir: str | Path, run_id: str | None = None) -> RunData:
    rid = resolve_run_id(reports_dir, run_id)
    d = runs_root(reports_dir) / rid
    if not d.exists():
        raise FileNotFoundError(f"run not found: {d}")
    metrics_blob = _read_json(d / "metrics.json")
    return RunData(
        run_id=rid,
        path=d,
        summary=_read_json(d / "summary.json"),
        metrics=metrics_blob.get("metrics", {}),
        benchmark=metrics_blob.get("benchmark", {}),
        counters=metrics_blob.get("counters", {}),
        trades=_read_csv(d / "trades.csv"),
        daily_pnl=_read_csv(d / "daily_pnl.csv"),
        equity=_read_csv(d / "equity_curve.csv"),
        agent_signals=_read_jsonl(d / "agent_signals.jsonl"),
        risk_decisions=_read_jsonl(d / "risk_decisions.jsonl"),
    )
