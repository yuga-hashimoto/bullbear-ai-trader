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


def load_diary_events(
    reports_dir: str | Path,
    limit: int = 30,
) -> list[dict[str, str]]:
    """Return the latest meaningful runtime events as Japanese diary entries."""
    from .diary import format_diary_event

    path = Path(reports_dir) / "runtime" / "paper_events.jsonl"
    events = _read_jsonl(path)
    if events.empty:
        return []
    entries = [
        formatted
        for event in events.to_dict(orient="records")
        if (formatted := format_diary_event(event)) is not None
    ]
    return list(reversed(entries[-limit:]))


def load_runtime_performance(
    reports_dir: str | Path,
    initial_cash: float,
) -> dict[str, Any]:
    """Calculate current PaperRunner performance from live runtime artifacts."""
    runtime = load_runtime(reports_dir)
    hb = runtime["heartbeat"]
    daily = runtime["daily_state"]
    events = _read_jsonl(Path(reports_dir) / "runtime" / "paper_events.jsonl")

    marked_equity = daily.get("marked_equity")
    if marked_equity is None:
        marked_equity = daily.get("cash")
    if marked_equity is None:
        marked_equity = initial_cash + float(hb.get("daily_pnl", 0.0))
    current_equity = float(marked_equity)
    total_pnl = current_equity - float(initial_cash)
    total_return_pct = (
        total_pnl / float(initial_cash) * 100.0
        if initial_cash
        else 0.0
    )

    closed = events[events["event"] == "POSITION_CLOSED"] if (
        not events.empty and "event" in events.columns
    ) else pd.DataFrame()
    closed_trades = len(closed)
    win_rate_pct: float | None = None
    if closed_trades and "net_pnl" in closed.columns:
        pnl = pd.to_numeric(closed["net_pnl"], errors="coerce").dropna()
        closed_trades = len(pnl)
        if closed_trades:
            win_rate_pct = float((pnl > 0).mean() * 100.0)

    return {
        "current_equity": round(current_equity, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 4),
        "daily_pnl": float(hb.get("daily_pnl", 0.0)),
        "daily_pnl_jpy": float(hb.get("daily_pnl_jpy", 0.0)),
        "closed_trades": closed_trades,
        "win_rate_pct": win_rate_pct,
        "timestamp": hb.get("timestamp"),
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
