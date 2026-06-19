"""Evidence-based readiness checks for research, paper, and live operation."""
from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import Config
from ..data.store import load_features
from ..pipeline import _slice
from ..runners.base import runtime_dir
from ..runners.heartbeat import RuntimeWriter


def _check(code: str, ok: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"code": code, "ok": bool(ok), "severity": severity, "detail": detail}


def _has_opencode_key() -> bool:
    if os.getenv("OPENCODE_GO_API_KEY") or os.getenv("OPENCODE_API_KEY"):
        return True
    return Path("/Users/yu-ga/.hermes/.env").exists()


def _recent_events(path: Path, window_seconds: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
            ts = datetime.fromisoformat(str(event["timestamp"])).astimezone(timezone.utc)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if ts.timestamp() >= cutoff:
            events.append(event)
    return events


def build_readiness(cfg: Config) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    features_ok = False
    try:
        matrix = load_features(cfg)
        test = _slice(matrix, cfg.test_start, cfg.test_end, cfg.timezone)
        features_ok = not test.empty
        checks.append(_check(
            "features_window",
            features_ok,
            (
                f"{len(test)} rows in {cfg.test_start}..{cfg.test_end}"
                if features_ok
                else "configured test window has no feature rows"
            ),
        ))
    except FileNotFoundError as exc:
        checks.append(_check("features_missing", False, str(exc)))

    key_ok = cfg.agent.type != "external" or _has_opencode_key()
    checks.append(_check(
        "opencode_key",
        key_ok,
        "configured" if key_ok else "external analysis selected but API key is unavailable",
    ))
    live_disabled = not (
        cfg.live_trading_enabled or cfg.trading.allow_live_orders
    )
    checks.append(_check(
        "live_disabled",
        live_disabled,
        "real-money order switches are disabled" if live_disabled else "live switch enabled",
    ))

    hb = RuntimeWriter(runtime_dir(cfg)).read_heartbeat()
    heartbeat_detail = "no heartbeat yet"
    heartbeat_fresh = False
    if hb and hb.get("timestamp"):
        try:
            age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(str(hb["timestamp"])).astimezone(timezone.utc)
            ).total_seconds()
            heartbeat_fresh = age <= max(cfg.runner.heartbeat_interval_seconds * 3, 120)
            heartbeat_detail = f"heartbeat age {round(age, 1)} seconds"
        except ValueError:
            heartbeat_detail = "heartbeat timestamp is invalid"
    checks.append(_check(
        "paper_heartbeat",
        heartbeat_fresh,
        heartbeat_detail,
        severity="warning",
    ))

    market_state = str((hb or {}).get("market_state", ""))
    status = str((hb or {}).get("status", ""))
    status_ok = (
        status == "running"
        if market_state in {"open", "early_close"}
        else status in {"running", "sleeping"}
    )
    checks.append(_check(
        "paper_status",
        status_ok,
        f"status={status or 'missing'} market_state={market_state or 'missing'}",
        severity="warning",
    ))

    reason = str((hb or {}).get("reason") or "")
    reason_ok = reason == ""
    checks.append(_check(
        "paper_reason",
        reason_ok,
        "no active runner error" if reason_ok else f"reason={reason}",
        severity="warning",
    ))

    processed_ok = (
        market_state not in {"open", "early_close"}
        or bool((hb or {}).get("last_processed_bar_time"))
    )
    checks.append(_check(
        "paper_processed_bar",
        processed_ok,
        (
            "processed bar present or market closed"
            if processed_ok
            else "market is open but no bar has been processed"
        ),
        severity="warning",
    ))

    runtime = runtime_dir(cfg)
    event_window = max(cfg.runner.heartbeat_interval_seconds * 10, 300)
    recent_errors = _recent_events(runtime / "errors.jsonl", event_window)
    errors_ok = len(recent_errors) == 0
    checks.append(_check(
        "recent_runtime_errors",
        errors_ok,
        f"{len(recent_errors)} errors in the last {event_window} seconds",
        severity="warning",
    ))

    recent_events = _recent_events(runtime / "paper_events.jsonl", event_window)
    starts = sum(1 for event in recent_events if event.get("event") == "RUNNER_STARTED")
    stops = sum(1 for event in recent_events if event.get("event") == "RUNNER_STOPPED")
    restart_loop = starts >= 3 and stops >= 2
    checks.append(_check(
        "restart_loop",
        not restart_loop,
        f"recent starts={starts}, stops={stops}",
        severity="warning",
    ))

    heartbeat_ok = (
        heartbeat_fresh
        and status_ok
        and reason_ok
        and processed_ok
        and errors_ok
        and not restart_loop
    )
    paper_ready = features_ok and key_ok and live_disabled and heartbeat_ok
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "ready": paper_ready,
            "running": heartbeat_ok,
        },
        "live_trading": {
            "enabled": not live_disabled,
            "eligible": False,
            "reason": "requires six months paper evidence and explicit reviewed live integration",
        },
        "checks": checks,
    }
