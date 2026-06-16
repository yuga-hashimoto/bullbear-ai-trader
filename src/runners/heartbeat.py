"""Runtime I/O: events, heartbeat and state artifacts.

Everything a running PaperRunner emits is written under ``reports/runtime/``:

    heartbeat.json            latest runner heartbeat
    paper_events.jsonl        append-only event stream (event-driven log)
    errors.jsonl              append-only error stream
    current_positions.json    snapshot of open paper positions
    daily_state.json          today's pnl / trade count / stop state
    latest_signal.json         most recent agent signal
    latest_risk_decision.json  most recent risk decision

A heartbeat write failure raises :class:`HeartbeatError` so the runner can stop
safely (a runner that can't report its liveness must not keep trading).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventType:
    RUNNER_STARTED = "RUNNER_STARTED"
    RUNNER_SLEEPING = "RUNNER_SLEEPING"
    MARKET_OPEN = "MARKET_OPEN"
    MARKET_CLOSED = "MARKET_CLOSED"
    MARKET_HOLIDAY = "MARKET_HOLIDAY"
    EARLY_CLOSE = "EARLY_CLOSE"
    MARKET_DATA = "MARKET_DATA"
    MARKET_DATA_STALE = "MARKET_DATA_STALE"
    FEATURE_BUILT = "FEATURE_BUILT"
    AGENT_CONTEXT = "AGENT_CONTEXT"
    AGENT_SIGNAL = "AGENT_SIGNAL"
    AGENT_ERROR = "AGENT_ERROR"
    SIGNAL_VALIDATION = "SIGNAL_VALIDATION"
    RISK_DECISION = "RISK_DECISION"
    ORDER_INTENT = "ORDER_INTENT"
    ORDER_REJECTED = "ORDER_REJECTED"
    PAPER_FILL = "PAPER_FILL"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    FORCE_EXIT = "FORCE_EXIT"
    DAILY_STOP = "DAILY_STOP"
    HEARTBEAT = "HEARTBEAT"
    RUNNER_STOPPED = "RUNNER_STOPPED"


_ERROR_EVENTS = {EventType.AGENT_ERROR, EventType.MARKET_DATA_STALE,
                 EventType.ORDER_REJECTED, EventType.DAILY_STOP}


class HeartbeatError(RuntimeError):
    """Raised when the heartbeat file cannot be written."""


class RuntimeWriter:
    def __init__(self, runtime_dir: str | Path) -> None:
        self.dir = Path(runtime_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- events --------------------------------------------------------------
    def emit(self, event_type: str, payload: dict[str, Any] | None = None,
             now: datetime | None = None) -> dict[str, Any]:
        event = {
            "timestamp": (now or datetime.now(timezone.utc)).isoformat(),
            "event": event_type,
            **(payload or {}),
        }
        self._append("paper_events.jsonl", event)
        if event_type in _ERROR_EVENTS:
            self._append("errors.jsonl", event)
        return event

    def emit_error(self, message: str, detail: dict[str, Any] | None = None) -> None:
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "event": EventType.AGENT_ERROR, "message": message, **(detail or {})}
        self._append("errors.jsonl", rec)
        self._append("paper_events.jsonl", rec)

    def _append(self, name: str, obj: dict[str, Any]) -> None:
        with open(self.dir / name, "a") as fh:
            fh.write(json.dumps(obj, default=str) + "\n")

    # -- json snapshots ------------------------------------------------------
    def write_json(self, name: str, obj: Any) -> None:
        tmp = self.dir / (name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        tmp.replace(self.dir / name)

    def read_json(self, name: str) -> dict[str, Any] | None:
        path = self.dir / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    # -- heartbeat -----------------------------------------------------------
    def write_heartbeat(self, hb: dict[str, Any]) -> None:
        try:
            self.write_json("heartbeat.json", hb)
        except OSError as exc:  # disk full, permissions, etc.
            raise HeartbeatError(f"failed to write heartbeat: {exc}") from exc

    def read_heartbeat(self) -> dict[str, Any] | None:
        return self.read_json("heartbeat.json")
