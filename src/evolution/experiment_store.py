"""Evolution event log + experiment artifacts.

Events are appended to ``reports/evolution/evolution_events.jsonl`` (and mirror
to the runner's ``paper_events.jsonl`` when a RuntimeWriter is supplied). Shadow
PnL goes to ``reports/evolution/shadow_pnl.jsonl``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EvolutionEventType:
    CHALLENGER_CREATED = "CHALLENGER_CREATED"
    CHALLENGER_SHADOW_STARTED = "CHALLENGER_SHADOW_STARTED"
    CHALLENGER_CANARY_STARTED = "CHALLENGER_CANARY_STARTED"
    ALLOCATION_UPDATED = "ALLOCATION_UPDATED"
    PROMOTION_EVALUATED = "PROMOTION_EVALUATED"
    PROMOTION_PASSED = "PROMOTION_PASSED"
    PROMOTION_FAILED = "PROMOTION_FAILED"
    CHAMPION_PROMOTED = "CHAMPION_PROMOTED"
    ROLLBACK_TRIGGERED = "ROLLBACK_TRIGGERED"
    CHAMPION_ROLLED_BACK = "CHAMPION_ROLLED_BACK"
    MUTATION_GENERATED = "MUTATION_GENERATED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    EVOLUTION_LOOP_COMPLETED = "EVOLUTION_LOOP_COMPLETED"


class ExperimentStore:
    def __init__(self, reports_dir: str | Path, runtime_writer=None) -> None:
        self.dir = Path(reports_dir) / "evolution"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runtime_writer = runtime_writer

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **(payload or {}),
        }
        self._append("evolution_events.jsonl", event)
        # Mirror to the runner event stream if available (spec: also events.jsonl).
        if self.runtime_writer is not None:
            try:
                self.runtime_writer.emit(event_type, payload or {})
            except Exception:  # noqa: BLE001 - mirroring must never break evolution
                pass
        return event

    def record_shadow(self, record: dict[str, Any]) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        self._append("shadow_pnl.jsonl", record)

    def record_mutation(self, record: dict[str, Any]) -> None:
        self._append("mutations.jsonl", {"timestamp": datetime.now(timezone.utc).isoformat(), **record})

    def record_drift(self, record: dict[str, Any]) -> None:
        self._append("drift_alerts.jsonl", {"timestamp": datetime.now(timezone.utc).isoformat(), **record})

    def write_status(self, status: dict[str, Any]) -> None:
        self._write_json("evolution_status.json", status)

    def read_status(self) -> dict[str, Any]:
        return self._read_json("evolution_status.json") or {}

    def read_events(self, limit: int = 100) -> list[dict[str, Any]]:
        path = self.dir / "evolution_events.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return rows[-limit:]

    # -- io helpers ----------------------------------------------------------
    def _append(self, name: str, obj: dict[str, Any]) -> None:
        with open(self.dir / name, "a") as fh:
            fh.write(json.dumps(obj, default=str) + "\n")

    def _write_json(self, name: str, obj: Any) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=str))
        tmp.replace(self.dir / name)

    def _read_json(self, name: str) -> dict[str, Any] | None:
        path = self.dir / name
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None
