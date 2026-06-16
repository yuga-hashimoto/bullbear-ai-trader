"""ReplayAgent — replays pre-recorded signals from a JSONL file.

Use cases:
  * Backtest signals that OpenClaw / HermesAgent produced in a separate process.
  * Reproduce a past agent run deterministically.
  * Compare different agents' decisions on the same data.

Behavior:
  * Loads a ``.jsonl`` file (one signal JSON per line), indexed by timestamp.
  * For a bar's timestamp, returns the matching signal; if none exists, returns
    NO_TRADE.
  * Malformed lines are skipped at load time and counted; at request time an
    unparseable/invalid stored signal is surfaced as-is so the engine's
    validation step counts it as invalid (and falls back to NO_TRADE).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.logging import get_logger
from .base import BaseAgent
from .signal_schema import Signal, SignalValidationError, no_trade_signal

log = get_logger(__name__)


class ReplayAgent(BaseAgent):
    name = "ReplayAgent"
    version = "1.0.0"

    def __init__(self, signal_file: str | Path) -> None:
        if signal_file is None:
            raise ValueError("ReplayAgent requires a signal_file (JSONL path)")
        self.signal_file = Path(signal_file)
        self._by_ts: dict[str, dict[str, Any]] = {}
        self.skipped_lines = 0
        self._load()

    def _load(self) -> None:
        if not self.signal_file.exists():
            raise FileNotFoundError(f"signal file not found: {self.signal_file}")
        with open(self.signal_file) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ts = str(obj["timestamp"])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    self.skipped_lines += 1
                    log.warning("replay: skipping line %d: %s", lineno, exc)
                    continue
                self._by_ts[ts] = obj
        log.info(
            "replay loaded %d signals (%d skipped) from %s",
            len(self._by_ts), self.skipped_lines, self.signal_file,
        )

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        ts = context["timestamp"]
        raw = self._by_ts.get(ts)
        if raw is None:
            return no_trade_signal(ts, self.name, reason="no recorded signal").to_dict()
        # Ensure the timestamp/agent_name are present for downstream logging.
        raw = dict(raw)
        raw.setdefault("agent_name", self.name)
        raw.setdefault("timestamp", ts)
        return raw

    @staticmethod
    def save_signals(signals: list[dict[str, Any]], path: str | Path) -> Path:
        """Persist a list of signal dicts to JSONL (one per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for s in signals:
                fh.write(json.dumps(s, default=str) + "\n")
        return path


def validate_signal_file(path: str | Path) -> dict[str, Any]:
    """Validate every signal in a JSONL file; return a summary report."""
    path = Path(path)
    total = valid = invalid = unparseable = 0
    errors: list[dict[str, Any]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                unparseable += 1
                errors.append({"line": lineno, "error": f"json: {exc}"})
                continue
            try:
                Signal.from_dict(obj).validate()
                valid += 1
            except SignalValidationError as exc:
                invalid += 1
                errors.append({"line": lineno, "error": str(exc)})
    return {
        "file": str(path),
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "unparseable": unparseable,
        "errors": errors[:50],
    }
