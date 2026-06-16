"""Structured logging helpers.

Every trading decision, feature snapshot, model output and risk verdict should
be traceable. We keep a single configured root logger plus a lightweight
``DecisionLogger`` that appends structured rows to a CSV for full auditability.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass
class DecisionLogger:
    """Appends one structured row per evaluated bar/decision to a CSV.

    This is the audit trail required by the spec: features used, model output,
    risk verdict and the final action are all recorded.
    """

    path: Path
    fieldnames: Sequence[str] = field(
        default_factory=lambda: [
            "timestamp",
            "decision_symbol",
            "direction",
            "confidence",
            "expected_return",
            "candidate_symbol",
            "risk_ok",
            "risk_reason",
            "action",
            "detail",
        ]
    )
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)

    def log(self, **kwargs: Any) -> None:
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        self._rows.append(row)

    def flush(self) -> None:
        if not self._rows:
            return
        os.makedirs(self.path.parent, exist_ok=True)
        with open(self.path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.fieldnames))
            writer.writeheader()
            writer.writerows(self._rows)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)
