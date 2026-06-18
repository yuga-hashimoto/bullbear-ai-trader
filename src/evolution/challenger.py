"""Challenger model: a candidate variation under evaluation.

A challenger holds a ``config_patch`` (dot-keyed overrides like
``{"risk.confidence_threshold": 0.7}``) plus lifecycle status. It is evaluated
in *shadow* (virtual fills, no capital), may graduate to *canary* (small
allocation), and can be auto-promoted to Champion only via the promotion policy.
"""
from __future__ import annotations

import dataclasses
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# Evidence lifecycle states. Historical backtests and forward shadow are
# intentionally distinct so a candidate cannot relabel reused data as paper
# evidence.
DRAFT = "DRAFT"
BACKTEST_PASSED = "BACKTEST_PASSED"
SEALED_OOS_PASSED = "SEALED_OOS_PASSED"
SHADOW = "SHADOW"
PAPER = "PAPER"
LIVE_ELIGIBLE = "LIVE_ELIGIBLE"
CHAMPION = "CHAMPION"
# Backward-compatible names for older CLI/report code.
CANARY = PAPER
PROMOTED = CHAMPION
REJECTED = "REJECTED"
ROLLED_BACK = "ROLLED_BACK"
VALID_STATUSES = {
    DRAFT, BACKTEST_PASSED, SEALED_OOS_PASSED, SHADOW, PAPER,
    LIVE_ELIGIBLE, CHAMPION, REJECTED, ROLLED_BACK,
}

VALID_SOURCES = {"learning_loop", "mutation", "manual", "agent_suggestion"}


@dataclass
class Challenger:
    challenger_id: str
    created_at: str
    source: str
    config_patch: dict[str, Any] = field(default_factory=dict)
    agent_prompt_version: str = ""
    risk_policy_version: str = ""
    status: str = DRAFT
    allocation_pct: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    trial_count: int = 0
    notes: str = ""

    @classmethod
    def create(cls, config_patch: dict[str, Any], source: str = "mutation",
               agent_prompt_version: str = "", risk_policy_version: str = "",
               notes: str = "") -> "Challenger":
        if source not in VALID_SOURCES:
            raise ValueError(f"invalid source: {source}")
        return cls(
            challenger_id="chal_" + uuid.uuid4().hex[:10],
            created_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            config_patch=dict(config_patch),
            agent_prompt_version=agent_prompt_version,
            risk_policy_version=risk_policy_version,
            status=DRAFT,
            allocation_pct=0.0,
            metrics={},
            evidence={},
            trial_count=0,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Challenger":
        values = {}
        for name, info in cls.__dataclass_fields__.items():
            if name in d:
                values[name] = d[name]
            elif info.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
                values[name] = info.default_factory()
            else:
                values[name] = info.default
        return cls(**values)
