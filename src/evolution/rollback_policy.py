"""Auto-rollback policy. Rollback is allowed DURING market hours (immediate)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS = {
    "enabled": True,
    "max_daily_loss_pct": 2.0,
    "max_intraday_drawdown_pct": 2.5,
    "max_consecutive_losses": 3,
    "max_underperformance_vs_fallback_pct": 1.0,
    "max_agent_errors_per_day": 3,
    "max_data_errors_per_day": 3,
    "rollback_to_previous_champion": True,
    "freeze_promotions_after_rollback_days": 3,
}


def load_rollback_policy(path: str | Path = "config/rollback_policy.yaml") -> dict[str, Any]:
    p = Path(path)
    policy = dict(_DEFAULTS)
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
        policy.update(raw.get("auto_rollback", {}))
    return policy


@dataclass
class RollbackResult:
    should_rollback: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_rollback(state: dict[str, Any], policy: dict[str, Any]) -> RollbackResult:
    """``state`` carries the live/paper degradation signals (positive numbers)."""
    if not policy.get("enabled", True):
        return RollbackResult(False, [])
    reasons: list[str] = []
    if state.get("daily_loss_pct", 0.0) >= policy["max_daily_loss_pct"]:
        reasons.append("max_daily_loss")
    if state.get("intraday_drawdown_pct", 0.0) >= policy["max_intraday_drawdown_pct"]:
        reasons.append("max_intraday_drawdown")
    if state.get("consecutive_losses", 0) >= policy["max_consecutive_losses"]:
        reasons.append("max_consecutive_losses")
    if state.get("underperformance_vs_fallback_pct", 0.0) >= policy["max_underperformance_vs_fallback_pct"]:
        reasons.append("underperformance_vs_fallback")
    if state.get("agent_errors", 0) >= policy["max_agent_errors_per_day"]:
        reasons.append("agent_errors")
    if state.get("data_errors", 0) >= policy["max_data_errors_per_day"]:
        reasons.append("data_errors")
    return RollbackResult(bool(reasons), reasons)
