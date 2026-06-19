"""Guardrails for challenger config patches / mutations.

These rules are the safety authority for auto-evolution. A patch is rejected if
it would:
  * touch any section other than risk/strategy/agent (so costs, the live gate,
    the market calendar, the runner, etc. can NEVER be changed by evolution);
  * loosen a hard safety limit (e.g. raise the allowed daily loss);
  * push a tunable parameter outside sane bounds (no extreme values);
  * attempt to disable a stop-loss / take-profit.

Auto-promotion applies a patch only after it passes ``check_patch``; the Risk
Engine itself is never bypassed.
"""
from __future__ import annotations

from typing import Any

from ..config.settings import Config

PATCHABLE_SECTIONS = {"risk", "strategy", "agent"}

# Inclusive [min, max] bounds for tunable parameters.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "risk.confidence_threshold": (0.40, 0.90),
    "risk.max_loss_per_trade_pct": (0.30, 1.20),
    "risk.take_profit_pct": (0.40, 3.00),
    "risk.trailing_stop_pct": (0.20, 1.50),
    "risk.max_holding_minutes": (15, 240),
    "risk.no_trade_first_minutes": (0, 60),
    "risk.no_new_entry_last_minutes": (10, 90),
    "risk.max_trades_per_day": (1, 10),
    "risk.max_consecutive_losses": (1, 5),
    "strategy.max_concurrent_positions": (1, 2),
    "strategy.expected_return_weight": (0.0, 5.0),
    # V9 entry genes — tight neighbourhood around the validated base so
    # challengers stay "V9 derivatives" rather than wild variants.
    "strategy.numeric_min_vwap_dev": (0.0, 0.0008),
    "strategy.numeric_min_strength": (0.0004, 0.0018),
    "strategy.numeric_rsi_bull_max": (65.0, 80.0),
    "strategy.numeric_rsi_bear_min": (20.0, 35.0),
}

# Fields that may only be *tightened* vs the base config (never loosened).
# For an "allowed loss" limit, tightening means a smaller value.
TIGHTEN_ONLY = {"risk.max_daily_loss_pct"}

# Fields that must stay strictly positive (cannot disable the protection).
MUST_BE_POSITIVE = {"risk.max_loss_per_trade_pct", "risk.take_profit_pct"}


class GuardrailViolation(ValueError):
    pass


def check_patch(patch: dict[str, Any], base_cfg: Config) -> list[str]:
    """Return a list of guardrail violations (empty == safe)."""
    violations: list[str] = []
    for key, value in patch.items():
        if "." not in key:
            violations.append(f"{key}: must be 'section.field'")
            continue
        section, field_name = key.split(".", 1)
        if section not in PATCHABLE_SECTIONS:
            violations.append(f"{key}: section '{section}' is not patchable by evolution")
            continue
        # numeric coercion for bound checks
        if key in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[key]
            try:
                v = float(value)
            except (TypeError, ValueError):
                violations.append(f"{key}: non-numeric value {value!r}")
                continue
            if not (lo <= v <= hi):
                violations.append(f"{key}={value} out of bounds [{lo}, {hi}]")
        if key in MUST_BE_POSITIVE:
            try:
                if float(value) <= 0:
                    violations.append(f"{key}={value} must be > 0 (cannot disable protection)")
            except (TypeError, ValueError):
                pass
        if key in TIGHTEN_ONLY:
            base_val = getattr(getattr(base_cfg, section), field_name)
            try:
                if float(value) > float(base_val):
                    violations.append(
                        f"{key}={value} loosens safety limit (base={base_val}); not allowed")
            except (TypeError, ValueError):
                pass
    return violations


def is_safe(patch: dict[str, Any], base_cfg: Config) -> bool:
    return not check_patch(patch, base_cfg)


def assert_safe(patch: dict[str, Any], base_cfg: Config) -> None:
    v = check_patch(patch, base_cfg)
    if v:
        raise GuardrailViolation("; ".join(v))
