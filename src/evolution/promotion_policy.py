"""Auto-promotion policy evaluation (no human approval, but policy-gated).

A challenger is promotable ONLY if every condition holds. Win rate alone is
never enough — drawdown, profit factor, expectancy, trade count, robustness and
out-of-sample behavior are all required. Live promotion is disabled by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

_DEFAULTS = {
    "enabled": True,
    "environment_allowed": {"paper": True, "live": False},
    "min_shadow_days": 10,
    "min_canary_days": 5,
    "min_trades": 100,
    "min_paper_days": 183,
    "min_profit_factor": 1.25,
    "min_sharpe_ratio": 1.0,
    "min_win_rate_delta_vs_champion": 0.03,
    "min_expectancy_delta_vs_champion": 0.001,
    "max_drawdown_not_worse_than_champion": True,
    "max_worst_day_pct": 2.0,
    "require_positive_net_pnl_after_costs": True,
    "require_robustness_check": True,
    "require_out_of_sample_pass": True,
    "max_overfitting_risk": "MEDIUM",
    "allow_auto_promote_to_champion": True,
}


def load_promotion_policy(path: str | Path = "config/promotion_policy.yaml") -> dict[str, Any]:
    p = Path(path)
    policy = dict(_DEFAULTS)
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
        policy.update(raw.get("auto_promotion", {}))
    return policy


@dataclass
class PromotionResult:
    passed: bool
    reasons: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def evaluate_promotion(
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    *,
    env: str,
    days_shadow: int,
    days_canary: int,
    policy: dict[str, Any],
    robustness: dict[str, Any] | None = None,
) -> PromotionResult:
    robustness = robustness or {}
    c, ch = champion_metrics, challenger_metrics
    reasons: dict[str, bool] = {}

    reasons["enabled"] = bool(policy.get("enabled"))
    reasons["allow_auto_promote"] = bool(policy.get("allow_auto_promote_to_champion"))
    reasons["environment_allowed"] = bool(policy.get("environment_allowed", {}).get(env, False))
    reasons["min_shadow_days"] = days_shadow >= policy["min_shadow_days"]
    reasons["min_canary_days"] = days_canary >= policy["min_canary_days"]
    reasons["min_paper_days"] = ch.get("paper_days", days_canary) >= policy["min_paper_days"]
    reasons["min_trades"] = ch.get("num_trades", 0) >= policy["min_trades"]
    reasons["min_profit_factor"] = ch.get("profit_factor", 0.0) >= policy["min_profit_factor"]
    reasons["min_sharpe"] = ch.get("sharpe_ratio", 0.0) >= policy["min_sharpe_ratio"]
    reasons["recent_3m_positive"] = ch.get("recent_3m_net_pnl", 0.0) > 0
    reasons["sealed_oos_pass"] = bool(ch.get("sealed_oos_pass", False))
    reasons["forward_shadow_pass"] = bool(ch.get("forward_shadow_pass", False))

    wr_delta = (ch.get("win_rate_pct", 0.0) - c.get("win_rate_pct", 0.0)) / 100.0
    reasons["win_rate_delta"] = wr_delta >= policy["min_win_rate_delta_vs_champion"]

    exp_delta = ch.get("expectancy", 0.0) - c.get("expectancy", 0.0)
    reasons["expectancy_delta"] = exp_delta >= policy["min_expectancy_delta_vs_champion"]

    if policy.get("max_drawdown_not_worse_than_champion", True):
        # drawdowns are negative; challenger must be >= champion (less negative).
        reasons["drawdown_not_worse"] = ch.get("max_drawdown_pct", -999) >= c.get("max_drawdown_pct", -999)
    if policy.get("require_positive_net_pnl_after_costs", True):
        reasons["positive_net_pnl"] = ch.get("net_pnl_after_costs", 0.0) > 0
    reasons["worst_day_ok"] = ch.get("worst_day_pct", -999) >= -abs(policy["max_worst_day_pct"])

    if policy.get("require_robustness_check", True):
        risk = robustness.get("overfitting_risk", "HIGH")
        reasons["robustness_ok"] = _RISK_ORDER.get(risk, 2) <= _RISK_ORDER.get(
            policy.get("max_overfitting_risk", "MEDIUM"), 1)
    if policy.get("require_out_of_sample_pass", True):
        reasons["out_of_sample_pass"] = bool(robustness.get("out_of_sample_pass", False))

    passed = all(reasons.values())
    return PromotionResult(passed=passed, reasons=reasons, detail={
        "win_rate_delta": round(wr_delta, 4),
        "expectancy_delta": round(exp_delta, 6),
        "robustness": robustness,
    })
