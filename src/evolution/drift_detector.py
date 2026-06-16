"""Drift / regime-change detection.

Compares recent behavior against a baseline and raises alerts. On drift the
caller should reduce new entries / allocation and increase challenger
exploration (handled in the evolution loop).
"""
from __future__ import annotations

from typing import Any

WIN_RATE_DROP_PP = 10.0     # percentage points
DRAWDOWN_ACCEL_PP = 5.0
VOL_RATIO_THRESHOLD = 1.5


def detect_drift(baseline: dict[str, Any], recent: dict[str, Any],
                 vol_recent: float | None = None, vol_baseline: float | None = None) -> list[dict[str, Any]]:
    """Return a list of drift alerts (each {type, detail})."""
    alerts: list[dict[str, Any]] = []

    if recent.get("win_rate_pct", 0) <= baseline.get("win_rate_pct", 0) - WIN_RATE_DROP_PP:
        alerts.append({"type": "win_rate_degradation",
                       "detail": {"baseline": baseline.get("win_rate_pct"),
                                  "recent": recent.get("win_rate_pct")}})

    if recent.get("expectancy", 0.0) < baseline.get("expectancy", 0.0) and recent.get("expectancy", 0.0) < 0:
        alerts.append({"type": "expectancy_degradation",
                       "detail": {"baseline": baseline.get("expectancy"),
                                  "recent": recent.get("expectancy")}})

    if recent.get("max_drawdown_pct", 0) <= baseline.get("max_drawdown_pct", 0) - DRAWDOWN_ACCEL_PP:
        alerts.append({"type": "drawdown_acceleration",
                       "detail": {"baseline": baseline.get("max_drawdown_pct"),
                                  "recent": recent.get("max_drawdown_pct")}})

    if vol_recent is not None and vol_baseline:
        ratio = vol_recent / vol_baseline if vol_baseline else 0.0
        if ratio >= VOL_RATIO_THRESHOLD or (ratio and ratio <= 1.0 / VOL_RATIO_THRESHOLD):
            alerts.append({"type": "volatility_regime_change",
                           "detail": {"ratio": round(ratio, 3)}})

    return alerts
