"""Evaluate a config (champion or challenger patch) by backtest.

Shadow evaluation == running the patched config through the existing
``BacktestEngine`` on the SAME feature matrix, with the SAME cost/slippage/spread
assumptions. No capital is used; this is virtual evaluation.

Also provides a robustness / out-of-sample check used by the promotion policy:
the test window is split in halves; an edge that only appears in one half is
flagged as higher overfitting risk.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from ..agents.factory import make_agent
from ..backtest.engine import BacktestEngine
from ..backtest.metrics import compute_metrics
from ..config.settings import Config
from ..data.store import load_features
from ..pipeline import _INTERVAL_MIN, _slice
from .champion import apply_patch


def expectancy_from_metrics(m: dict[str, Any]) -> float:
    """Per-trade expectancy in dollars: p*avg_win - (1-p)*avg_loss."""
    p = m.get("win_rate_pct", 0.0) / 100.0
    return p * m.get("avg_win", 0.0) - (1.0 - p) * m.get("avg_loss", 0.0)


def evaluate(
    base_cfg: Config,
    patch: dict[str, Any] | None,
    agent_type: str | None = None,
    signal_file: str | None = None,
    matrix: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Backtest the (patched) config; return metrics enriched with expectancy."""
    cfg = apply_patch(base_cfg, patch or {})
    if matrix is None:
        matrix = load_features(cfg)
    test = _slice(matrix, start or cfg.test_start, end or cfg.test_end, cfg.timezone)
    if test.empty:
        return {"num_trades": 0, "error": "empty_test_slice"}
    agent = make_agent(cfg, agent_type or cfg.agent.type, signal_file)
    result = BacktestEngine(cfg, agent).run(test)
    m = compute_metrics(result, _INTERVAL_MIN[cfg.interval])
    m["expectancy"] = round(expectancy_from_metrics(m), 4)
    m["net_pnl_after_costs"] = round(m["final_equity"] - m["initial_cash"], 2)
    return m


def robustness_check(
    base_cfg: Config,
    patch: dict[str, Any] | None,
    agent_type: str | None = None,
    signal_file: str | None = None,
    matrix: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Split the test window in halves; grade overfitting risk + OOS pass.

    LOW   : both halves have profit_factor >= 1
    MEDIUM: exactly one half does
    HIGH  : neither half does
    out_of_sample pass == second (later) half profit_factor >= 1
    """
    cfg = apply_patch(base_cfg, patch or {})
    if matrix is None:
        matrix = load_features(cfg)
    lo = pd.Timestamp(cfg.test_start, tz=cfg.timezone)
    hi = pd.Timestamp(cfg.test_end, tz=cfg.timezone)
    mid = lo + (hi - lo) / 2
    mid_str = mid.date().isoformat()

    first = evaluate(base_cfg, patch, agent_type, signal_file, matrix, cfg.test_start, mid_str)
    second = evaluate(base_cfg, patch, agent_type, signal_file, matrix, mid_str, cfg.test_end)
    pf1 = first.get("profit_factor", 0.0)
    pf2 = second.get("profit_factor", 0.0)
    good = (pf1 >= 1.0) + (pf2 >= 1.0)
    risk = {2: "LOW", 1: "MEDIUM", 0: "HIGH"}[good]
    return {
        "overfitting_risk": risk,
        "out_of_sample_pass": pf2 >= 1.0,
        "first_half_pf": pf1,
        "second_half_pf": pf2,
    }
