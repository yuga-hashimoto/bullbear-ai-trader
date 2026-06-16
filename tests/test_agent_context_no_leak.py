"""Agent context must contain only observable (past/current) data.

Property: the context built at bar ``i`` from the full matrix equals the context
built from the matrix truncated at ``i``. If any future data leaked into the
context, truncation would change it.
"""
from __future__ import annotations

from src.agents.context import ContextInputs, build_agent_context
from src.features.builder import feature_columns


def _ctx_at(matrix, i, symbols):
    row = matrix.iloc[i]
    t = matrix.index[i]
    return build_agent_context(ContextInputs(timestamp=t, row=row, symbols=symbols))


def test_context_is_causal(cfg, labeled_matrix):
    matrix = labeled_matrix.dropna(subset=feature_columns(labeled_matrix))
    symbols = [s for s in cfg.all_symbols if s in ("QQQ", "SMH", "SPY", "TQQQ")]
    assert len(matrix) > 130
    for i in (60, 100, 130):
        full = _ctx_at(matrix, i, symbols)
        truncated = _ctx_at(matrix.iloc[: i + 1], i, symbols)
        assert full == truncated, f"context leaked future info at bar {i}"


def test_context_shape_has_required_fields(cfg, labeled_matrix):
    matrix = labeled_matrix.dropna(subset=feature_columns(labeled_matrix))
    ctx = _ctx_at(matrix, 100, ["QQQ"])
    assert set(ctx) >= {"timestamp", "market_session", "symbols", "positions",
                        "daily_pnl", "risk_state"}
    qqq = ctx["symbols"]["QQQ"]
    assert set(qqq) >= {"open", "high", "low", "close", "volume", "vwap",
                        "rsi", "atr", "returns"}
    assert set(qqq["returns"]) == {"1_bar", "3_bar", "6_bar", "12_bar"}
