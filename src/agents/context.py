"""Agent market-context generation.

Builds the observable-only snapshot handed to an agent at a single bar. It reads
exclusively from the current feature-matrix row (causal features) plus the
current portfolio/risk state — **no future data ever enters**. A dedicated test
(`tests/test_agent_context_no_leak.py`) asserts the context at bar *i* is
identical whether built from the full matrix or the matrix truncated at *i*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..features.builder import PX_PREFIX, FEAT_PREFIX


def _clean(v: Any) -> Any:
    """JSON-safe scalar: NaN/inf -> None, numpy -> python float."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _get(row: pd.Series, col: str) -> Any:
    return _clean(row[col]) if col in row.index else None


@dataclass(frozen=True)
class ContextInputs:
    """Everything the engine knows at a bar, fed to ``agent.build_context``."""

    timestamp: pd.Timestamp
    row: pd.Series
    symbols: list[str]
    positions: list[dict[str, Any]] = field(default_factory=list)
    daily_pnl: float = 0.0
    risk_state: dict[str, Any] = field(default_factory=dict)
    market_session: str = "regular"


def _symbol_block(row: pd.Series, sym: str) -> dict[str, Any]:
    f = f"{FEAT_PREFIX}{sym}__"
    p = f"{PX_PREFIX}{sym}__"
    return {
        "open": _get(row, f"{p}open"),
        "high": _get(row, f"{p}high"),
        "low": _get(row, f"{p}low"),
        "close": _get(row, f"{p}close"),
        "volume": _get(row, f"{p}volume"),
        "vwap": _get(row, f"{p}vwap"),
        "rsi": _get(row, f"{f}rsi_14"),
        "atr": _get(row, f"{f}atr_pct"),
        "returns": {
            "1_bar": _get(row, f"{f}ret_1"),
            "3_bar": _get(row, f"{f}ret_3"),
            "6_bar": _get(row, f"{f}ret_6"),
            "12_bar": _get(row, f"{f}ret_12"),
        },
    }


def build_agent_context(inputs: ContextInputs) -> dict[str, Any]:
    """Assemble the standard observable context dict for an agent."""
    symbols = {sym: _symbol_block(inputs.row, sym) for sym in inputs.symbols}
    return {
        "timestamp": inputs.timestamp.isoformat(),
        "market_session": inputs.market_session,
        "symbols": symbols,
        "positions": list(inputs.positions),
        "daily_pnl": _clean(inputs.daily_pnl),
        "risk_state": dict(inputs.risk_state),
    }
