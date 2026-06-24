"""RuleStrategyAgent: wraps deterministic research strategies as Signal JSON.

This agent lets the normal BacktestEngine/RiskEngine evaluate classic rule
strategies without bypassing any safety checks.  It is useful for baselines,
strategy sweeps and Champion/Challenger seeding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ..config.settings import Config
from ..features.builder import price_col
from ..research.technical_strategies import generate_signal_frame, strategy_names
from .base import BaseAgent
from .signal_schema import FAMILY_BEAR, FAMILY_BULL


@dataclass(frozen=True)
class _Candidate:
    family: str
    direction: str
    confidence: float
    reason: str
    score: float
    strategy: str


_DECISION_BY_FAMILY = {
    "NASDAQ": "QQQ",
    "SEMICONDUCTOR": "SMH",
}
_SYMBOL_TO_FAMILY = {
    "TQQQ": "NASDAQ",
    "SQQQ": "NASDAQ",
    "SOXL": "SEMICONDUCTOR",
    "SOXS": "SEMICONDUCTOR",
}


class RuleStrategyAgent(BaseAgent):
    """Deterministic strategy agent driven by cfg.raw['rule_agent']."""

    name = "RuleStrategyAgent"
    version = "1.0.0"

    def __init__(
        self,
        cfg: Config,
        strategy_name: str | None = None,
        family: str | None = None,
        params: Mapping[str, float | int] | None = None,
    ) -> None:
        raw = dict(cfg.raw.get("rule_agent", {}) or {})
        self.strategy_name = str(strategy_name or raw.get("strategy", "sma_cross")).lower()
        if self.strategy_name not in strategy_names():
            raise ValueError(f"unknown rule strategy {self.strategy_name!r}; available={strategy_names()}")
        self.family = str(family or raw.get("family", "auto")).upper()
        if self.family not in {"AUTO", "NASDAQ", "SEMICONDUCTOR"}:
            raise ValueError("rule_agent.family must be auto, NASDAQ or SEMICONDUCTOR")
        base_params = dict(raw.get("params", {}) or {})
        base_params.update(dict(params or {}))
        self.params = base_params
        self.expected_holding_minutes = int(raw.get("expected_holding_minutes", 60))
        self._by_timestamp: dict[str, list[_Candidate]] = {}

    def prepare(self, matrix: pd.DataFrame, feat_cols: list[str]) -> None:  # noqa: ARG002
        self._by_timestamp = {}
        families = [self.family] if self.family != "AUTO" else ["NASDAQ", "SEMICONDUCTOR"]
        for family in families:
            decision_symbol = _DECISION_BY_FAMILY[family]
            try:
                ohlcv = _ohlcv_from_matrix(matrix, decision_symbol)
            except KeyError:
                continue
            signals = generate_signal_frame(ohlcv, self.strategy_name, self.params)
            for ts, row in signals.iterrows():
                candidate = _Candidate(
                    family=family,
                    direction=str(row["direction"]),
                    confidence=float(row["confidence"]),
                    reason=str(row["reason"]),
                    score=float(row["score"]),
                    strategy=str(row["strategy"]),
                )
                self._by_timestamp.setdefault(ts.isoformat(), []).append(candidate)

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        timestamp = str(context["timestamp"])
        candidates = self._by_timestamp.get(timestamp, [])
        selected = _select_candidate(candidates)
        positions = list(context.get("positions", []) or [])

        if selected is None or selected.direction == "FLAT":
            if positions:
                return self._exit_signal(timestamp, positions[0], selected, "flat_or_no_candidate")
            return self._no_trade(timestamp, selected, "flat_or_no_candidate")

        action, symbol = _action_and_symbol(selected.family, selected.direction)
        if positions:
            current_symbol = str(positions[0].get("symbol", ""))
            if current_symbol != symbol:
                return self._exit_signal(timestamp, positions[0], selected, f"rotate_to_{symbol}")
            return self._no_trade(timestamp, selected, "already_holding_target")

        return {
            "timestamp": timestamp,
            "agent_name": self.name,
            "agent_version": self.version,
            "target_family": selected.family,
            "direction": selected.direction,
            "action": action,
            "symbol": symbol,
            "confidence": selected.confidence,
            "expected_holding_minutes": self.expected_holding_minutes,
            "reason": selected.reason,
            "risk_notes": ["rule strategy baseline; RiskEngine remains authoritative"],
            "features_used": {
                "strategy": selected.strategy,
                "score": selected.score,
                "family_mode": self.family,
                "params": self.params,
            },
            "raw_response": {},
        }

    def _no_trade(self, timestamp: str, selected: _Candidate | None, reason: str) -> dict[str, Any]:
        return {
            "timestamp": timestamp,
            "agent_name": self.name,
            "agent_version": self.version,
            "target_family": "MARKET",
            "direction": "FLAT",
            "action": "NO_TRADE",
            "symbol": None,
            "confidence": 0.0,
            "expected_holding_minutes": 0,
            "reason": reason if selected is None else f"{reason}: {selected.reason}",
            "risk_notes": [],
            "features_used": {"strategy": self.strategy_name, "family_mode": self.family},
            "raw_response": {},
        }

    def _exit_signal(
        self,
        timestamp: str,
        position: dict[str, Any],
        selected: _Candidate | None,
        reason: str,
    ) -> dict[str, Any]:
        symbol = str(position.get("symbol", ""))
        family = _SYMBOL_TO_FAMILY.get(symbol, "MARKET")
        confidence = max(float(selected.confidence) if selected else 0.0, 0.66)
        return {
            "timestamp": timestamp,
            "agent_name": self.name,
            "agent_version": self.version,
            "target_family": family,
            "direction": selected.direction if selected else "FLAT",
            "action": "EXIT",
            "symbol": symbol if symbol else None,
            "confidence": min(confidence, 0.98),
            "expected_holding_minutes": 0,
            "reason": f"rule_exit:{reason}",
            "risk_notes": ["exit request still flows through engine execution timing"],
            "features_used": {"strategy": self.strategy_name, "family_mode": self.family},
            "raw_response": {},
        }


def _ohlcv_from_matrix(matrix: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = {field: price_col(symbol, field) for field in ["open", "high", "low", "close", "volume"]}
    missing = [col for col in cols.values() if col not in matrix.columns]
    if missing:
        raise KeyError(f"matrix missing price columns for {symbol}: {missing}")
    out = pd.DataFrame({field: matrix[col] for field, col in cols.items()}, index=matrix.index)
    return out.dropna(subset=["open", "high", "low", "close"])


def _select_candidate(candidates: list[_Candidate]) -> _Candidate | None:
    actionable = [c for c in candidates if c.direction != "FLAT" and c.confidence > 0]
    if actionable:
        return max(actionable, key=lambda c: (c.confidence, abs(c.score)))
    if candidates:
        return max(candidates, key=lambda c: (c.confidence, abs(c.score)))
    return None


def _action_and_symbol(family: str, direction: str) -> tuple[str, str]:
    if direction == "UP":
        return "BUY_BULL", FAMILY_BULL[family]
    if direction == "DOWN":
        return "BUY_BEAR", FAMILY_BEAR[family]
    raise ValueError(f"cannot map direction to action: {direction}")
