"""Deterministic numeric candidate generation from observable market context."""
from __future__ import annotations

from typing import Any

from ..agents.signal_schema import FAMILY_BEAR, FAMILY_BULL, no_trade_signal


class NumericSignalStrategy:
    def __init__(self, base_confidence: float = 0.66) -> None:
        self.base_confidence = base_confidence

    @staticmethod
    def _candidate(block: dict[str, Any] | None) -> tuple[str, float] | None:
        if not block:
            return None
        close = block.get("close")
        vwap = block.get("vwap")
        ret = block.get("returns", {}).get("3_bar")
        if close is None or vwap is None or ret is None:
            return None
        if close > vwap and ret > 0:
            return "BULL", abs(float(ret))
        if close < vwap and ret < 0:
            return "BEAR", abs(float(ret))
        return None

    def signal(self, context: dict[str, Any], agent_name: str = "NumericAgent") -> dict:
        candidates: list[tuple[str, str, float]] = []
        for family, symbol in (("NASDAQ", "QQQ"), ("SEMICONDUCTOR", "SMH")):
            candidate = self._candidate(context.get("symbols", {}).get(symbol))
            if candidate:
                candidates.append((family, candidate[0], candidate[1]))
        if not candidates:
            return no_trade_signal(
                context["timestamp"], agent_name, "no_numeric_edge"
            ).to_dict()
        family, side, strength = max(candidates, key=lambda item: item[2])
        action = "BUY_BULL" if side == "BULL" else "BUY_BEAR"
        symbol = (FAMILY_BULL if side == "BULL" else FAMILY_BEAR)[family]
        return {
            "timestamp": context["timestamp"],
            "agent_name": agent_name,
            "agent_version": "1.0.0",
            "target_family": family,
            "direction": "UP" if side == "BULL" else "DOWN",
            "action": action,
            "symbol": symbol,
            "confidence": round(min(0.99, self.base_confidence + 50.0 * strength), 4),
            "expected_holding_minutes": 30,
            "reason": "numeric close-vwap and momentum candidate",
            "risk_notes": [],
            "features_used": {"ret_3_bar": strength, "close_vs_vwap": True},
            "raw_response": {},
        }
