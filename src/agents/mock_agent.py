"""MockAgent — a simple rule-based agent for plumbing/backtest verification.

IMPORTANT: MockAgent is NOT designed to be profitable. It exists only to drive
the backtest infrastructure end-to-end and to demonstrate the
context -> signal -> validation -> risk flow. Do not interpret its results as a
strategy.

Rules (per bar, using only observable context):
  * QQQ above VWAP and short return > 0  -> BUY_BULL NASDAQ (TQQQ)
  * QQQ below VWAP and short return < 0  -> BUY_BEAR NASDAQ (SQQQ)
  * SMH above VWAP and short return > 0  -> BUY_BULL SEMICONDUCTOR (SOXL)
  * SMH below VWAP and short return < 0  -> BUY_BEAR SEMICONDUCTOR (SOXS)
  * otherwise / weak                     -> NO_TRADE
The strongest (largest |return|) of the two families wins for the bar.
"""
from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .signal_schema import FAMILY_BEAR, FAMILY_BULL, no_trade_signal


class MockAgent(BaseAgent):
    name = "MockAgent"
    version = "1.0.0"

    def __init__(self, base_confidence: float = 0.66) -> None:
        self.base_confidence = base_confidence

    def _evaluate(self, block: dict[str, Any]) -> tuple[str, float] | None:
        """Return (side, strength) or None if no/weak signal."""
        if block is None:
            return None
        close, vwap = block.get("close"), block.get("vwap")
        ret = block.get("returns", {}).get("3_bar")
        if close is None or vwap is None or ret is None:
            return None
        if close > vwap and ret > 0:
            return ("BULL", abs(ret))
        if close < vwap and ret < 0:
            return ("BEAR", abs(ret))
        return None

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        ts = context["timestamp"]
        candidates: list[tuple[str, str, float]] = []  # (family, side, strength)
        for family, sym_key in (("NASDAQ", "QQQ"), ("SEMICONDUCTOR", "SMH")):
            block = context["symbols"].get(sym_key)
            ev = self._evaluate(block)
            if ev is not None:
                candidates.append((family, ev[0], ev[1]))

        if not candidates:
            return no_trade_signal(ts, self.name, reason="no directional edge").to_dict()

        family, side, strength = max(candidates, key=lambda c: c[2])
        action = "BUY_BULL" if side == "BULL" else "BUY_BEAR"
        symbol = (FAMILY_BULL if side == "BULL" else FAMILY_BEAR)[family]
        confidence = min(0.99, self.base_confidence + 50.0 * strength)
        return {
            "timestamp": ts,
            "agent_name": self.name,
            "agent_version": self.version,
            "target_family": family,
            "direction": "UP" if side == "BULL" else "DOWN",
            "action": action,
            "symbol": symbol,
            "confidence": round(confidence, 4),
            "expected_holding_minutes": 30,
            "reason": f"{symbol}: close vs vwap + 3-bar return rule",
            "risk_notes": [],
            "features_used": {"close_vs_vwap": True, "ret_3_bar": strength},
            "raw_response": {},
        }
