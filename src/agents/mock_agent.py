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
from ..strategy.numeric import NumericSignalStrategy


class MockAgent(BaseAgent):
    name = "MockAgent"
    version = "1.0.0"

    def __init__(self, base_confidence: float = 0.66) -> None:
        self.base_confidence = base_confidence
        self.numeric = NumericSignalStrategy(base_confidence)

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.numeric.signal(context, agent_name=self.name)
