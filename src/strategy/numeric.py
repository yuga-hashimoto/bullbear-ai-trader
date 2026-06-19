"""Deterministic numeric candidate generation from observable market context.

Entry edge (validated on 2026-04..06 5m data, train/test split, profit factor
~1.2 / Sharpe ~2.2 on both halves — see scripts/edge_lab.py "V9"):

  Trade only with the trend fully aligned across timeframes. Go long an ETF when
  price is above session VWAP and the 1/3/6/12-bar returns are ALL positive
  (fresh, sustained momentum), the 3-bar move clears a noise threshold, and RSI
  is not already overbought. Symmetric for shorts (inverse ETF). This filters
  out the weak, mean-reverting blips that made the naive "close>vwap and rising"
  rule lose money (profit factor < 0.5).
"""
from __future__ import annotations

from typing import Any

from ..agents.signal_schema import FAMILY_BEAR, FAMILY_BULL, no_trade_signal


class NumericSignalStrategy:
    def __init__(
        self,
        base_confidence: float = 0.66,
        min_vwap_dev: float = 0.0003,
        min_strength: float = 0.0008,
        rsi_bull_max: float = 72.0,
        rsi_bear_min: float = 28.0,
    ) -> None:
        self.base_confidence = base_confidence
        self.min_vwap_dev = min_vwap_dev          # min |close-VWAP|/VWAP to act
        self.min_strength = min_strength          # min |3-bar return|
        self.rsi_bull_max = rsi_bull_max          # skip longs once overbought
        self.rsi_bear_min = rsi_bear_min          # skip shorts once oversold

    def _candidate(self, block: dict[str, Any] | None) -> tuple[str, float] | None:
        if not block:
            return None
        close = block.get("close")
        vwap = block.get("vwap")
        rsi = block.get("rsi")
        rets = block.get("returns") or {}
        r1, r3 = rets.get("1_bar"), rets.get("3_bar")
        r6, r12 = rets.get("6_bar"), rets.get("12_bar")
        if None in (close, vwap, rsi, r1, r3, r6, r12) or not vwap:
            return None

        r1, r3, r6, r12 = float(r1), float(r3), float(r6), float(r12)
        rsi = float(rsi)
        vwap_dev = (float(close) - float(vwap)) / float(vwap)
        if abs(r3) < self.min_strength:
            return None

        bull = (close > vwap and vwap_dev >= self.min_vwap_dev
                and r1 > 0 and r3 > 0 and r6 > 0 and r12 > 0
                and rsi < self.rsi_bull_max)
        if bull:
            return "BULL", abs(r3)
        bear = (close < vwap and -vwap_dev >= self.min_vwap_dev
                and r1 < 0 and r3 < 0 and r6 < 0 and r12 < 0
                and rsi > self.rsi_bear_min)
        if bear:
            return "BEAR", abs(r3)
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
            "agent_version": "2.0.0",
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
