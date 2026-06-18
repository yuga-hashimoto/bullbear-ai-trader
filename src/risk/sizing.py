"""Deterministic position sizing from explicit loss budgets."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class PositionSizingInput:
    entry_price_usd: float
    stop_price_usd: float | None
    cash_usd: float
    usd_jpy: float
    max_trade_loss_jpy: float
    portfolio_risk_remaining_jpy: float
    overnight_gap_pct: float = 0.0


@dataclass(frozen=True)
class PositionSizingResult:
    quantity: int
    planned_loss_jpy: float
    risk_per_share_jpy: float
    reason: str = "ok"


class PositionSizer:
    def size(self, item: PositionSizingInput) -> PositionSizingResult:
        if item.stop_price_usd is None:
            return PositionSizingResult(0, 0.0, 0.0, "missing_stop")
        if item.entry_price_usd <= 0 or item.usd_jpy <= 0:
            return PositionSizingResult(0, 0.0, 0.0, "invalid_price_or_fx")
        stop_distance = item.entry_price_usd - item.stop_price_usd
        if stop_distance <= 0:
            return PositionSizingResult(0, 0.0, 0.0, "invalid_stop")

        gap_distance = item.entry_price_usd * max(item.overnight_gap_pct, 0.0) / 100.0
        risk_per_share_jpy = max(stop_distance, gap_distance) * item.usd_jpy
        risk_budget = min(item.max_trade_loss_jpy, item.portfolio_risk_remaining_jpy)
        if risk_budget <= 0:
            return PositionSizingResult(0, 0.0, risk_per_share_jpy, "risk_budget_exhausted")

        by_risk = floor(risk_budget / risk_per_share_jpy)
        by_cash = floor(item.cash_usd / item.entry_price_usd)
        quantity = max(min(by_risk, by_cash), 0)
        if quantity <= 0:
            return PositionSizingResult(0, 0.0, risk_per_share_jpy, "insufficient_cash_or_risk")
        planned = round(quantity * risk_per_share_jpy, 2)
        return PositionSizingResult(quantity, planned, risk_per_share_jpy)
