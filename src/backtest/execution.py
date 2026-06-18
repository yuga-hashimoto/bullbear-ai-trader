"""Realistic fill modelling: commission, spread and slippage.

Fills are deliberately *pessimistic*. A market buy pays up (ask + slippage); a
market sell receives less (bid - slippage). This prevents the backtest from
crediting the strategy with prices it could never achieve, which is one of the
most common sources of fake edge.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config.settings import CostConfig


@dataclass(frozen=True)
class Fill:
    price: float       # effective fill price after spread + slippage
    shares: float
    commission: float
    notional: float    # shares * fill price (before commission)


class ExecutionModel:
    def __init__(self, costs: CostConfig) -> None:
        self.c = costs

    def _adverse(self, ref_price: float, side: str) -> float:
        """Apply half-spread + slippage against the trader."""
        adj = (self.c.spread_pct + self.c.slippage_pct) / 100.0
        if side == "buy":
            return ref_price * (1.0 + adj)
        return ref_price * (1.0 - adj)

    def _commission(self, shares: float, notional: float) -> float:
        comm = self.c.commission_pct * notional + self.c.commission_per_share * shares
        return max(comm, self.c.min_commission)

    def fill_buy(self, ref_price: float, cash_to_deploy: float) -> Fill:
        price = self._adverse(ref_price, "buy")
        if price <= 0:
            return Fill(price, 0.0, 0.0, 0.0)
        # Reserve room for commission so we don't overspend.
        approx_shares = cash_to_deploy / (price * (1.0 + self.c.commission_pct))
        shares = float(int(approx_shares))  # whole shares only
        if shares <= 0:
            return Fill(price, 0.0, 0.0, 0.0)
        notional = shares * price
        commission = self._commission(shares, notional)
        return Fill(price, shares, commission, notional)

    def fill_buy_quantity(self, ref_price: float, shares: float) -> Fill:
        price = self._adverse(ref_price, "buy")
        if price <= 0 or shares <= 0:
            return Fill(price, 0.0, 0.0, 0.0)
        shares = float(int(shares))
        notional = shares * price
        commission = self._commission(shares, notional)
        return Fill(price, shares, commission, notional)

    def fill_sell(self, ref_price: float, shares: float) -> Fill:
        price = self._adverse(ref_price, "sell")
        notional = shares * price
        commission = self._commission(shares, notional)
        return Fill(price, shares, commission, notional)
