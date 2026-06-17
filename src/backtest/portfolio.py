"""Portfolio primitives shared by the risk engine and backtest loop."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd


@dataclass(frozen=True)
class Position:
    """An open long position in a tradable ETF.

    All ETFs are bought *long* — bearish views are expressed by buying an
    inverse ETF (SQQQ/SOXS), never by shorting. peak_price tracks the best
    price seen for the trailing stop.
    """

    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    shares: float
    entry_bar: int
    peak_price: float
    trade_id: int = 0
    entry_reason: str = ""
    entry_commission: float = 0.0

    def update_peak(self, price: float) -> "Position":
        return replace(self, peak_price=max(self.peak_price, price))

    def unrealized_pct(self, price: float) -> float:
        """Return % move from entry (positive = profit for a long)."""
        return (price - self.entry_price) / self.entry_price


@dataclass(frozen=True)
class ClosedTrade:
    trade_id: int
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    holding_minutes: float
    entry_reason: str
    exit_reason: str
