"""Broker abstraction.

A single interface for every execution venue. Backtest and paper brokers are
fully simulated and safe. The moomoo broker is a future-use skeleton that
refuses to place real orders unless the triple safety gate is satisfied.

Order placement uses long-only semantics here (buy/sell to flat); bearish
exposure is taken via inverse ETFs, never by shorting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    status: OrderStatus = OrderStatus.NEW
    fill_price: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    quantity: float
    avg_price: float


class BrokerBase(ABC):
    """Common broker interface (see module docstring)."""

    @abstractmethod
    def get_positions(self) -> list[PositionInfo]: ...

    @abstractmethod
    def get_cash(self) -> float: ...

    @abstractmethod
    def get_market_data(self, symbol: str) -> dict[str, float]: ...

    @abstractmethod
    def submit_order(self, symbol: str, side: OrderSide, quantity: float, **kw) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order: ...

    @abstractmethod
    def close_position(self, symbol: str) -> Order | None: ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order: ...
