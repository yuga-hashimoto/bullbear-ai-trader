"""In-memory simulated brokers (paper / backtest).

PaperBroker fills market orders immediately at the last known price adjusted by
the execution cost model. It holds no real connection and can never move real
money. BacktestBroker is an alias with the same simulated behavior, provided so
the broker factory has a distinct ``backtest`` entry.
"""
from __future__ import annotations

import itertools

from ..backtest.execution import ExecutionModel
from ..config.settings import CostConfig
from .base import BrokerBase, Order, OrderSide, OrderStatus, PositionInfo

_ID = itertools.count(1)


class PaperBroker(BrokerBase):
    def __init__(self, cash: float = 100000.0, costs: CostConfig | None = None) -> None:
        self._cash = cash
        self._positions: dict[str, PositionInfo] = {}
        self._orders: dict[str, Order] = {}
        self._last_price: dict[str, float] = {}
        self._exec = ExecutionModel(costs or CostConfig())

    # -- market data state (fed by the caller / live loop) ------------------
    def set_price(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = price

    def get_market_data(self, symbol: str) -> dict[str, float]:
        return {"last": self._last_price.get(symbol, 0.0)}

    # -- account ------------------------------------------------------------
    def get_cash(self) -> float:
        return self._cash

    def get_positions(self) -> list[PositionInfo]:
        return list(self._positions.values())

    # -- orders -------------------------------------------------------------
    def submit_order(self, symbol: str, side: OrderSide, quantity: float, **kw) -> Order:
        ref = self._last_price.get(symbol)
        order = Order(order_id=f"paper-{next(_ID)}", symbol=symbol, side=side, quantity=quantity)
        if ref is None or ref <= 0 or quantity <= 0:
            order.status = OrderStatus.REJECTED
            self._orders[order.order_id] = order
            return order

        if side == OrderSide.BUY:
            fill = self._fill_buy_exact(ref, quantity)
            cost = fill.notional + fill.commission
            if fill.shares <= 0 or cost > self._cash:
                order.status = OrderStatus.REJECTED
            else:
                self._cash -= cost
                self._positions[symbol] = PositionInfo(
                    symbol, fill.shares, fill.price, entry_commission=fill.commission
                )
                order.status = OrderStatus.FILLED
                order.fill_price = fill.price
                order.meta.update({"commission": fill.commission, "notional": fill.notional})
        else:  # SELL; long-only, close up to the requested quantity.
            pos = self._positions.get(symbol)
            qty = min(quantity, pos.quantity) if pos else 0.0
            if qty <= 0:
                order.status = OrderStatus.REJECTED
            else:
                fill = self._exec.fill_sell(ref, qty)
                self._cash += fill.notional - fill.commission
                remaining = pos.quantity - qty
                if remaining > 1e-9:
                    self._positions[symbol] = PositionInfo(
                        symbol, remaining, pos.avg_price, entry_commission=pos.entry_commission
                    )
                else:
                    self._positions.pop(symbol, None)
                order.status = OrderStatus.FILLED
                order.fill_price = fill.price
                order.quantity = qty
                order.meta.update({"commission": fill.commission, "notional": fill.notional})
        self._orders[order.order_id] = order
        return order

    def _fill_buy_exact(self, ref_price: float, shares: float):
        price = self._exec._adverse(ref_price, "buy")  # centralized cost model
        notional = shares * price
        commission = self._exec._commission(shares, notional)
        from ..backtest.execution import Fill

        return Fill(price, shares, commission, notional)

    def cancel_order(self, order_id: str) -> Order:
        order = self._orders[order_id]
        if order.status == OrderStatus.NEW:
            order.status = OrderStatus.CANCELLED
        return order

    def close_position(self, symbol: str) -> Order | None:
        pos = self._positions.get(symbol)
        if not pos:
            return None
        return self.submit_order(symbol, OrderSide.SELL, pos.quantity)

    def get_order_status(self, order_id: str) -> Order:
        return self._orders[order_id]


class BacktestBroker(PaperBroker):
    """Simulated broker used in backtests (same semantics as PaperBroker)."""
