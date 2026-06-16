"""moomoo (Futu) live broker — future-use skeleton, hard-disabled.

This class exists so the live path has a concrete shape, but EVERY order method
routes through :func:`assert_live_trading_allowed` and otherwise raises. Even
when the safety gate is satisfied, order placement currently raises
``NotImplementedError`` — the real OpenD wiring is intentionally not built yet.

Activating live trading requires ALL of:
  1. config ``live_trading_enabled: true``
  2. env ``BULLBEAR_ALLOW_LIVE=1``
  3. explicit ``allow_live=True`` passed to the constructor
"""
from __future__ import annotations

from ..config.settings import Config, assert_live_trading_allowed
from .base import BrokerBase, Order, OrderSide, PositionInfo


class MoomooBroker(BrokerBase):
    def __init__(self, cfg: Config, allow_live: bool = False) -> None:
        # Triple safety gate: refuses to construct a live broker otherwise.
        assert_live_trading_allowed(cfg, explicit_flag=allow_live)
        self.cfg = cfg
        self._connected = False

    def _not_built(self) -> "Order":
        raise NotImplementedError(
            "MoomooBroker live trading is not implemented. This is a "
            "future-use skeleton; real order routing via Futu OpenD must be "
            "built and audited before enabling."
        )

    def get_positions(self) -> list[PositionInfo]:
        self._not_built()

    def get_cash(self) -> float:
        self._not_built()

    def get_market_data(self, symbol: str) -> dict[str, float]:
        self._not_built()

    def submit_order(self, symbol: str, side: OrderSide, quantity: float, **kw) -> Order:
        self._not_built()

    def cancel_order(self, order_id: str) -> Order:
        self._not_built()

    def close_position(self, symbol: str) -> Order | None:
        self._not_built()

    def get_order_status(self, order_id: str) -> Order:
        self._not_built()
