"""LiveRunner — FUTURE USE, hard-disabled.

Constructing or running a LiveRunner requires the full triple safety gate
(config + env + explicit flag) AND ``trading.allow_live_orders``. Until real
moomoo order routing is built and audited, it raises even when unlocked. In the
default configuration it refuses to start.
"""
from __future__ import annotations

from ..config.settings import Config, assert_live_trading_allowed
from .base import BaseRunner


class LiveRunner(BaseRunner):
    name = "live"

    def __init__(self, cfg: Config, enable_live_trading: bool = False) -> None:
        super().__init__(cfg)
        # Triple gate (config + env + explicit flag) ...
        assert_live_trading_allowed(cfg, explicit_flag=enable_live_trading)
        # ... PLUS an independent order-routing switch.
        if not cfg.trading.allow_live_orders:
            from ..config.settings import LiveTradingDisabledError

            raise LiveTradingDisabledError("trading.allow_live_orders is false")

    def run(self) -> None:
        raise NotImplementedError(
            "LiveRunner is not implemented. Real moomoo order routing must be "
            "built and audited before live trading is possible."
        )
