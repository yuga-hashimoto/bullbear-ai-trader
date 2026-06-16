"""Strategy = thin mapping from a validated Agent Signal to a trade intent.

The strategy performs NO AI/ML inference. It only translates an action into an
intent (enter a specific ETF / exit / do nothing). Symbol-policy and all risk
gating happen in the Risk Engine, which is authoritative.

Mapping:
  BUY_BULL + NASDAQ        -> ENTER TQQQ
  BUY_BEAR + NASDAQ        -> ENTER SQQQ
  BUY_BULL + SEMICONDUCTOR -> ENTER SOXL
  BUY_BEAR + SEMICONDUCTOR -> ENTER SOXS
  NO_TRADE                 -> NONE
  EXIT                     -> EXIT (close existing position)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..agents.signal_schema import FAMILY_BEAR, FAMILY_BULL, Signal
from ..config.settings import Config


@dataclass(frozen=True)
class TradeIntent:
    kind: str                 # "ENTER" | "EXIT" | "NONE"
    symbol: str | None
    signal: Signal


class Strategy:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.allowed_symbols = set(cfg.symbols)

    def map_signal(self, signal: Signal) -> TradeIntent:
        action = signal.action
        if action == "NO_TRADE":
            return TradeIntent("NONE", None, signal)
        if action == "EXIT":
            return TradeIntent("EXIT", signal.symbol, signal)

        # BUY_BULL / BUY_BEAR: prefer the explicit symbol; otherwise derive it
        # from (family, side). Allowed-symbol enforcement is in the Risk Engine.
        if signal.symbol:
            symbol = signal.symbol
        else:
            table = FAMILY_BULL if action == "BUY_BULL" else FAMILY_BEAR
            symbol = table.get(signal.target_family)
        if symbol is None:
            return TradeIntent("NONE", None, signal)
        return TradeIntent("ENTER", symbol, signal)
