"""Risk engine — authoritative over the model.

The strategy proposes; the risk engine disposes. Every entry must pass
:meth:`check_entry`; every open position is tested each bar by
:meth:`check_exit`. When in doubt the engine blocks (NO_TRADE) — that is the
single most important behavior in this system.

The engine keeps per-day accounting (PnL, trade count, consecutive losses,
per-symbol cooldown). It is reset at each new session via :meth:`new_day`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..agents.signal_schema import FAMILY_BEAR, FAMILY_BULL, Signal
from ..config.settings import RiskConfig
from ..utils.timeutils import minutes_since_open, minutes_to_close
from ..backtest.portfolio import Position


@dataclass(frozen=True)
class EntryContext:
    now: pd.Timestamp
    equity: float
    spread_pct: float          # quoted half-spread proxy, in percent
    atr_pct: float             # volatility proxy, in percent
    n_open_positions: int
    candidate_symbol: str
    current_bar: int
    session_open: str
    session_close: str
    max_concurrent: int


@dataclass(frozen=True)
class Decision:
    ok: bool
    reason: str = ""


@dataclass
class RiskEngine:
    cfg: RiskConfig
    day_start_equity: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    halted_today: bool = False
    last_exit_bar: dict[str, int] = field(default_factory=dict)
    _current_day: object = field(default=None, repr=False)

    # --- day lifecycle ------------------------------------------------------
    def new_day(self, equity: float, day) -> None:
        self.day_start_equity = equity
        self.trades_today = 0
        self.consecutive_losses = 0
        self.halted_today = False
        self.last_exit_bar = {}
        self._current_day = day

    def maybe_roll_day(self, ts: pd.Timestamp, equity: float) -> None:
        if ts.date() != self._current_day:
            self.new_day(equity, ts.date())

    # --- signal gate (Agent proposal -> accept/reject) ----------------------
    def validate_signal(self, signal: Signal, allowed_symbols: set[str]) -> Decision:
        """Business validation of an ENTRY signal (authoritative over the agent).

        Structural validity is assumed (the engine validates the schema first).
        Here we enforce policy: confidence threshold, allowed-symbol list, and
        family/side/symbol consistency. NO_TRADE / EXIT are not entry signals
        and are handled by the engine, not here.
        """
        if signal.action not in ("BUY_BULL", "BUY_BEAR"):
            return Decision(False, "not_an_entry_action")
        if signal.confidence < self.cfg.confidence_threshold:
            return Decision(False, "low_confidence")
        if signal.symbol is None or signal.symbol not in allowed_symbols:
            return Decision(False, "symbol_not_allowed")
        table = FAMILY_BULL if signal.action == "BUY_BULL" else FAMILY_BEAR
        expected = table.get(signal.target_family)
        if expected is None:
            return Decision(False, "family_not_tradable")
        if signal.symbol != expected:
            return Decision(False, "family_symbol_mismatch")
        return Decision(True, "ok")

    # --- entry gate ---------------------------------------------------------
    def check_entry(self, ctx: EntryContext) -> Decision:
        c = self.cfg
        if self.halted_today:
            return Decision(False, "daily_loss_halt")
        if ctx.n_open_positions >= ctx.max_concurrent:
            return Decision(False, "position_already_open")
        if self.trades_today >= c.max_trades_per_day:
            return Decision(False, "max_trades_per_day")
        if self.consecutive_losses >= c.max_consecutive_losses:
            return Decision(False, "max_consecutive_losses")

        mins_open = minutes_since_open(ctx.now, ctx.session_open)
        if mins_open < c.no_trade_first_minutes:
            return Decision(False, "no_trade_first_minutes")

        mins_close = minutes_to_close(ctx.now, ctx.session_close)
        if mins_close <= c.no_new_entry_last_minutes:
            return Decision(False, "no_new_entry_last_minutes")

        if ctx.spread_pct > c.max_spread_pct:
            return Decision(False, "spread_too_wide")
        if ctx.atr_pct > c.max_atr_pct:
            return Decision(False, "volatility_too_high")

        last_exit = self.last_exit_bar.get(ctx.candidate_symbol)
        if last_exit is not None and (ctx.current_bar - last_exit) < c.min_bars_between_same_symbol:
            return Decision(False, "symbol_cooldown")

        return Decision(True, "ok")

    # --- exit gate ----------------------------------------------------------
    def check_exit(
        self,
        now: pd.Timestamp,
        position: Position,
        price: float,
        session_close: str,
    ) -> Decision:
        c = self.cfg
        ret_pct = position.unrealized_pct(price) * 100.0

        # Force flat near the close (no overnight holds).
        if minutes_to_close(now, session_close) <= c.force_close_minutes_before_close:
            return Decision(True, "force_close_before_close")

        if ret_pct <= -c.max_loss_per_trade_pct:
            return Decision(True, "stop_loss")
        if ret_pct >= c.take_profit_pct:
            return Decision(True, "take_profit")

        # Trailing stop measured from the peak since entry.
        drawdown_from_peak = (price - position.peak_price) / position.peak_price * 100.0
        if drawdown_from_peak <= -c.trailing_stop_pct:
            return Decision(True, "trailing_stop")

        held_min = (now - position.entry_time).total_seconds() / 60.0
        if held_min >= c.max_holding_minutes:
            return Decision(True, "max_holding_time")

        return Decision(False, "hold")

    # --- accounting ---------------------------------------------------------
    def on_open(self) -> None:
        self.trades_today += 1

    def on_close(self, symbol: str, net_pnl: float, exit_bar: int, equity: float) -> None:
        self.last_exit_bar[symbol] = exit_bar
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        # Daily loss halt.
        if self.day_start_equity > 0:
            day_pnl_pct = (equity - self.day_start_equity) / self.day_start_equity * 100.0
            if day_pnl_pct <= -self.cfg.max_daily_loss_pct:
                self.halted_today = True
