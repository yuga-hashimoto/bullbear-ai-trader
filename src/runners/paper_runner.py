"""PaperRunner — always-on paper trading during US regular market hours.

Runs the per-bar loop only while the market is OPEN (America/New_York, Mon–Fri,
honoring holidays and early closes). Outside hours it sleeps. It uses ONLY the
PaperBroker — no real orders, ever. The Agent proposes; the Risk Engine decides;
a rejected signal never produces an OrderIntent.

Design for testability: ``step(now)`` performs exactly one iteration's logic and
is side-effect-explicit; ``run()`` is the thin loop that calls ``step`` and
sleeps to the next bar boundary. A clock and sleeper can be injected.
"""
from __future__ import annotations

import time as _time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from ..agents.base import BaseAgent
from ..agents.context import ContextInputs
from ..agents.signal_schema import Signal, SignalValidationError, no_trade_signal
from ..backtest.portfolio import Position
from ..brokers.base import OrderSide, OrderStatus, PositionInfo
from ..brokers.paper_broker import PaperBroker
from ..config.settings import Config
from ..features.builder import (
    build_feature_matrix,
    feature_columns,
    prepare_feature_matrix,
    price_col,
)
from ..market.calendar import MarketCalendar, make_calendar
from ..market.sessions import MarketState, to_zone
from ..risk.engine import EntryContext, RiskEngine
from ..risk.sizing import PositionSizer, PositionSizingInput
from ..strategy.strategy import Strategy
from ..utils.logging import get_logger
from .base import BaseRunner
from .feed import MarketDataFeed, make_live_feed
from .heartbeat import EventType, HeartbeatError
from .scheduler import bar_floor, interval_to_seconds, next_bar_boundary, seconds_until

log = get_logger(__name__)


class PaperRunner(BaseRunner):
    name = "paper"

    def __init__(
        self,
        cfg: Config,
        agent: BaseAgent,
        feed: MarketDataFeed | None = None,
        calendar: MarketCalendar | None = None,
        broker: PaperBroker | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.agent = agent
        self.tz = cfg.runner.timezone
        self.interval = cfg.runner.interval
        self.interval_s = interval_to_seconds(self.interval)
        self.calendar = calendar or make_calendar(cfg.market.calendar, self.tz)
        self.feed = feed or make_live_feed(cfg)
        self.broker = broker or PaperBroker(cash=cfg.backtest.initial_cash, costs=cfg.costs)
        self.strategy = Strategy(cfg)
        self.risk = RiskEngine(cfg.risk)
        self.sizer = PositionSizer()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper or _time.sleep

        self.symbols = cfg.symbols
        self.context_symbols = cfg.all_symbols
        self.allowed_symbols = set(cfg.symbols)

        # mutable state
        self.position: Position | None = None
        self.last_processed_bar: pd.Timestamp | None = None
        self.last_bar_time: pd.Timestamp | None = None
        self.last_signal_time: str | None = None
        self.last_order_time: str | None = None
        self.data_errors = 0
        self.agent_errors = 0
        self.daily_stop = False
        self.day_start_equity = cfg.backtest.initial_cash
        self.trade_seq = 0
        self.current_bar = 0
        self._current_day = None
        self._open_announced = None
        self._broker_disabled = False
        self.peak_equity = cfg.backtest.initial_cash
        self._last_context: dict[str, Any] | None = None
        self.shadow = self._make_shadow_book()
        self._restore_state()

    def _make_shadow_book(self):
        """Live shadow A/B book (challengers decide on the same bar as champion).

        Never affects the champion: if construction fails, shadow is disabled.
        """
        if not self.cfg.raw.get("evolution", {}).get("live_shadow", {}).get("enabled", True):
            return None
        try:
            from ..evolution.live_shadow import LiveShadowBook

            return LiveShadowBook(self.cfg)
        except Exception as exc:  # noqa: BLE001 - shadow must never break the runner
            log.warning("live shadow disabled (init failed): %s", exc)
            return None

    def _champion_analysis(self) -> dict[str, Any] | None:
        store = getattr(getattr(self.agent, "analysis_agent", None), "news_store", None)
        return store.latest_analysis if store is not None else None

    def _run_shadow(self, now, row, session) -> None:
        if self.shadow is None or self._last_context is None:
            return
        try:
            self.shadow.on_bar(now, row, self._last_context,
                               self._champion_analysis(), session)
        except Exception as exc:  # noqa: BLE001 - isolate shadow from the champion
            log.warning("live shadow bar error (champion unaffected): %s", exc)

    # ================================================================= run
    def run(self) -> None:
        self.writer.emit(EventType.RUNNER_STARTED, {"runner": self.name,
                          "interval": self.interval, "tz": self.tz})
        try:
            while not self.should_stop():
                now = self.clock()
                self.step(now)
                self._sleep_after(now)
        except KeyboardInterrupt:
            log.warning("KeyboardInterrupt: stopping safely")
        except HeartbeatError as exc:
            log.error("heartbeat failure: %s; stopping", exc)
            self.writer.emit_error(f"heartbeat failure: {exc}")
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            log.exception("unexpected error; saving state and stopping")
            self.writer.emit_error(f"unexpected: {exc}")
        finally:
            self._save_state(self.clock())
            self.writer.emit(EventType.RUNNER_STOPPED, {"runner": self.name})

    def _sleep_after(self, now: datetime) -> None:
        now = to_zone(now, self.tz)
        if self.calendar.is_market_open(now):
            target = next_bar_boundary(now, self.interval_s, self.tz)
        else:
            try:
                target = self.calendar.next_market_open(now)
            except RuntimeError:
                target = now + pd.Timedelta(seconds=self.cfg.runner.poll_interval_seconds)
        # Cap by poll interval so stop requests are picked up promptly.
        secs = min(seconds_until(target, now, self.tz), self.cfg.runner.poll_interval_seconds)
        self.sleeper(max(secs, 0.0))

    # ================================================================ step
    def step(self, now: datetime) -> dict[str, Any]:
        """One iteration. Returns a small status dict (for tests/inspection)."""
        now = to_zone(now, self.tz)
        self._roll_day(now)
        state = self.calendar.classify_state(now)

        if state not in (MarketState.OPEN, MarketState.EARLY_CLOSE):
            return self._handle_closed(now, state)

        if self._open_announced != now.date():
            self.writer.emit(EventType.MARKET_OPEN, {"date": str(now.date()),
                             "early_close": state == MarketState.EARLY_CLOSE})
            self._open_announced = now.date()

        session = self.calendar.session_for_date(now.date())
        bar_time = bar_floor(now, self.interval_s, self.tz)
        if self.cfg.runner.prevent_duplicate_bar_processing and bar_time == self.last_processed_bar:
            self._heartbeat(now, "running", state, session)
            return {"action": "wait_bar", "state": state.value}

        frames = self._fetch(now)
        last_bar = self._latest_bar_time(frames)
        self.last_bar_time = last_bar
        if self._is_stale(now, last_bar):
            return self._handle_stale(now, state, session, last_bar)
        self.data_errors = 0
        self.writer.emit(EventType.MARKET_DATA, {"last_bar": str(last_bar)})

        matrix = build_feature_matrix(frames, self.cfg)
        matrix, _health = prepare_feature_matrix(matrix)
        feat_cols = feature_columns(matrix)
        if matrix.empty:
            self.last_processed_bar = bar_time
            self._heartbeat(now, "running", state, session, reason="warmup")
            return {"action": "warmup", "state": state.value}
        row = matrix.iloc[-1]
        self.writer.emit(EventType.FEATURE_BUILT, {"bar": str(matrix.index[-1])})
        for sym in self.symbols:
            self.broker.set_price(sym, float(row[price_col(sym, "close")]))

        self.current_bar += 1
        self._check_runtime_circuit_breakers(now)
        if self.daily_stop:
            self.last_processed_bar = bar_time
            self._heartbeat(now, "stopped", state, session, reason="risk_circuit_breaker")
            return {"action": "risk_stop", "state": state.value, "bar": str(bar_time)}

        session_close_str = session.close_dt.strftime("%H:%M")
        self._manage_exit(now, row, session, session_close_str)
        self._maybe_enter(now, row, session, state)
        self._run_shadow(now, row, session)

        self.last_processed_bar = bar_time
        self._heartbeat(now, "running", state, session)
        return {"action": "processed", "state": state.value, "bar": str(bar_time)}

    # =========================================================== handlers
    def _handle_closed(self, now, state) -> dict[str, Any]:
        if self.position is not None:
            self.writer.emit_error("position still open after market close",
                                   {"symbol": self.position.symbol})
        evt = {MarketState.HOLIDAY: EventType.MARKET_HOLIDAY,
               MarketState.CLOSED: EventType.MARKET_CLOSED}.get(state, EventType.MARKET_CLOSED)
        self.writer.emit(evt, {"state": state.value})
        self.writer.emit(EventType.RUNNER_SLEEPING, {"state": state.value})
        self._heartbeat(now, "sleeping", state, session=None)
        return {"action": "sleep", "state": state.value}

    def _handle_stale(self, now, state, session, last_bar) -> dict[str, Any]:
        self.data_errors += 1
        self.writer.emit(EventType.MARKET_DATA_STALE,
                         {"last_bar": str(last_bar), "count": self.data_errors})
        self._heartbeat(now, "waiting", state, session, reason="stale_data")
        return {"action": "stale", "state": state.value}

    def _manage_exit(self, now, row, session, session_close_str) -> None:
        if self.position is None:
            return
        close_px = float(row[price_col(self.position.symbol, "close")])
        self.position = self.position.update_peak(close_px)
        now_ts = pd.Timestamp(now)
        dec = self.risk.check_exit(now_ts, self.position, close_px, session_close_str)
        force = self._should_force_close(now, session)
        if dec.ok or force:
            reason = dec.reason if dec.ok else "force_close_early" if session.is_early_close else "force_close_eod"
            self._close_position(now, close_px, reason)

    def _maybe_enter(self, now, row, session, state) -> None:
        allow = True
        block_reason = ""
        if self.daily_stop:
            allow, block_reason = False, "daily_stop"
        elif self.position is not None:
            allow, block_reason = False, "position_open"
        elif session.minutes_since_open(now) < self.cfg.risk.no_trade_first_minutes:
            allow, block_reason = False, "no_trade_first_minutes"
        elif session.minutes_to_close(now) <= self.cfg.risk.no_new_entry_last_minutes:
            allow, block_reason = False, "no_new_entry_last_minutes"

        signal = self._get_signal(now, row)
        self.writer.emit(EventType.AGENT_SIGNAL, signal.to_dict())
        self.last_signal_time = signal.timestamp
        self.writer.write_json("latest_signal.json", signal.to_dict())

        # Agent EXIT closes an existing position regardless of entry gating.
        if signal.action == "EXIT" and self.position is not None:
            close_px = float(row[price_col(self.position.symbol, "close")])
            self._close_position(now, close_px, "agent_exit")
            return

        if signal.action not in ("BUY_BULL", "BUY_BEAR"):
            self._risk_decision(now, signal, "NO_ACTION", signal.action.lower())
            return
        if not allow:
            self._risk_decision(now, signal, "REJECT", block_reason)
            return

        vdec = self.risk.validate_signal(signal, self.allowed_symbols)
        if not vdec.ok:
            self._risk_decision(now, signal, "REJECT", vdec.reason)
            return
        intent = self.strategy.map_signal(signal)
        if intent.kind != "ENTER" or intent.symbol not in self.allowed_symbols:
            self._risk_decision(now, signal, "REJECT", "symbol_not_allowed")
            return

        atr_feat = f"feat__{intent.symbol}__atr_pct"
        atr_pct = float(row[atr_feat]) * 100.0 if atr_feat in row.index else 0.0
        ctx = EntryContext(
            now=pd.Timestamp(now), equity=self.broker.get_cash(),
            spread_pct=self.cfg.costs.spread_pct, atr_pct=atr_pct,
            n_open_positions=0, candidate_symbol=intent.symbol, current_bar=self.current_bar,
            session_open=self.cfg.market.regular_open, session_close=session.close_dt.strftime("%H:%M"),
            max_concurrent=self.cfg.strategy.max_concurrent_positions,
        )
        edec = self.risk.check_entry(ctx)
        if not edec.ok:
            self._risk_decision(now, signal, "REJECT", edec.reason)
            return
        self._risk_decision(now, signal, "ACCEPT", "ok")
        self._open_position(now, row, intent, signal)

    # ============================================================ trading
    def _open_position(self, now, row, intent, signal) -> None:
        if self._broker_disabled:
            self.writer.emit(EventType.ORDER_REJECTED, {"reason": "broker_disabled"})
            return
        ref = float(row[price_col(intent.symbol, "close")])
        stop_pct = min(self.cfg.risk.max_loss_per_trade_pct, 99.0)
        stop_px = ref * (1.0 - stop_pct / 100.0)
        portfolio_budget = (
            self.cfg.account.initial_capital_jpy
            * self.cfg.risk.max_portfolio_risk_pct
            / 100.0
        )
        sizing = self.sizer.size(PositionSizingInput(
            entry_price_usd=ref,
            stop_price_usd=stop_px,
            cash_usd=self.broker.get_cash() * self.cfg.backtest.fraction_per_trade,
            usd_jpy=self.cfg.account.usd_jpy_rate,
            max_trade_loss_jpy=self.cfg.risk.max_loss_per_trade_jpy,
            portfolio_risk_remaining_jpy=portfolio_budget,
            overnight_gap_pct=(
                self.cfg.risk.overnight_gap_risk_pct
                if self.cfg.risk.allow_overnight_positions
                else 0.0
            ),
        ))
        target_shares = sizing.quantity
        self.writer.emit(EventType.ORDER_INTENT,
                         {"symbol": intent.symbol, "side": "BUY", "shares": target_shares})
        if target_shares <= 0:
            self.writer.emit(EventType.ORDER_REJECTED, {"reason": "insufficient_cash"})
            return
        try:
            order = self.broker.submit_order(intent.symbol, OrderSide.BUY, target_shares)
        except Exception as exc:  # noqa: BLE001 - broker boundary
            self._disable_broker(f"submit failed: {exc}")
            return
        if order.status != OrderStatus.FILLED:
            self.writer.emit(EventType.ORDER_REJECTED, {"reason": order.status.value})
            return
        pos_info = next((p for p in self.broker.get_positions() if p.symbol == intent.symbol), None)
        if pos_info is None:
            return
        self.trade_seq += 1
        self.position = Position(
            symbol=intent.symbol, direction=signal.direction, entry_time=pd.Timestamp(now),
            entry_price=pos_info.avg_price, shares=pos_info.quantity, entry_bar=0,
            peak_price=pos_info.avg_price, trade_id=self.trade_seq,
            entry_reason=signal.reason or signal.action,
            entry_commission=pos_info.entry_commission,
            stop_price=stop_px,
            planned_loss_jpy=sizing.planned_loss_jpy,
        )
        self.risk.on_open()
        self.last_order_time = pd.Timestamp(now).isoformat()
        self.writer.emit(EventType.PAPER_FILL,
                         {"symbol": intent.symbol, "price": order.fill_price, "shares": pos_info.quantity})
        self.writer.emit(EventType.POSITION_OPENED,
                         {"symbol": intent.symbol, "trade_id": self.trade_seq,
                          "entry_price": pos_info.avg_price})

    def _close_position(self, now, ref_px: float, reason: str) -> None:
        pos = self.position
        if pos is None:
            return
        try:
            order = self.broker.close_position(pos.symbol)
        except Exception as exc:  # noqa: BLE001 - broker boundary
            self._disable_broker(f"close failed: {exc}")
            return
        exit_px = order.fill_price if order and order.fill_price else ref_px
        net = float(
            order.meta.get("realized_pnl")
            if order is not None and "realized_pnl" in order.meta
            else pos.shares * (exit_px - pos.entry_price) - pos.entry_commission
        )
        self.position = None
        self.risk.on_close(pos.symbol, net, self.current_bar, self.broker.get_cash())
        self.last_order_time = pd.Timestamp(now).isoformat()
        is_force = reason.startswith("force_close")
        self.writer.emit(EventType.FORCE_EXIT if is_force else EventType.POSITION_CLOSED,
                         {"symbol": pos.symbol, "trade_id": pos.trade_id,
                          "exit_price": exit_px, "net_pnl": round(net, 2), "reason": reason})
        self._check_daily_stop(now)

    # ============================================================ helpers
    def _get_signal(self, now, row) -> Signal:
        positions = []
        if self.position is not None:
            close_px = float(row[price_col(self.position.symbol, "close")])
            positions.append({"symbol": self.position.symbol, "direction": self.position.direction,
                              "entry_price": round(self.position.entry_price, 4),
                              "shares": self.position.shares,
                              "unrealized_pct": round(self.position.unrealized_pct(close_px) * 100, 4),
                              "trade_id": self.position.trade_id})
        equity = self._equity(row)
        risk_state = {"trades_today": self.risk.trades_today,
                      "consecutive_losses": self.risk.consecutive_losses,
                      "daily_stop": self.daily_stop,
                      "can_open_new_position": self.position is None and not self.daily_stop}
        inputs = ContextInputs(timestamp=pd.Timestamp(now), row=row, symbols=self.context_symbols,
                               positions=positions, daily_pnl=round(equity - self.day_start_equity, 2),
                               risk_state=risk_state)
        context = self.agent.build_context(inputs)
        self._last_context = context
        self.writer.emit(EventType.AGENT_CONTEXT, {"timestamp": context["timestamp"]})
        try:
            raw = self._request_with_timeout(context)
            signal = self.agent.parse_response(raw)
            self.agent.validate_signal(signal)
            self.writer.emit(EventType.SIGNAL_VALIDATION, {"valid": True})
            self.agent_errors = 0
            return signal
        except FutureTimeout:
            self.agent_errors += 1
            self.writer.emit(EventType.AGENT_ERROR, {"reason": "timeout", "count": self.agent_errors})
            if self.agent_errors >= self.cfg.runner.max_agent_errors_before_stop:
                self._safe_stop(now, "max_agent_errors")
            return no_trade_signal(pd.Timestamp(now).isoformat(), getattr(self.agent, "name", "agent"), "timeout")
        except SignalValidationError as exc:
            self.writer.emit(EventType.SIGNAL_VALIDATION, {"valid": False, "error": str(exc)})
            return no_trade_signal(pd.Timestamp(now).isoformat(), getattr(self.agent, "name", "agent"), f"invalid: {exc}")
        except Exception as exc:  # noqa: BLE001 - agent boundary
            self.agent_errors += 1
            self.writer.emit(EventType.AGENT_ERROR, {"reason": str(exc), "count": self.agent_errors})
            if self.agent_errors >= self.cfg.runner.max_agent_errors_before_stop:
                self._safe_stop(now, "max_agent_errors")
            return no_trade_signal(pd.Timestamp(now).isoformat(), getattr(self.agent, "name", "agent"), "agent_error")

    def _request_with_timeout(self, context):
        t = self.cfg.runner.max_agent_latency_seconds
        if not t or t <= 0:
            return self.agent.request_signal(context)
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(self.agent.request_signal, context).result(timeout=t)

    def _risk_decision(self, now, signal, decision, reason) -> None:
        rec = {"timestamp": pd.Timestamp(now).isoformat(), "action": signal.action,
               "symbol": signal.symbol, "confidence": signal.confidence,
               "decision": decision, "rejection_reason": "" if decision in ("ACCEPT", "NO_ACTION") else reason,
               "trades_today": self.risk.trades_today, "consecutive_losses": self.risk.consecutive_losses,
               "daily_stop": self.daily_stop,
               "current_position": self.position.symbol if self.position else None}
        self.writer.emit(EventType.RISK_DECISION, rec)
        self.writer.write_json("latest_risk_decision.json", rec)

    def _should_force_close(self, now, session) -> bool:
        if self.cfg.risk.allow_overnight_positions:
            return False
        mins = self.cfg.market.early_close_force_exit_minutes_before_close if session.is_early_close \
            else self.cfg.risk.force_close_minutes_before_close
        return session.minutes_to_close(now) <= mins

    def _check_daily_stop(self, now) -> None:
        if self.daily_stop:
            return
        if self.risk.halted_today or self.risk.consecutive_losses >= self.cfg.risk.max_consecutive_losses:
            self.daily_stop = True
            self.writer.emit(EventType.DAILY_STOP,
                             {"consecutive_losses": self.risk.consecutive_losses,
                              "halted": self.risk.halted_today})

    def _marked_equity(self) -> float:
        equity = self.broker.get_cash()
        broker_positions = self.broker.get_positions()
        for pos in broker_positions:
            price = self.broker.get_market_data(pos.symbol).get("last", pos.avg_price)
            equity += pos.quantity * price
        if self.position is not None and not any(
            pos.symbol == self.position.symbol for pos in broker_positions
        ):
            price = self.broker.get_market_data(self.position.symbol).get(
                "last", self.position.entry_price
            )
            equity += self.position.shares * price
        return equity

    def _check_runtime_circuit_breakers(self, now) -> None:
        equity = self._marked_equity()
        self.peak_equity = max(self.peak_equity, equity)
        daily_loss_jpy = max(self.day_start_equity - equity, 0.0) * self.cfg.account.usd_jpy_rate
        drawdown_pct = (
            (self.peak_equity - equity) / self.peak_equity * 100.0
            if self.peak_equity > 0
            else 0.0
        )
        if (
            daily_loss_jpy < self.cfg.risk.max_daily_loss_jpy
            and drawdown_pct < self.cfg.risk.max_drawdown_pct
        ):
            return
        reason = (
            "max_daily_loss_jpy"
            if daily_loss_jpy >= self.cfg.risk.max_daily_loss_jpy
            else "max_drawdown_pct"
        )
        self.daily_stop = True
        if self.position is not None:
            ref = self.broker.get_market_data(self.position.symbol).get(
                "last", self.position.entry_price
            )
            self._close_position(now, float(ref), reason)
        self.writer.emit(EventType.DAILY_STOP, {
            "reason": reason,
            "daily_loss_jpy": round(daily_loss_jpy, 2),
            "drawdown_pct": round(drawdown_pct, 4),
        })

    def _roll_day(self, now) -> None:
        d = now.date()
        if d != self._current_day:
            equity = self.broker.get_cash() + (
                self.position.shares * self.position.entry_price if self.position else 0.0)
            self.risk.new_day(equity, d)
            self.day_start_equity = equity
            self.peak_equity = max(self.peak_equity, equity)
            self.daily_stop = False
            self._current_day = d
            self._open_announced = None

    def _restore_state(self) -> None:
        daily = self.writer.read_json("daily_state.json")
        position_data = self.writer.read_json("runner_position.json")
        if not daily:
            return
        self.day_start_equity = float(
            daily.get("day_start_equity", self.cfg.backtest.initial_cash)
        )
        self.peak_equity = float(daily.get("peak_equity", self.day_start_equity))
        self.daily_stop = bool(daily.get("daily_stop", False))
        self.risk.trades_today = int(daily.get("trades_today", 0))
        self.risk.consecutive_losses = int(daily.get("consecutive_losses", 0))
        date_text = daily.get("date")
        if date_text and date_text != "None":
            self._current_day = pd.Timestamp(date_text).date()
            self.risk._current_day = self._current_day
            self.risk.day_start_equity = self.day_start_equity

        positions: list[PositionInfo] = []
        if position_data:
            self.position = Position(
                symbol=str(position_data["symbol"]),
                direction=str(position_data["direction"]),
                entry_time=pd.Timestamp(position_data["entry_time"]),
                entry_price=float(position_data["entry_price"]),
                shares=float(position_data["shares"]),
                entry_bar=int(position_data.get("entry_bar", 0)),
                peak_price=float(position_data.get("peak_price", position_data["entry_price"])),
                trade_id=int(position_data.get("trade_id", 0)),
                entry_reason=str(position_data.get("entry_reason", "")),
                entry_commission=float(position_data.get("entry_commission", 0.0)),
                stop_price=position_data.get("stop_price"),
                planned_loss_jpy=float(position_data.get("planned_loss_jpy", 0.0)),
            )
            positions.append(PositionInfo(
                self.position.symbol,
                self.position.shares,
                self.position.entry_price,
                self.position.entry_commission,
            ))
            self.trade_seq = max(self.trade_seq, self.position.trade_id)
            self.current_bar = max(self.current_bar, self.position.entry_bar)
        self.broker.restore_account(float(daily.get("cash", self.broker.get_cash())), positions)

    def _fetch(self, now) -> dict[str, pd.DataFrame]:
        return self.feed.fetch_recent(self.context_symbols, self.interval, now)

    @staticmethod
    def _latest_bar_time(frames) -> pd.Timestamp | None:
        times = [df.index[-1] for df in frames.values() if not df.empty]
        return max(times) if times else None

    def _is_stale(self, now, last_bar: pd.Timestamp | None) -> bool:
        if last_bar is None:
            return True
        bar_close = last_bar + pd.Timedelta(seconds=self.interval_s)
        allowed_delay = (
            self.cfg.runner.vendor_delay_seconds
            + self.cfg.runner.stale_data_threshold_seconds
        )
        return (pd.Timestamp(now) - bar_close).total_seconds() > allowed_delay

    def _equity(self, row) -> float:
        if self.position is None:
            return self.broker.get_cash()
        close_px = float(row[price_col(self.position.symbol, "close")])
        return self.broker.get_cash() + self.position.shares * close_px

    def _disable_broker(self, reason: str) -> None:
        self._broker_disabled = True
        self.writer.emit_error(f"broker disabled: {reason}")
        self.writer.emit(EventType.ORDER_REJECTED, {"reason": "broker_disabled"})

    def _safe_stop(self, now, reason: str) -> None:
        self.writer.emit_error(f"safe stop: {reason}")
        self._save_state(now)
        self.stop()

    def _heartbeat(self, now, status, state, session, reason: str = "") -> None:
        try:
            nmo = self.calendar.next_market_open(now)
            nmc = self.calendar.next_market_close(now)
        except RuntimeError:
            nmo = nmc = None
        hb = {
            "timestamp": pd.Timestamp(now).isoformat(),
            "runner": self.name,
            "status": status,
            "market_state": state.value,
            "reason": reason,
            "current_session_open": session.open_dt.isoformat() if session else None,
            "current_session_close": session.close_dt.isoformat() if session else None,
            "next_market_open": nmo.isoformat() if nmo else None,
            "next_market_close": nmc.isoformat() if nmc else None,
            "last_bar_time": str(self.last_bar_time) if self.last_bar_time is not None else None,
            "last_processed_bar_time": str(self.last_processed_bar) if self.last_processed_bar is not None else None,
            "last_signal_time": self.last_signal_time,
            "last_order_time": self.last_order_time,
            "positions": self._position_snapshots(),
            "daily_pnl": round(self._marked_equity() - self.day_start_equity, 2),
            "daily_pnl_jpy": round(
                (self._marked_equity() - self.day_start_equity)
                * self.cfg.account.usd_jpy_rate,
                2,
            ),
            "peak_equity": round(self.peak_equity, 2),
            "trades_today": self.risk.trades_today,
            "consecutive_losses": self.risk.consecutive_losses,
            "daily_stop": self.daily_stop,
            "errors": [],
        }
        self.writer.write_heartbeat(hb)
        self.writer.emit(EventType.HEARTBEAT, {"status": status, "market_state": state.value})
        self._save_state(now)

    def _save_state(self, now) -> None:
        self.writer.write_json("current_positions.json", self._position_snapshots())
        self.writer.write_json("daily_state.json", {
            "date": str(self._current_day),
            "day_start_equity": self.day_start_equity,
            "cash": self.broker.get_cash(),
            "marked_equity": self._marked_equity(),
            "peak_equity": self.peak_equity,
            "trades_today": self.risk.trades_today,
            "consecutive_losses": self.risk.consecutive_losses,
            "daily_stop": self.daily_stop,
        })
        self.writer.write_json(
            "runner_position.json",
            self.position.__dict__ if self.position is not None else None,
        )

    def _position_snapshots(self) -> list[dict[str, Any]]:
        if self.position is None:
            return []
        current_price = float(
            self.broker.get_market_data(self.position.symbol).get(
                "last", self.position.entry_price
            )
        )
        return [{
            "symbol": self.position.symbol,
            "direction": self.position.direction,
            "entry_price": self.position.entry_price,
            "current_price": current_price,
            "shares": self.position.shares,
            "unrealized_pct": round(
                self.position.unrealized_pct(current_price) * 100.0,
                4,
            ),
            "trade_id": self.position.trade_id,
        }]
