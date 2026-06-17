"""Event-driven, Agent-driven backtest engine.

Per-bar flow (matches the project spec):
  1. compute features (already in the matrix) for the current bar
  2. build the observable Agent Context (no future data)
  3. request a Signal from the Agent
  4. validate the Signal schema (invalid -> NO_TRADE, counted)
  5. Risk Engine decides accept/reject (authoritative over the Agent)
  6. only accepted signals reach execution simulation
  7. update PnL / positions
  8. log agent signals, risk decisions and trades

Timing (no look-ahead): decisions use the **close** of bar *i*; fills happen at
the **open** of bar *i+1*. Positions are force-closed at the last in-session bar
so nothing is held overnight. The Agent NEVER places orders — it only proposes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..agents.base import BaseAgent
from ..agents.context import ContextInputs
from ..agents.signal_schema import Signal, SignalValidationError, no_trade_signal
from ..config.settings import Config
from ..features.builder import feature_columns, price_col
from ..risk.engine import EntryContext, RiskEngine
from ..strategy.strategy import Strategy, TradeIntent
from ..utils.logging import get_logger
from .execution import ExecutionModel
from .portfolio import ClosedTrade, Position

log = get_logger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[ClosedTrade]
    daily_pnl: pd.DataFrame
    agent_signals: list[dict]
    risk_decisions: list[dict]
    counters: dict[str, Any]
    initial_cash: float

    @property
    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[f.name for f in ClosedTrade.__dataclass_fields__.values()]
            )
        return pd.DataFrame([t.__dict__ for t in self.trades])


@dataclass
class _State:
    cash: float
    position: Position | None = None
    pending_entry: TradeIntent | None = None
    pending_entry_id: int = 0
    pending_exit_reason: str | None = None
    trade_seq: int = 0
    equity_times: list[pd.Timestamp] = field(default_factory=list)
    equity_values: list[float] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)


class BacktestEngine:
    def __init__(self, cfg: Config, agent: BaseAgent) -> None:
        self.cfg = cfg
        self.agent = agent
        self.strategy = Strategy(cfg)
        self.risk = RiskEngine(cfg.risk)
        self.execution = ExecutionModel(cfg.costs)
        self.allowed_symbols = set(cfg.symbols)
        self.context_symbols = cfg.all_symbols

    # ------------------------------------------------------------------ run
    def run(self, matrix: pd.DataFrame) -> BacktestResult:
        cfg = self.cfg
        feat_cols = feature_columns(matrix)
        matrix = matrix.dropna(subset=feat_cols).sort_index()
        if matrix.empty:
            raise ValueError("feature matrix is empty after dropping warmup NaNs")

        self.agent.prepare(matrix, feat_cols)
        timestamps = matrix.index
        n = len(timestamps)

        st = _State(cash=cfg.backtest.initial_cash)
        self.risk.new_day(st.cash, timestamps[0].date())

        agent_signals: list[dict] = []
        risk_decisions: list[dict] = []
        action_dist: Counter = Counter()
        reject_reasons: Counter = Counter()
        invalid_count = 0
        rejected_count = 0
        no_trade_count = 0

        for i in range(n):
            t = timestamps[i]
            row = matrix.iloc[i]
            is_last_of_day = (i == n - 1) or (timestamps[i + 1].date() != t.date())

            equity_now = self._equity(st, row)
            self.risk.maybe_roll_day(t, equity_now)

            # STEP A: execute orders pending from the previous bar's decision.
            self._execute_pending(st, t, i, row)

            # STEP B(1): risk-driven exit on the open position.
            if st.position is not None:
                close_px = self._px(row, st.position.symbol, "close")
                st.position = st.position.update_peak(close_px)
                exit_dec = self.risk.check_exit(t, st.position, close_px, cfg.session_close)
                if exit_dec.ok or is_last_of_day:
                    reason = exit_dec.reason if exit_dec.ok else "force_close_eod"
                    if is_last_of_day:
                        self._close_position(st, t, i, close_px, reason)
                    elif st.pending_exit_reason is None:
                        st.pending_exit_reason = reason

            # STEP B(2): ask the agent for a signal (every bar).
            equity_now = self._equity(st, row)
            context = self.agent.build_context(self._context_inputs(st, t, row, equity_now))
            raw = self.agent.request_signal(context)
            signal, invalid = self._parse(raw, t)
            if invalid:
                invalid_count += 1
            action_dist[signal.action] += 1
            if signal.action == "NO_TRADE":
                no_trade_count += 1

            accepted = False
            decision = "REJECT"
            reason = ""
            trade_id_for_log: int | None = None

            # STEP B(3): agent-requested EXIT.
            if (
                signal.action == "EXIT"
                and st.position is not None
                and st.pending_exit_reason is None
                and not is_last_of_day
            ):
                st.pending_exit_reason = "agent_exit"
                decision, reason = "FORCE_EXIT", "agent_exit"
                trade_id_for_log = st.position.trade_id

            # STEP B(4): entry path (Risk Engine is authoritative).
            elif signal.action in ("BUY_BULL", "BUY_BEAR"):
                decision, reason, accepted, trade_id_for_log = self._handle_entry(
                    st, t, i, row, signal, is_last_of_day
                )
                if not accepted:
                    rejected_count += 1
                    reject_reasons[reason] += 1
            else:
                decision, reason = "NO_ACTION", signal.action.lower()

            agent_signals.append(self._signal_log(signal, accepted, reason, trade_id_for_log))
            risk_decisions.append(self._risk_log(st, t, signal, decision, reason, equity_now, trade_id_for_log))

            st.equity_times.append(t)
            st.equity_values.append(self._equity(st, row))

        equity = pd.Series(st.equity_values, index=pd.DatetimeIndex(st.equity_times))
        daily = self._daily_pnl(equity)
        forced_exits = sum(
            1 for tr in st.trades
            if tr.exit_reason in ("force_close_eod", "force_close_before_close")
        )
        counters = {
            "num_signals": n,
            "no_trade_count": no_trade_count,
            "no_trade_ratio": round(no_trade_count / n, 4) if n else 0.0,
            "invalid_signals": invalid_count,
            "rejected_signals": rejected_count,
            "forced_exits": forced_exits,
            "action_distribution": dict(action_dist),
            "risk_rejection_reasons": dict(reject_reasons),
        }
        return BacktestResult(
            equity_curve=equity,
            trades=st.trades,
            daily_pnl=daily,
            agent_signals=agent_signals,
            risk_decisions=risk_decisions,
            counters=counters,
            initial_cash=cfg.backtest.initial_cash,
        )

    # -------------------------------------------------------------- helpers
    def _parse(self, raw: dict, t: pd.Timestamp) -> tuple[Signal, bool]:
        """Parse + validate a raw signal. On failure -> NO_TRADE, invalid=True."""
        try:
            signal = self.agent.parse_response(raw)
            self.agent.validate_signal(signal)
            return signal, False
        except SignalValidationError as exc:
            log.warning("invalid signal at %s: %s", t, exc)
            return no_trade_signal(t.isoformat(), getattr(self.agent, "name", "agent"),
                                   reason=f"invalid: {exc}"), True

    def _handle_entry(
        self, st: _State, t: pd.Timestamp, i: int, row: pd.Series,
        signal: Signal, is_last_of_day: bool,
    ) -> tuple[str, str, bool, int | None]:
        if st.position is not None or st.pending_entry is not None or is_last_of_day:
            return "REJECT", "position_open_or_eod", False, None

        vdec = self.risk.validate_signal(signal, self.allowed_symbols)
        if not vdec.ok:
            return "REJECT", vdec.reason, False, None

        intent = self.strategy.map_signal(signal)
        if intent.kind != "ENTER" or intent.symbol not in self.allowed_symbols:
            return "REJECT", "symbol_not_allowed", False, None

        atr_feat = f"feat__{intent.symbol}__atr_pct"
        atr_pct = float(row[atr_feat]) * 100.0 if atr_feat in row.index else 0.0
        ctx = EntryContext(
            now=t, equity=st.cash, spread_pct=self.cfg.costs.spread_pct, atr_pct=atr_pct,
            n_open_positions=1 if st.position is not None else 0,
            candidate_symbol=intent.symbol, current_bar=i,
            session_open=self.cfg.session_open, session_close=self.cfg.session_close,
            max_concurrent=self.cfg.strategy.max_concurrent_positions,
        )
        edec = self.risk.check_entry(ctx)
        if not edec.ok:
            return "REJECT", edec.reason, False, None

        st.trade_seq += 1
        st.pending_entry = intent
        st.pending_entry_id = st.trade_seq
        return "ACCEPT", "ok", True, st.trade_seq

    def _context_inputs(self, st: _State, t: pd.Timestamp, row: pd.Series, equity: float) -> ContextInputs:
        positions = []
        if st.position is not None:
            close_px = self._px(row, st.position.symbol, "close")
            positions.append({
                "symbol": st.position.symbol,
                "direction": st.position.direction,
                "entry_price": round(st.position.entry_price, 4),
                "shares": st.position.shares,
                "unrealized_pct": round(st.position.unrealized_pct(close_px) * 100.0, 4),
                "trade_id": st.position.trade_id,
            })
        can_open = (
            st.position is None and not self.risk.halted_today
            and self.risk.trades_today < self.cfg.risk.max_trades_per_day
            and self.risk.consecutive_losses < self.cfg.risk.max_consecutive_losses
        )
        risk_state = {
            "trades_today": self.risk.trades_today,
            "consecutive_losses": self.risk.consecutive_losses,
            "halted_today": self.risk.halted_today,
            "can_open_new_position": bool(can_open),
        }
        return ContextInputs(
            timestamp=t, row=row, symbols=self.context_symbols,
            positions=positions, daily_pnl=round(equity - self.risk.day_start_equity, 2),
            risk_state=risk_state,
        )

    def _px(self, row: pd.Series, symbol: str, field_: str) -> float:
        return float(row[price_col(symbol, field_)])

    def _equity(self, st: _State, row: pd.Series) -> float:
        if st.position is None:
            return st.cash
        return st.cash + st.position.shares * self._px(row, st.position.symbol, "close")

    def _execute_pending(self, st: _State, t: pd.Timestamp, i: int, row: pd.Series) -> None:
        if st.position is not None and st.pending_exit_reason is not None:
            open_px = self._px(row, st.position.symbol, "open")
            self._close_position(st, t, i, open_px, st.pending_exit_reason)
            st.pending_exit_reason = None
        if st.position is None and st.pending_entry is not None:
            intent = st.pending_entry
            open_px = self._px(row, intent.symbol, "open")
            self._open_position(st, t, i, intent, st.pending_entry_id, open_px)
            st.pending_entry = None
            st.pending_entry_id = 0

    def _open_position(
        self, st: _State, t: pd.Timestamp, i: int, intent: TradeIntent, trade_id: int, ref_px: float
    ) -> None:
        deploy = min(st.cash * self.cfg.backtest.fraction_per_trade, st.cash)
        fill = self.execution.fill_buy(ref_px, deploy)
        if fill.shares <= 0:
            return
        cost = fill.notional + fill.commission
        if cost > st.cash:
            return
        st.cash -= cost
        st.position = Position(
            symbol=intent.symbol, direction=intent.signal.direction, entry_time=t,
            entry_price=fill.price, shares=fill.shares, entry_bar=i, peak_price=fill.price,
            trade_id=trade_id, entry_reason=intent.signal.reason or intent.signal.action,
            entry_commission=fill.commission,
        )
        self.risk.on_open()

    def _close_position(self, st: _State, t: pd.Timestamp, i: int, ref_px: float, reason: str) -> None:
        pos = st.position
        assert pos is not None
        fill = self.execution.fill_sell(ref_px, pos.shares)
        proceeds = fill.notional - fill.commission
        st.cash += proceeds
        entry_notional = pos.shares * pos.entry_price
        entry_notional = pos.shares * pos.entry_price
        entry_cost = entry_notional + pos.entry_commission
        total_fees = pos.entry_commission + fill.commission
        net_pnl = proceeds - entry_cost
        return_pct = (net_pnl / entry_cost * 100.0) if entry_cost > 0 else 0.0
        st.trades.append(ClosedTrade(
            trade_id=pos.trade_id, symbol=pos.symbol, direction=pos.direction,
            entry_time=pos.entry_time, exit_time=t, entry_price=pos.entry_price,
            exit_price=fill.price, shares=pos.shares,
            gross_pnl=pos.shares * (fill.price - pos.entry_price), fees=total_fees,
            net_pnl=net_pnl,
            return_pct=return_pct,
            holding_minutes=(t - pos.entry_time).total_seconds() / 60.0,
            entry_reason=pos.entry_reason, exit_reason=reason,
        ))
        self.risk.on_close(pos.symbol, net_pnl, i, st.cash)
        st.position = None

    def _signal_log(self, signal: Signal, accepted: bool, reason: str, trade_id: int | None) -> dict:
        d = signal.to_dict()
        d["accepted"] = accepted
        d["rejection_reason"] = "" if accepted else reason
        d["trade_id"] = trade_id
        return d

    def _risk_log(
        self, st: _State, t: pd.Timestamp, signal: Signal, decision: str, reason: str,
        equity: float, trade_id: int | None,
    ) -> dict:
        return {
            "timestamp": t.isoformat(),
            "trade_id": trade_id,
            "action": signal.action,
            "symbol": signal.symbol,
            "confidence": signal.confidence,
            "decision": decision,
            "rejection_reason": "" if decision in ("ACCEPT", "NO_ACTION") else reason,
            "daily_pnl": round(equity - self.risk.day_start_equity, 2),
            "trades_today": self.risk.trades_today,
            "consecutive_losses": self.risk.consecutive_losses,
            "halted_today": self.risk.halted_today,
            "current_position": st.position.symbol if st.position else None,
        }

    def _daily_pnl(self, equity: pd.Series) -> pd.DataFrame:
        if equity.empty:
            return pd.DataFrame(columns=["date", "equity_close", "pnl", "return_pct"])
        day = pd.Series(equity.index.date, index=equity.index)
        eod = equity.groupby(day).last()
        prev = eod.shift(1).fillna(self.cfg.backtest.initial_cash)
        pnl = eod - prev
        ret = pnl / prev * 100.0
        return pd.DataFrame({
            "date": eod.index, "equity_close": eod.to_numpy(),
            "pnl": pnl.to_numpy(), "return_pct": ret.to_numpy(),
        }).reset_index(drop=True)
