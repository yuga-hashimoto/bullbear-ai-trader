"""Live per-bar shadow execution for challengers.

Challengers make BUY/SELL decisions on the *same* 5-minute bar as the live
Champion (PaperRunner), using identical prices and features. Each challenger
runs the full numeric -> fusion -> risk -> virtual-fill pipeline against its own
``config_patch`` (risk/strategy thresholds), with its own virtual PaperBroker
(no real capital, ever). The shared OpenCode analysis is reused — challengers
never call the AI themselves, so there is zero extra AI cost.

This module is deliberately isolated: the Champion's code path is untouched. A
failure here can never affect the live Champion (PaperRunner wraps the call in a
try/except). State survives restarts via ``registry/shadow_state.json``.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..backtest.portfolio import Position
from ..brokers.base import OrderSide, OrderStatus, PositionInfo
from ..brokers.paper_broker import PaperBroker
from ..config.settings import Config
from ..features.builder import price_col
from ..agents.signal_schema import Signal
from ..risk.engine import EntryContext, RiskEngine
from ..risk.sizing import PositionSizer, PositionSizingInput
from ..strategy.fusion import SignalFusion
from ..strategy.numeric import NumericSignalStrategy
from ..strategy.strategy import Strategy
from ..utils.logging import get_logger
from .champion import apply_patch
from .challenger import SHADOW
from .mutation_generator import generate_mutations
from .registry import EvolutionRegistry

log = get_logger(__name__)

_BUY_ACTIONS = {"BUY_BULL", "BUY_BEAR"}


class ShadowChallenger:
    """One challenger evaluated live, bar by bar, with a virtual broker."""

    def __init__(self, challenger_id: str, patch: dict[str, Any], base_cfg: Config,
                 capital: float, state: dict[str, Any] | None = None) -> None:
        self.challenger_id = challenger_id
        self.patch = patch
        self.cfg = apply_patch(base_cfg, patch)
        self.start_equity = capital
        self.symbols = self.cfg.symbols
        self.allowed_symbols = set(self.cfg.symbols)

        self.strategy = Strategy(self.cfg)
        self.risk = RiskEngine(self.cfg.risk)
        self.sizer = PositionSizer()
        self.numeric = NumericSignalStrategy.from_config(self.cfg.strategy)
        self.fusion = SignalFusion()

        self.broker = PaperBroker(cash=capital, costs=self.cfg.costs)
        self.position: Position | None = None
        self.trade_seq = 0
        self.current_bar = 0
        self.closed_trades = 0
        self.wins = 0
        self.realized_pnl = 0.0
        self.daily_stop = False
        self._current_day = None
        self.day_start_equity = capital
        if state:
            self._restore(state)

    # --------------------------------------------------------------- state
    def _restore(self, s: dict[str, Any]) -> None:
        self.trade_seq = int(s.get("trade_seq", 0))
        self.current_bar = int(s.get("current_bar", 0))
        self.closed_trades = int(s.get("closed_trades", 0))
        self.wins = int(s.get("wins", 0))
        self.realized_pnl = float(s.get("realized_pnl", 0.0))
        self.daily_stop = bool(s.get("daily_stop", False))
        self.start_equity = float(s.get("start_equity", self.start_equity))
        self.day_start_equity = float(s.get("day_start_equity", self.start_equity))
        cash = float(s.get("cash", self.broker.get_cash()))
        positions: list[PositionInfo] = []
        pos = s.get("position")
        if pos:
            self.position = Position(
                symbol=str(pos["symbol"]), direction=str(pos["direction"]),
                entry_time=pd.Timestamp(pos["entry_time"]),
                entry_price=float(pos["entry_price"]), shares=float(pos["shares"]),
                entry_bar=int(pos.get("entry_bar", 0)),
                peak_price=float(pos.get("peak_price", pos["entry_price"])),
                trade_id=int(pos.get("trade_id", 0)),
                entry_reason=str(pos.get("entry_reason", "")),
                entry_commission=float(pos.get("entry_commission", 0.0)),
                stop_price=pos.get("stop_price"),
                planned_loss_jpy=float(pos.get("planned_loss_jpy", 0.0)),
            )
            positions.append(PositionInfo(self.position.symbol, self.position.shares,
                                          self.position.entry_price, self.position.entry_commission))
        d = s.get("current_day")
        if d:
            self._current_day = pd.Timestamp(d).date()
        self.broker.restore_account(cash, positions)

    def dump_state(self) -> dict[str, Any]:
        return {
            "trade_seq": self.trade_seq, "current_bar": self.current_bar,
            "closed_trades": self.closed_trades, "wins": self.wins,
            "realized_pnl": self.realized_pnl, "daily_stop": self.daily_stop,
            "start_equity": self.start_equity, "day_start_equity": self.day_start_equity,
            "cash": self.broker.get_cash(),
            "current_day": str(self._current_day) if self._current_day else None,
            "position": self.position.__dict__ if self.position is not None else None,
        }

    # ----------------------------------------------------------------- bar
    def on_bar(self, now: datetime, row: pd.Series, context: dict[str, Any],
               analysis: dict[str, Any] | None, session: Any) -> None:
        self._roll_day(now)
        for sym in self.symbols:
            self.broker.set_price(sym, float(row[price_col(sym, "close")]))
        self.current_bar += 1

        session_close_str = session.close_dt.strftime("%H:%M")
        self._manage_exit(now, row, session, session_close_str)

        signal = Signal.from_dict(self.fusion.fuse(
            self.numeric.signal(context), analysis, context["timestamp"]))

        if signal.action == "EXIT" and self.position is not None:
            self._close(now, float(row[price_col(self.position.symbol, "close")]), "agent_exit")
            return
        if signal.action not in _BUY_ACTIONS:
            return
        if not self._entry_allowed(now, session):
            return
        if not self.risk.validate_signal(signal, self.allowed_symbols).ok:
            return
        intent = self.strategy.map_signal(signal)
        if intent.kind != "ENTER" or intent.symbol not in self.allowed_symbols:
            return
        self._maybe_open(now, row, intent, signal, session_close_str)

    def _entry_allowed(self, now, session) -> bool:
        if self.daily_stop or self.position is not None:
            return False
        if session.minutes_since_open(now) < self.cfg.risk.no_trade_first_minutes:
            return False
        if session.minutes_to_close(now) <= self.cfg.risk.no_new_entry_last_minutes:
            return False
        return True

    def _manage_exit(self, now, row, session, session_close_str) -> None:
        if self.position is None:
            return
        close_px = float(row[price_col(self.position.symbol, "close")])
        self.position = self.position.update_peak(close_px)
        dec = self.risk.check_exit(pd.Timestamp(now), self.position, close_px, session_close_str)
        force = (not self.cfg.risk.allow_overnight_positions
                 and session.minutes_to_close(now) <= self.cfg.risk.force_close_minutes_before_close)
        if dec.ok or force:
            reason = dec.reason if dec.ok else "force_close_eod"
            self._close(now, close_px, reason)

    def _maybe_open(self, now, row, intent, signal, session_close_str) -> None:
        atr_feat = f"feat__{intent.symbol}__atr_pct"
        atr_pct = float(row[atr_feat]) * 100.0 if atr_feat in row.index else 0.0
        ctx = EntryContext(
            now=pd.Timestamp(now), equity=self.broker.get_cash(),
            spread_pct=self.cfg.costs.spread_pct, atr_pct=atr_pct,
            n_open_positions=0, candidate_symbol=intent.symbol, current_bar=self.current_bar,
            session_open=self.cfg.market.regular_open, session_close=session_close_str,
            max_concurrent=self.cfg.strategy.max_concurrent_positions,
        )
        if not self.risk.check_entry(ctx).ok:
            return
        ref = float(row[price_col(intent.symbol, "close")])
        stop_pct = min(self.cfg.risk.max_loss_per_trade_pct, 99.0)
        stop_px = ref * (1.0 - stop_pct / 100.0)
        portfolio_budget = (self.cfg.account.initial_capital_jpy
                            * self.cfg.risk.max_portfolio_risk_pct / 100.0)
        sizing = self.sizer.size(PositionSizingInput(
            entry_price_usd=ref, stop_price_usd=stop_px,
            cash_usd=self.broker.get_cash() * self.cfg.backtest.fraction_per_trade,
            usd_jpy=self.cfg.account.usd_jpy_rate,
            max_trade_loss_jpy=self.cfg.risk.max_loss_per_trade_jpy,
            portfolio_risk_remaining_jpy=portfolio_budget,
            overnight_gap_pct=(self.cfg.risk.overnight_gap_risk_pct
                               if self.cfg.risk.allow_overnight_positions else 0.0),
        ))
        if sizing.quantity <= 0:
            return
        try:
            order = self.broker.submit_order(intent.symbol, OrderSide.BUY, sizing.quantity)
        except Exception as exc:  # noqa: BLE001 - virtual broker boundary
            log.warning("shadow %s submit failed: %s", self.challenger_id, exc)
            return
        if order.status != OrderStatus.FILLED:
            return
        pos_info = next((p for p in self.broker.get_positions() if p.symbol == intent.symbol), None)
        if pos_info is None:
            return
        self.trade_seq += 1
        self.position = Position(
            symbol=intent.symbol, direction=signal.direction, entry_time=pd.Timestamp(now),
            entry_price=pos_info.avg_price, shares=pos_info.quantity, entry_bar=self.current_bar,
            peak_price=pos_info.avg_price, trade_id=self.trade_seq,
            entry_reason=signal.reason or signal.action,
            entry_commission=pos_info.entry_commission, stop_price=stop_px,
            planned_loss_jpy=sizing.planned_loss_jpy,
        )
        self.risk.on_open()

    def _close(self, now, ref_px: float, reason: str) -> None:
        pos = self.position
        if pos is None:
            return
        try:
            order = self.broker.close_position(pos.symbol)
        except Exception as exc:  # noqa: BLE001 - virtual broker boundary
            log.warning("shadow %s close failed: %s", self.challenger_id, exc)
            return
        exit_px = order.fill_price if order and order.fill_price else ref_px
        net = float(order.meta.get("realized_pnl")
                    if order is not None and "realized_pnl" in order.meta
                    else pos.shares * (exit_px - pos.entry_price) - pos.entry_commission)
        self.position = None
        self.realized_pnl += net
        self.closed_trades += 1
        if net > 0:
            self.wins += 1
        self.risk.on_close(pos.symbol, net, self.current_bar, self.broker.get_cash())
        if (self.risk.halted_today
                or self.risk.consecutive_losses >= self.cfg.risk.max_consecutive_losses):
            self.daily_stop = True

    def _roll_day(self, now) -> None:
        d = now.date()
        if d != self._current_day:
            self.risk.new_day(self.equity(), d)
            self.day_start_equity = self.equity()
            self.daily_stop = False
            self._current_day = d

    # ------------------------------------------------------------- metrics
    def equity(self) -> float:
        eq = self.broker.get_cash()
        for p in self.broker.get_positions():
            eq += p.quantity * self.broker.get_market_data(p.symbol).get("last", p.avg_price)
        return eq

    def metrics(self) -> dict[str, Any]:
        eq = self.equity()
        return {
            "total_return_pct": round((eq - self.start_equity) / self.start_equity * 100.0, 4)
            if self.start_equity else 0.0,
            "win_rate_pct": round(self.wins / self.closed_trades * 100.0, 2)
            if self.closed_trades else 0.0,
            "num_trades": self.closed_trades,
            "net_pnl_after_costs": round(self.realized_pnl, 2),
            "equity": round(eq, 2),
            "open_position": self.position.symbol if self.position else None,
        }


class LiveShadowBook:
    """Manage the set of live shadow challengers and persist their state."""

    def __init__(self, cfg: Config, reports_dir: str | Path | None = None,
                 num_challengers: int | None = None, capital: float | None = None) -> None:
        self.base_cfg = cfg
        self.reports_dir = Path(reports_dir or cfg.path("reports_dir"))
        evo = cfg.raw.get("evolution", {}).get("live_shadow", {})
        self.num_challengers = int(num_challengers if num_challengers is not None
                                   else evo.get("num_challengers", 5))
        self.capital = float(capital if capital is not None
                             else evo.get("capital", cfg.backtest.initial_cash))
        self.registry = EvolutionRegistry(self.reports_dir)
        self.registry.ensure_champion()
        self.state_path = self.registry.dir / "shadow_state.json"
        self._state = self._load_state()
        self._ensure_challengers()
        self.challengers = self._build()
        self._day = None  # set on first bar; triggers daily resync with the registry

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _ensure_challengers(self) -> None:
        active = [c for c in self.registry.list_challengers() if c.status == SHADOW]
        missing = self.num_challengers - len(active)
        if missing <= 0:
            return
        seed = random.Random().randint(0, 10_000_000)
        for cand in generate_mutations(self.base_cfg, n=missing, seed=seed):
            chal = self.registry.create_challenger(cand["config_patch"], source="mutation",
                                                   notes=f"live-shadow from {cand['candidate_id']}")
            chal.status = SHADOW
            self.registry.update_challenger(chal)

    def _build(self) -> list[ShadowChallenger]:
        out: list[ShadowChallenger] = []
        for chal in self.registry.list_challengers():
            if chal.status != SHADOW:
                continue
            try:
                out.append(ShadowChallenger(chal.challenger_id, chal.config_patch,
                                            self.base_cfg, self.capital,
                                            self._state.get(chal.challenger_id)))
            except Exception as exc:  # noqa: BLE001 - never break the book on one bad patch
                log.warning("skip shadow challenger %s: %s", chal.challenger_id, exc)
        return out

    def _maybe_resync(self, now: datetime) -> None:
        """At each new trading day, follow the registry: keep surviving live
        challengers (with their in-memory state), add freshly spawned ones, drop
        retired ones. Lets the daily evolution churn take effect without restart.
        """
        d = now.date()
        if d == self._day:
            return
        self._day = d
        current = {sc.challenger_id: sc for sc in self.challengers}
        rebuilt: list[ShadowChallenger] = []
        for chal in self.registry.list_challengers():
            if chal.status != SHADOW:
                continue
            if chal.challenger_id in current:
                rebuilt.append(current[chal.challenger_id])
                continue
            try:
                rebuilt.append(ShadowChallenger(chal.challenger_id, chal.config_patch,
                                                self.base_cfg, self.capital,
                                                self._state.get(chal.challenger_id)))
            except Exception as exc:  # noqa: BLE001
                log.warning("skip resynced challenger %s: %s", chal.challenger_id, exc)
        if rebuilt:
            self.challengers = rebuilt

    def on_bar(self, now: datetime, row: pd.Series, context: dict[str, Any],
               analysis: dict[str, Any] | None, session: Any) -> None:
        self._maybe_resync(now)
        for sc in self.challengers:
            try:
                sc.on_bar(now, row, context, analysis, session)
            except Exception as exc:  # noqa: BLE001 - isolate one challenger's failure
                log.warning("shadow challenger %s bar error: %s", sc.challenger_id, exc)
        self._persist(now)

    def _persist(self, now: datetime) -> None:
        state: dict[str, Any] = {}
        ts = pd.Timestamp(now).isoformat()
        for sc in self.challengers:
            state[sc.challenger_id] = sc.dump_state()
            m = sc.metrics()
            chal = self.registry.get_challenger(sc.challenger_id)
            if chal is not None:
                chal.metrics = {**chal.metrics, **m, "updated_at": ts}
                self.registry.update_challenger(chal)
            self._append_pnl({"timestamp": ts, "challenger_id": sc.challenger_id, **m})
        self._state = state
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.replace(self.state_path)

    def _append_pnl(self, rec: dict[str, Any]) -> None:
        path = self.reports_dir / "evolution" / "shadow_pnl.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
