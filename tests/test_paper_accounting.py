from __future__ import annotations

import dataclasses
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.portfolio import Position
from src.brokers.base import OrderSide
from src.brokers.paper_broker import PaperBroker
from src.runners.feed import FrozenFramesFeed
from src.runners.paper_runner import PaperRunner
from src.runners.base import runtime_dir
from src.runners.heartbeat import RuntimeWriter
from tests.conftest import ConstantSignalAgent

ET = ZoneInfo("America/New_York")


def test_sell_order_reports_realized_pnl_after_entry_and_exit_commissions(cfg):
    broker = PaperBroker(cash=10_000.0, costs=cfg.costs)
    broker.set_price("TQQQ", 100.0)
    buy = broker.submit_order("TQQQ", OrderSide.BUY, 10)
    broker.set_price("TQQQ", 101.0)
    sell = broker.close_position("TQQQ")

    assert sell is not None
    expected = (
        sell.meta["notional"]
        - sell.meta["commission"]
        - buy.meta["notional"]
        - buy.meta["commission"]
    )
    assert sell.meta["realized_pnl"] == expected


def test_unrealized_daily_loss_triggers_emergency_stop(cfg, frames, tmp_path):
    paths = {**cfg.paths, "reports_dir": str(tmp_path / "reports")}
    risk = dataclasses.replace(cfg.risk, max_daily_loss_jpy=1_000.0)
    cfg2 = dataclasses.replace(cfg, paths=paths, risk=risk)
    runner = PaperRunner(
        cfg2,
        ConstantSignalAgent("NO_TRADE"),
        feed=FrozenFramesFeed(frames, tz=cfg2.runner.timezone),
    )
    now = datetime(2024, 1, 10, 11, 0, tzinfo=ET)
    runner.broker.set_price("TQQQ", 100.0)
    order = runner.broker.submit_order("TQQQ", OrderSide.BUY, 10)
    pos = runner.broker.get_positions()[0]
    from src.backtest.portfolio import Position

    runner.position = Position(
        symbol="TQQQ",
        direction="UP",
        entry_time=pd.Timestamp(now),
        entry_price=pos.avg_price,
        shares=pos.quantity,
        entry_bar=0,
        peak_price=pos.avg_price,
        entry_commission=pos.entry_commission,
    )
    runner.day_start_equity = runner.broker.get_cash() + pos.quantity * pos.avg_price
    runner.broker.set_price("TQQQ", 90.0)

    runner._check_runtime_circuit_breakers(now)

    assert runner.daily_stop is True
    assert runner.position is None
    assert order.status.value == "FILLED"


def test_runner_restores_paper_position_and_cash(cfg, frames, tmp_path):
    paths = {**cfg.paths, "reports_dir": str(tmp_path / "reports")}
    cfg2 = dataclasses.replace(cfg, paths=paths)
    writer = RuntimeWriter(runtime_dir(cfg2))
    writer.write_json("daily_state.json", {
        "date": "2024-01-10",
        "cash": 9_000.0,
        "day_start_equity": 10_000.0,
        "peak_equity": 10_100.0,
        "trades_today": 1,
        "consecutive_losses": 0,
        "daily_stop": False,
    })
    writer.write_json("runner_position.json", {
        "symbol": "TQQQ",
        "direction": "UP",
        "entry_time": "2024-01-10T11:00:00-05:00",
        "entry_price": 100.0,
        "shares": 10.0,
        "entry_bar": 3,
        "peak_price": 101.0,
        "trade_id": 1,
        "entry_reason": "test",
        "entry_commission": 0.5,
        "stop_price": 99.0,
        "planned_loss_jpy": 1_500.0,
    })

    restored = PaperRunner(
        cfg2,
        ConstantSignalAgent("NO_TRADE"),
        feed=FrozenFramesFeed(frames, tz=cfg2.runner.timezone),
    )

    assert restored.broker.get_cash() == 9_000.0
    assert restored.position is not None
    assert restored.position.symbol == "TQQQ"
    assert restored.broker.get_positions()[0].quantity == 10.0


def test_position_snapshot_contains_dashboard_fields(cfg, frames, tmp_path):
    paths = {**cfg.paths, "reports_dir": str(tmp_path / "reports")}
    cfg2 = dataclasses.replace(cfg, paths=paths)
    runner = PaperRunner(
        cfg2,
        ConstantSignalAgent("NO_TRADE"),
        feed=FrozenFramesFeed(frames, tz=cfg2.runner.timezone),
    )
    now = datetime(2024, 1, 10, 11, 0, tzinfo=ET)
    runner.broker.set_price("SOXL", 102.0)
    runner.position = Position(
        symbol="SOXL",
        direction="UP",
        entry_time=pd.Timestamp(now),
        entry_price=100.0,
        shares=4,
        entry_bar=1,
        peak_price=102.0,
        trade_id=2,
        entry_reason="test",
    )

    snapshot = runner._position_snapshots()

    assert snapshot == [{
        "symbol": "SOXL",
        "direction": "UP",
        "entry_price": 100.0,
        "current_price": 102.0,
        "shares": 4,
        "unrealized_pct": 2.0,
        "trade_id": 2,
    }]
