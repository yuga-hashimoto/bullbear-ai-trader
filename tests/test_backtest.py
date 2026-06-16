"""Backtest engine: NO_TRADE, single position, no overnight, repeatable,
risk-rejection blocks orders."""
from __future__ import annotations

import dataclasses

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.config.settings import RiskConfig
from tests.conftest import ConstantSignalAgent


def _bull_agent(confidence=0.99):
    return ConstantSignalAgent(action="BUY_BULL", symbol="TQQQ",
                               family="NASDAQ", direction="UP", confidence=confidence)


def test_no_trade_agent_produces_no_trades(cfg, labeled_matrix):
    engine = BacktestEngine(cfg, ConstantSignalAgent(action="NO_TRADE"))
    result = engine.run(labeled_matrix)
    assert len(result.trades) == 0
    assert abs(result.equity_curve.iloc[-1] - cfg.backtest.initial_cash) < 1e-6
    assert result.counters["no_trade_count"] == result.counters["num_signals"]


def test_low_confidence_signal_is_rejected_no_order(cfg, labeled_matrix):
    # confidence below threshold -> risk rejects -> zero trades.
    engine = BacktestEngine(cfg, _bull_agent(confidence=0.10))
    result = engine.run(labeled_matrix)
    assert len(result.trades) == 0
    assert result.counters["rejected_signals"] > 0
    assert "low_confidence" in result.counters["risk_rejection_reasons"]


def test_bull_agent_trades_single_position_no_overnight(cfg, labeled_matrix):
    engine = BacktestEngine(cfg, _bull_agent())
    result = engine.run(labeled_matrix)
    assert result.trades, "expected trades from a high-confidence BUY_BULL agent"
    trades = sorted(result.trades, key=lambda t: t.entry_time)
    for a, b in zip(trades, trades[1:]):
        assert a.exit_time <= b.entry_time, "overlapping positions"
    for t in result.trades:
        assert t.entry_time.date() == t.exit_time.date(), "held overnight"


def test_force_close_eod_with_loose_risk(cfg, labeled_matrix):
    loose = RiskConfig(
        confidence_threshold=0.0, max_loss_per_trade_pct=999.0, take_profit_pct=999.0,
        trailing_stop_pct=999.0, max_daily_loss_pct=999.0, max_trades_per_day=999,
        max_consecutive_losses=999, max_holding_minutes=999999, no_trade_first_minutes=0,
        no_new_entry_last_minutes=10, force_close_minutes_before_close=5,
        max_spread_pct=999.0, max_atr_pct=999.0, min_bars_between_same_symbol=0,
    )
    cfg_loose = dataclasses.replace(cfg, risk=loose)
    engine = BacktestEngine(cfg_loose, _bull_agent())
    result = engine.run(labeled_matrix)
    reasons = {t.exit_reason for t in result.trades}
    assert reasons & {"force_close_eod", "force_close_before_close"}
    assert result.counters["forced_exits"] >= 1


def test_reproducible(cfg, labeled_matrix):
    r1 = BacktestEngine(cfg, _bull_agent()).run(labeled_matrix)
    r2 = BacktestEngine(cfg, _bull_agent()).run(labeled_matrix)
    pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)
    assert len(r1.trades) == len(r2.trades)
