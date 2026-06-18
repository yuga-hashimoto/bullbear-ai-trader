from __future__ import annotations

from src.risk.sizing import PositionSizingInput, PositionSizer
from src.backtest.engine import BacktestEngine
from tests.conftest import ConstantSignalAgent


def test_sizes_from_jpy_loss_budget_and_stop_distance():
    result = PositionSizer().size(
        PositionSizingInput(
            entry_price_usd=100.0,
            stop_price_usd=98.0,
            cash_usd=10_000.0,
            usd_jpy=150.0,
            max_trade_loss_jpy=10_000.0,
            portfolio_risk_remaining_jpy=30_000.0,
        )
    )

    assert result.quantity == 33
    assert result.planned_loss_jpy == 9_900.0


def test_cash_and_portfolio_risk_cap_quantity():
    cash_capped = PositionSizer().size(
        PositionSizingInput(
            entry_price_usd=100.0,
            stop_price_usd=99.0,
            cash_usd=550.0,
            usd_jpy=150.0,
            max_trade_loss_jpy=10_000.0,
            portfolio_risk_remaining_jpy=30_000.0,
        )
    )
    portfolio_capped = PositionSizer().size(
        PositionSizingInput(
            entry_price_usd=100.0,
            stop_price_usd=98.0,
            cash_usd=10_000.0,
            usd_jpy=150.0,
            max_trade_loss_jpy=10_000.0,
            portfolio_risk_remaining_jpy=3_000.0,
        )
    )

    assert cash_capped.quantity == 5
    assert portfolio_capped.quantity == 10


def test_rejects_missing_or_invalid_stop():
    result = PositionSizer().size(
        PositionSizingInput(
            entry_price_usd=100.0,
            stop_price_usd=None,
            cash_usd=10_000.0,
            usd_jpy=150.0,
            max_trade_loss_jpy=10_000.0,
            portfolio_risk_remaining_jpy=30_000.0,
        )
    )

    assert result.quantity == 0
    assert result.reason == "missing_stop"


def test_overnight_gap_risk_reduces_quantity():
    intraday = PositionSizer().size(
        PositionSizingInput(
            entry_price_usd=100.0,
            stop_price_usd=98.0,
            cash_usd=10_000.0,
            usd_jpy=150.0,
            max_trade_loss_jpy=10_000.0,
            portfolio_risk_remaining_jpy=30_000.0,
        )
    )
    overnight = PositionSizer().size(
        PositionSizingInput(
            entry_price_usd=100.0,
            stop_price_usd=98.0,
            cash_usd=10_000.0,
            usd_jpy=150.0,
            max_trade_loss_jpy=10_000.0,
            portfolio_risk_remaining_jpy=30_000.0,
            overnight_gap_pct=5.0,
        )
    )

    assert overnight.quantity == 13
    assert overnight.quantity < intraday.quantity


def test_backtest_uses_jpy_trade_risk_budget(cfg, labeled_matrix):
    agent = ConstantSignalAgent(
        action="BUY_BULL",
        symbol="TQQQ",
        family="NASDAQ",
        direction="UP",
        confidence=0.99,
    )
    result = BacktestEngine(cfg, agent).run(labeled_matrix)

    assert result.trades
    first = result.trades[0]
    planned_stop_loss_jpy = (
        first.shares
        * first.entry_price
        * cfg.risk.max_loss_per_trade_pct
        / 100.0
        * cfg.account.usd_jpy_rate
    )
    assert planned_stop_loss_jpy <= cfg.risk.max_loss_per_trade_jpy
