"""Risk engine: entry gates, exit gates, daily halt, consecutive losses."""
from __future__ import annotations

import pandas as pd

from src.agents.signal_schema import Signal
from src.config.settings import RiskConfig
from src.risk.engine import EntryContext, RiskEngine
from src.backtest.portfolio import Position

ALLOWED = {"TQQQ", "SQQQ", "SOXL", "SOXS"}


def _sig(**over):
    d = {"timestamp": "t", "agent_name": "a", "target_family": "NASDAQ",
         "direction": "UP", "action": "BUY_BULL", "symbol": "TQQQ", "confidence": 0.8}
    d.update(over)
    return Signal.from_dict(d)


def test_validate_signal_accepts_good_entry():
    eng = RiskEngine(RiskConfig(confidence_threshold=0.65))
    assert eng.validate_signal(_sig(), ALLOWED).ok


def test_validate_signal_low_confidence():
    eng = RiskEngine(RiskConfig(confidence_threshold=0.65))
    assert eng.validate_signal(_sig(confidence=0.4), ALLOWED).reason == "low_confidence"


def test_validate_signal_symbol_not_allowed():
    eng = RiskEngine(RiskConfig(confidence_threshold=0.65))
    assert eng.validate_signal(_sig(), {"SOXL"}).reason == "symbol_not_allowed"


def test_validate_signal_family_mismatch():
    eng = RiskEngine(RiskConfig(confidence_threshold=0.65))
    # NASDAQ family but SOXL symbol -> mismatch.
    assert eng.validate_signal(_sig(symbol="SOXL"), ALLOWED).reason == "family_symbol_mismatch"

TZ = "America/New_York"


def _engine() -> RiskEngine:
    eng = RiskEngine(RiskConfig())
    eng.new_day(100000.0, pd.Timestamp("2024-01-02 09:30", tz=TZ).date())
    return eng


def _ctx(now, **over) -> EntryContext:
    base = dict(
        now=now, equity=100000.0, spread_pct=0.03, atr_pct=1.0,
        n_open_positions=0, candidate_symbol="TQQQ", current_bar=10,
        session_open="09:30", session_close="16:00", max_concurrent=1,
    )
    base.update(over)
    return EntryContext(**base)


def test_no_trade_first_minutes():
    eng = _engine()
    now = pd.Timestamp("2024-01-02 09:35", tz=TZ)  # 5 min after open < 10
    assert eng.check_entry(_ctx(now)).reason == "no_trade_first_minutes"


def test_no_new_entry_last_minutes():
    eng = _engine()
    now = pd.Timestamp("2024-01-02 15:40", tz=TZ)  # 20 min to close <= 30
    assert eng.check_entry(_ctx(now)).reason == "no_new_entry_last_minutes"


def test_blocks_when_position_open():
    eng = _engine()
    now = pd.Timestamp("2024-01-02 11:00", tz=TZ)
    assert eng.check_entry(_ctx(now, n_open_positions=1)).reason == "position_already_open"


def test_volatility_and_spread_gates():
    eng = _engine()
    now = pd.Timestamp("2024-01-02 11:00", tz=TZ)
    assert eng.check_entry(_ctx(now, atr_pct=99.0)).reason == "volatility_too_high"
    assert eng.check_entry(_ctx(now, spread_pct=99.0)).reason == "spread_too_wide"


def test_max_trades_and_consecutive_losses():
    eng = _engine()
    now = pd.Timestamp("2024-01-02 11:00", tz=TZ)
    assert eng.check_entry(_ctx(now)).ok
    # Two losses -> consecutive-loss block (default max_consecutive_losses=2).
    eng.on_close("TQQQ", -100.0, 1, 99900.0)
    eng.on_close("TQQQ", -100.0, 2, 99800.0)
    assert eng.check_entry(_ctx(now)).reason == "max_consecutive_losses"


def test_daily_loss_halt():
    eng = _engine()
    now = pd.Timestamp("2024-01-02 11:00", tz=TZ)
    # Lose > 2% of day-start equity in one trade.
    eng.on_close("TQQQ", -2500.0, 1, 97500.0)
    assert eng.halted_today
    assert eng.check_entry(_ctx(now)).reason == "daily_loss_halt"


def _pos(entry_price: float) -> Position:
    return Position(
        symbol="TQQQ", direction="UP",
        entry_time=pd.Timestamp("2024-01-02 11:00", tz=TZ),
        entry_price=entry_price, shares=100.0, entry_bar=5, peak_price=entry_price,
    )


def test_stop_loss_and_take_profit():
    eng = _engine()
    now = pd.Timestamp("2024-01-02 11:30", tz=TZ)
    pos = _pos(100.0)
    assert eng.check_exit(now, pos, 99.0, "16:00").reason == "stop_loss"      # -1%
    assert eng.check_exit(now, pos, 101.5, "16:00").reason == "take_profit"   # +1.5%


def test_force_close_before_close():
    eng = _engine()
    now = pd.Timestamp("2024-01-02 15:57", tz=TZ)  # 3 min to close <= 5
    pos = _pos(100.0)
    assert eng.check_exit(now, pos, 100.0, "16:00").reason == "force_close_before_close"
