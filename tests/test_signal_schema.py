"""Signal schema validation: normal + abnormal cases."""
from __future__ import annotations

import pytest

from src.agents.signal_schema import Signal, SignalValidationError


def _base(**over):
    d = {
        "timestamp": "2026-01-01T14:35:00-05:00",
        "agent_name": "HermesAgent",
        "target_family": "NASDAQ",
        "direction": "UP",
        "action": "BUY_BULL",
        "symbol": "TQQQ",
        "confidence": 0.8,
    }
    d.update(over)
    return d


def test_valid_buy_bull():
    s = Signal.from_dict(_base()).validate()
    assert s.symbol == "TQQQ" and s.action == "BUY_BULL"


def test_valid_no_trade_null_symbol():
    s = Signal.from_dict(_base(action="NO_TRADE", direction="FLAT", symbol=None,
                               target_family="MARKET")).validate()
    assert s.action == "NO_TRADE" and s.symbol is None


def test_missing_required_field():
    d = _base()
    del d["action"]
    with pytest.raises(SignalValidationError):
        Signal.from_dict(d)


def test_invalid_action():
    with pytest.raises(SignalValidationError):
        Signal.from_dict(_base(action="SELL_SHORT")).validate()


def test_invalid_symbol_for_bull():
    # BUY_BULL with a bear symbol must be rejected.
    with pytest.raises(SignalValidationError):
        Signal.from_dict(_base(symbol="SQQQ")).validate()


def test_confidence_out_of_range():
    with pytest.raises(SignalValidationError):
        Signal.from_dict(_base(confidence=1.5)).validate()


def test_family_symbol_mismatch():
    # SEMICONDUCTOR family must use SOXL/SOXS, not TQQQ.
    with pytest.raises(SignalValidationError):
        Signal.from_dict(_base(target_family="SEMICONDUCTOR", symbol="TQQQ")).validate()


def test_no_trade_with_symbol_rejected():
    with pytest.raises(SignalValidationError):
        Signal.from_dict(_base(action="NO_TRADE", symbol="TQQQ")).validate()


def test_buy_bear_requires_bear_symbol():
    s = Signal.from_dict(_base(action="BUY_BEAR", direction="DOWN", symbol="SQQQ")).validate()
    assert s.symbol == "SQQQ"
