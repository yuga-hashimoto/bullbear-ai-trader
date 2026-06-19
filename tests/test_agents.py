"""Mock / Replay agent behavior."""
from __future__ import annotations

import json

from src.agents.mock_agent import MockAgent
from src.agents.replay_agent import ReplayAgent
from src.agents.signal_schema import Signal


def _ctx(ts, qqq_close, qqq_vwap, ret3, rsi=55.0):
    # V9 entry needs full multi-timeframe alignment + RSI; mirror ret3 across lags.
    return {
        "timestamp": ts,
        "symbols": {
            "QQQ": {"close": qqq_close, "vwap": qqq_vwap, "rsi": rsi,
                    "returns": {"1_bar": ret3, "3_bar": ret3, "6_bar": ret3, "12_bar": ret3}},
        },
    }


def test_mock_agent_emits_valid_signal():
    agent = MockAgent()
    raw = agent.request_signal(_ctx("2026-01-01T10:00:00-05:00", 100.0, 99.0, 0.003))
    sig = Signal.from_dict(raw).validate()
    assert sig.action == "BUY_BULL"
    assert sig.symbol == "TQQQ"
    assert sig.confidence > 0.6


def test_mock_agent_no_trade_when_flat():
    agent = MockAgent()
    raw = agent.request_signal(_ctx("2026-01-01T10:00:00-05:00", 100.0, 100.0, 0.0))
    sig = Signal.from_dict(raw).validate()
    assert sig.action == "NO_TRADE"
    assert sig.symbol is None


def test_replay_agent_returns_recorded_signal(tmp_path):
    ts = "2026-01-01T10:00:00-05:00"
    rec = {"timestamp": ts, "agent_name": "Hermes", "target_family": "NASDAQ",
           "direction": "UP", "action": "BUY_BULL", "symbol": "TQQQ", "confidence": 0.9}
    f = tmp_path / "signals.jsonl"
    f.write_text(json.dumps(rec) + "\n")
    agent = ReplayAgent(f)
    raw = agent.request_signal({"timestamp": ts, "symbols": {}})
    assert raw["action"] == "BUY_BULL" and raw["symbol"] == "TQQQ"


def test_replay_agent_missing_timestamp_is_no_trade(tmp_path):
    f = tmp_path / "signals.jsonl"
    f.write_text(json.dumps({"timestamp": "2026-01-01T10:00:00-05:00",
                             "agent_name": "x", "target_family": "MARKET",
                             "direction": "FLAT", "action": "NO_TRADE"}) + "\n")
    agent = ReplayAgent(f)
    raw = agent.request_signal({"timestamp": "2099-01-01T00:00:00-05:00", "symbols": {}})
    assert raw["action"] == "NO_TRADE"


def test_replay_agent_skips_malformed_lines(tmp_path):
    f = tmp_path / "signals.jsonl"
    f.write_text("not json\n" + json.dumps({"timestamp": "t1", "action": "NO_TRADE",
                 "agent_name": "x", "target_family": "MARKET", "direction": "FLAT"}) + "\n")
    agent = ReplayAgent(f)
    assert agent.skipped_lines == 1
