"""PaperRunner behavior: market gating, entry windows, force close, stale data,
agent timeout, dedup, heartbeat, safe stop, and live-runner refusal."""
from __future__ import annotations

import dataclasses
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.agents.base import BaseAgent
from src.agents.signal_schema import no_trade_signal
from src.backtest.portfolio import Position
from src.config.settings import LiveTradingDisabledError
from src.runners.base import request_stop, runner_disabled, runtime_dir
from src.runners.feed import FrozenFramesFeed
from src.runners.heartbeat import RuntimeWriter
from src.runners.live_runner import LiveRunner
from src.runners.paper_runner import PaperRunner
from tests.conftest import ConstantSignalAgent

ET = ZoneInfo("America/New_York")


def _dt(hh, mm=0, day=10):
    return datetime(2024, 1, day, hh, mm, tzinfo=ET)


class CountingAgent(BaseAgent):
    name = "CountingAgent"

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def request_signal(self, context):
        self.calls += 1
        return self.inner.request_signal(context)


class SlowAgent(BaseAgent):
    name = "SlowAgent"

    def __init__(self, delay):
        self.delay = delay

    def request_signal(self, context):
        time.sleep(self.delay)
        return no_trade_signal(context["timestamp"], self.name).to_dict()


def _cfg(cfg, tmp_path, **runner_over):
    paths = {**cfg.paths, "reports_dir": str(tmp_path / "reports")}
    cfg = dataclasses.replace(cfg, paths=paths)
    if runner_over:
        cfg = dataclasses.replace(cfg, runner=dataclasses.replace(cfg.runner, **runner_over))
    return cfg


def _bull():
    return ConstantSignalAgent(action="BUY_BULL", symbol="TQQQ", family="NASDAQ",
                               direction="UP", confidence=0.99)


def _runner(cfg, agent, frames):
    return PaperRunner(cfg, agent, feed=FrozenFramesFeed(frames, tz=cfg.runner.timezone))


def test_does_not_call_agent_when_market_closed(cfg, frames, tmp_path):
    agent = CountingAgent(_bull())
    r = _runner(_cfg(cfg, tmp_path), agent, frames)
    res = r.step(_dt(20, 0))  # 8pm ET = closed
    assert res["action"] == "sleep"
    assert agent.calls == 0
    assert r.position is None


def test_calls_agent_and_opens_during_market_hours(cfg, frames, tmp_path):
    agent = CountingAgent(_bull())
    r = _runner(_cfg(cfg, tmp_path), agent, frames)
    res = r.step(_dt(11, 0))
    assert res["action"] == "processed"
    assert agent.calls == 1
    assert r.position is not None and r.position.symbol == "TQQQ"
    # heartbeat written.
    assert RuntimeWriter(runtime_dir(r.cfg)).read_heartbeat()["status"] == "running"


def test_no_trade_first_minutes_blocks_entry(cfg, frames, tmp_path):
    r = _runner(_cfg(cfg, tmp_path), _bull(), frames)
    r.step(_dt(9, 35))  # 5 min after open < 10
    assert r.position is None


def test_no_new_entry_last_minutes_blocks_entry(cfg, frames, tmp_path):
    r = _runner(_cfg(cfg, tmp_path), _bull(), frames)
    r.step(_dt(15, 40))  # 20 min to close <= 30
    assert r.position is None


def test_force_close_before_close(cfg, frames, tmp_path):
    r = _runner(_cfg(cfg, tmp_path), ConstantSignalAgent("NO_TRADE"), frames)
    r.position = Position(symbol="TQQQ", direction="UP", entry_time=pd.Timestamp(_dt(14, 0)),
                          entry_price=50.0, shares=10, entry_bar=0, peak_price=50.0,
                          trade_id=1, entry_reason="test")
    r.step(_dt(15, 57))  # 3 min to close <= 5
    assert r.position is None


def test_overnight_setting_skips_force_close(cfg, frames, tmp_path):
    risk = dataclasses.replace(cfg.risk, allow_overnight_positions=True)
    cfg2 = dataclasses.replace(_cfg(cfg, tmp_path), risk=risk)
    r = _runner(cfg2, ConstantSignalAgent("NO_TRADE"), frames)

    session = r.calendar.session_for_date(_dt(15, 57).date())

    assert r._should_force_close(_dt(15, 57), session) is False


def test_stale_data_no_trade(cfg, frames, tmp_path):
    # Feed only old bars (up to Jan 3) while "now" is Jan 10 -> stale.
    old = {s: df[df.index <= pd.Timestamp("2024-01-03 16:00", tz=ET)] for s, df in frames.items()}
    r = _runner(_cfg(cfg, tmp_path), _bull(), old)
    res = r.step(_dt(11, 0))
    assert res["action"] == "stale"
    assert r.position is None


def test_bar_start_timestamp_is_fresh_through_close_plus_vendor_delay(
    cfg, frames, tmp_path
):
    last_bar = pd.Timestamp("2024-01-10 09:30", tz=ET)
    limited = {s: df[df.index <= last_bar] for s, df in frames.items()}
    cfg2 = _cfg(
        cfg,
        tmp_path,
        stale_data_threshold_seconds=180,
        vendor_delay_seconds=600,
    )
    r = _runner(cfg2, ConstantSignalAgent("NO_TRADE"), limited)

    res = r.step(_dt(9, 39))

    assert res["action"] in {"processed", "warmup"}
    assert r.data_errors == 0


def test_repeated_stale_data_waits_without_stopping_runner(cfg, frames, tmp_path):
    old = {s: df[df.index <= pd.Timestamp("2024-01-03 16:00", tz=ET)] for s, df in frames.items()}
    r = _runner(
        _cfg(cfg, tmp_path, max_data_errors_before_stop=3),
        ConstantSignalAgent("NO_TRADE"),
        old,
    )

    for _ in range(3):
        assert r.step(_dt(11, 0))["action"] == "stale"

    assert not r.should_stop()
    heartbeat = RuntimeWriter(runtime_dir(r.cfg)).read_heartbeat()
    assert heartbeat["status"] == "waiting"
    assert heartbeat["reason"] == "stale_data"


def test_agent_timeout_results_in_no_trade(cfg, frames, tmp_path):
    cfg2 = _cfg(cfg, tmp_path, max_agent_latency_seconds=1)
    r = _runner(cfg2, SlowAgent(delay=1.5), frames)
    res = r.step(_dt(11, 0))
    assert res["action"] == "processed"
    assert r.position is None
    assert r.agent_errors >= 1


def test_duplicate_bar_not_processed_twice(cfg, frames, tmp_path):
    agent = CountingAgent(_bull())
    r = _runner(_cfg(cfg, tmp_path), agent, frames)
    r.step(_dt(11, 0))
    calls_after_first = agent.calls
    res = r.step(_dt(11, 0))  # same bar
    assert res["action"] == "wait_bar"
    assert agent.calls == calls_after_first  # not re-processed
    heartbeat = RuntimeWriter(runtime_dir(r.cfg)).read_heartbeat()
    assert heartbeat["reason"] == ""


def test_stop_runner_flag_and_safe_stop(cfg, frames, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    r = _runner(cfg2, ConstantSignalAgent("NO_TRADE"), frames)
    assert not r.should_stop()
    request_stop(cfg2)
    assert r.should_stop()


def test_run_loop_exits_and_emits_stopped(cfg, frames, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    r = _runner(cfg2, ConstantSignalAgent("NO_TRADE"), frames)
    r.clock = lambda: _dt(20, 0)        # always closed
    r.sleeper = lambda secs: r.stop()   # stop after first iteration
    r.run()
    events = (runtime_dir(cfg2) / "paper_events.jsonl").read_text()
    assert "RUNNER_STOPPED" in events and "RUNNER_STARTED" in events


def test_run_does_not_clear_persistent_disable_flag(cfg, frames, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    request_stop(cfg2)
    r = _runner(cfg2, ConstantSignalAgent("NO_TRADE"), frames)

    r.run()

    assert runner_disabled(cfg2)


def test_live_runner_refuses_to_start(cfg, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    with pytest.raises(LiveTradingDisabledError):
        LiveRunner(cfg2, enable_live_trading=True)  # env + allow_live_orders still block


def test_dashboard_has_no_order_controls():
    # The viewer must not be able to place orders / start live trading.
    src = open("src/reports/dashboard.py").read()
    for forbidden in ("submit_order", "run-live", "MoomooBroker", "close_position"):
        assert forbidden not in src, f"dashboard must not reference {forbidden}"
