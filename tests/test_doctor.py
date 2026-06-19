from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone

from src.ops.doctor import build_readiness
from src.runners.base import runtime_dir
from src.runners.heartbeat import EventType, RuntimeWriter


def _cfg(cfg, tmp_path):
    paths = {
        **cfg.paths,
        "features_dir": str(tmp_path / "features"),
        "reports_dir": str(tmp_path / "reports"),
    }
    return dataclasses.replace(cfg, paths=paths)


def _heartbeat(**overrides):
    hb = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "market_state": "open",
        "reason": "",
        "last_processed_bar_time": datetime.now(timezone.utc).isoformat(),
    }
    hb.update(overrides)
    return hb


def test_readiness_reports_live_disabled_and_missing_data(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    cfg2 = _cfg(cfg, tmp_path)

    result = build_readiness(cfg2)

    assert result["live_trading"]["enabled"] is False
    assert result["paper"]["ready"] is False
    codes = {item["code"] for item in result["checks"] if not item["ok"]}
    assert "features_missing" in codes


def test_readiness_is_json_serializable(cfg, tmp_path):
    paths = {**cfg.paths, "reports_dir": str(tmp_path / "reports")}
    result = build_readiness(dataclasses.replace(cfg, paths=paths))

    json.dumps(result)


def test_error_heartbeat_is_not_reported_running(cfg, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    RuntimeWriter(runtime_dir(cfg2)).write_heartbeat(
        _heartbeat(status="error", reason="stale_data")
    )

    result = build_readiness(cfg2)

    assert result["paper"]["running"] is False
    failed = {c["code"] for c in result["checks"] if not c["ok"]}
    assert "paper_status" in failed
    assert "paper_reason" in failed


def test_open_market_requires_a_processed_bar(cfg, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    RuntimeWriter(runtime_dir(cfg2)).write_heartbeat(
        _heartbeat(last_processed_bar_time=None)
    )

    result = build_readiness(cfg2)

    assert result["paper"]["running"] is False
    assert any(
        c["code"] == "paper_processed_bar" and not c["ok"]
        for c in result["checks"]
    )


def test_recent_runtime_error_makes_running_false(cfg, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    writer = RuntimeWriter(runtime_dir(cfg2))
    writer.write_heartbeat(_heartbeat())
    writer.emit(EventType.AGENT_ERROR, {"reason": "timeout"})

    result = build_readiness(cfg2)

    assert result["paper"]["running"] is False
    assert any(
        c["code"] == "recent_runtime_errors" and not c["ok"]
        for c in result["checks"]
    )


def test_restart_loop_makes_running_false(cfg, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    writer = RuntimeWriter(runtime_dir(cfg2))
    writer.write_heartbeat(_heartbeat())
    for _ in range(3):
        writer.emit(EventType.RUNNER_STARTED)
        writer.emit(EventType.RUNNER_STOPPED)

    result = build_readiness(cfg2)

    assert result["paper"]["running"] is False
    assert any(
        c["code"] == "restart_loop" and not c["ok"]
        for c in result["checks"]
    )


def test_closed_market_sleeping_heartbeat_is_healthy(cfg, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)
    RuntimeWriter(runtime_dir(cfg2)).write_heartbeat(
        _heartbeat(
            status="sleeping",
            market_state="closed",
            last_processed_bar_time=None,
        )
    )

    result = build_readiness(cfg2)

    assert result["paper"]["running"] is True
