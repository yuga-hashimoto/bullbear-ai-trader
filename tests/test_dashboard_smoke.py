"""Dashboard must render for normal / NO_TRADE-only / rejected-only runs.

Skipped automatically when streamlit is not installed (it is an optional,
viewer-only dependency).
"""
from __future__ import annotations

import dataclasses
import json

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.metrics import benchmark_comparison, compute_metrics  # noqa: E402
from src.reports.report import write_reports  # noqa: E402
from src.reports.runs import new_run_id, run_dir, save_run  # noqa: E402
from tests.conftest import ConstantSignalAgent  # noqa: E402

DASHBOARD = "src/reports/dashboard.py"


def _make_run(cfg, labeled_matrix, agent, reports_dir):
    cfg = dataclasses.replace(cfg, paths={**cfg.paths, "reports_dir": str(reports_dir)})
    result = BacktestEngine(cfg, agent).run(labeled_matrix)
    metrics = compute_metrics(result, 5)
    bench = benchmark_comparison({}, [])
    run_id = new_run_id()
    d = run_dir(cfg, run_id)
    paths = write_reports(d, metrics, bench, result.trades_frame, result.daily_pnl,
                          counters=result.counters)
    save_run(cfg, run_id, result, metrics, bench, {"run_id": run_id}, paths)
    # Seed a minimal evolution registry so the Evolution tab renders populated.
    from src.evolution.registry import EvolutionRegistry
    reg = EvolutionRegistry(reports_dir)
    reg.ensure_champion()
    reg.create_challenger({"risk.confidence_threshold": 0.7}, source="manual")


@pytest.mark.parametrize("agent", [
    ConstantSignalAgent(action="BUY_BULL", symbol="TQQQ", family="NASDAQ",
                        direction="UP", confidence=0.99),                 # has trades
    ConstantSignalAgent(action="NO_TRADE"),                               # NO_TRADE only
    ConstantSignalAgent(action="BUY_BULL", symbol="TQQQ", family="NASDAQ",
                        direction="UP", confidence=0.05),                 # rejected only
])
def test_dashboard_renders(cfg, labeled_matrix, agent, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    _make_run(cfg, labeled_matrix, agent, reports_dir)
    monkeypatch.setenv("BULLBEAR_REPORTS_DIR", str(reports_dir))
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception, at.exception


def test_dashboard_labels_live_paper_metrics_separately_from_backtest(
    cfg, labeled_matrix, tmp_path, monkeypatch
):
    reports_dir = tmp_path / "reports"
    _make_run(cfg, labeled_matrix, ConstantSignalAgent("NO_TRADE"), reports_dir)
    runtime = reports_dir / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "heartbeat.json").write_text(json.dumps({
        "timestamp": "2026-06-18T10:12:56-04:00",
        "status": "running",
        "daily_pnl": -8.76,
        "daily_pnl_jpy": -1313.93,
        "trades_today": 1,
    }))
    (runtime / "daily_state.json").write_text(json.dumps({
        "marked_equity": 6666.67 - 8.76,
    }))
    (runtime / "paper_events.jsonl").write_text(json.dumps({
        "event": "POSITION_CLOSED",
        "net_pnl": -8.76,
    }))
    monkeypatch.setenv("BULLBEAR_REPORTS_DIR", str(reports_dir))

    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()

    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["💹 ペーパー運用の利益率（リアルタイム）"] == "-0.13%"
    assert metrics["🎯 ペーパー決済勝率"] == "0.0%"
    assert not any("最新バックテスト" in markdown.value for markdown in at.markdown)
    assert not any("これまでの成績" in tab.label for tab in at.tabs)
