"""Dashboard must render for normal / NO_TRADE-only / rejected-only runs.

Skipped automatically when streamlit is not installed (it is an optional,
viewer-only dependency).
"""
from __future__ import annotations

import dataclasses

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
