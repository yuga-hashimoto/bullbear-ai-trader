"""Run output files + dashboard loader robustness."""
from __future__ import annotations

import dataclasses

import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import benchmark_comparison, compute_metrics
from src.config.settings import RiskConfig
from src.reports.loader import list_run_ids, load_run, resolve_run_id
from src.reports.report import write_reports
from src.reports.runs import new_run_id, run_dir, save_run
from tests.conftest import ConstantSignalAgent


def _run_and_save(cfg, labeled_matrix, agent, tmp_path):
    cfg = dataclasses.replace(cfg, paths={**cfg.paths, "reports_dir": str(tmp_path / "reports")})
    engine = BacktestEngine(cfg, agent)
    result = engine.run(labeled_matrix)
    metrics = compute_metrics(result, 5)
    bench = benchmark_comparison({}, [])
    run_id = new_run_id()
    d = run_dir(cfg, run_id)
    paths = write_reports(d, metrics, bench, result.trades_frame, result.daily_pnl,
                          counters=result.counters)
    save_run(cfg, run_id, result, metrics, bench, {"run_id": run_id}, paths)
    return cfg, run_id


def test_backtest_writes_all_run_files(cfg, labeled_matrix, tmp_path):
    agent = ConstantSignalAgent(action="BUY_BULL", symbol="TQQQ", family="NASDAQ",
                                direction="UP", confidence=0.99)
    cfg2, run_id = _run_and_save(cfg, labeled_matrix, agent, tmp_path)
    d = run_dir(cfg2, run_id)
    for fname in ("config.yaml", "summary.json", "metrics.json", "trades.csv",
                  "daily_pnl.csv", "equity_curve.csv", "agent_signals.jsonl",
                  "risk_decisions.jsonl", "report.html", "report.md"):
        assert (d / fname).exists(), f"missing {fname}"
    # latest pointer + loader.
    assert resolve_run_id(cfg2.path("reports_dir"), "latest") == run_id
    run = load_run(cfg2.path("reports_dir"), "latest")
    assert run.metrics and not run.trades.empty
    assert not run.agent_signals.empty and not run.risk_decisions.empty


def test_dashboard_loader_handles_no_trade_run(cfg, labeled_matrix, tmp_path):
    cfg2, run_id = _run_and_save(cfg, labeled_matrix, ConstantSignalAgent("NO_TRADE"), tmp_path)
    run = load_run(cfg2.path("reports_dir"), run_id)
    assert run.trades.empty          # no trades, but load must not crash
    assert run.metrics["num_trades"] == 0
    assert not run.agent_signals.empty  # NO_TRADE signals are still logged


def test_dashboard_loader_handles_rejected_only_run(cfg, labeled_matrix, tmp_path):
    # Low-confidence BUY signals -> all rejected -> no trades.
    agent = ConstantSignalAgent(action="BUY_BULL", symbol="TQQQ", family="NASDAQ",
                                direction="UP", confidence=0.05)
    cfg2, run_id = _run_and_save(cfg, labeled_matrix, agent, tmp_path)
    run = load_run(cfg2.path("reports_dir"), run_id)
    assert run.trades.empty
    assert run.counters["rejected_signals"] > 0
    assert list_run_ids(cfg2.path("reports_dir")) == [run_id]
