"""Dual-momentum runner: paper equity + recommendation persistence (no network)."""
from __future__ import annotations

import json

import pandas as pd

from src.runners.dual_momentum_runner import DualMomentumRunner
from src.strategy.dual_momentum import DualMomentumConfig


def _monthly(n: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2022-01-31", periods=n, freq="ME")
    return pd.DataFrame({
        "SPY": [100 * 1.01 ** i for i in range(n)],
        "QQQ": [100 * 1.03 ** i for i in range(n)],   # strongest
        "EFA": [100 * 1.00 ** i for i in range(n)],
        "EEM": [100 * 0.99 ** i for i in range(n)],
        "GLD": [100 * 1.005 ** i for i in range(n)],
        "TLT": [100 * 1.002 ** i for i in range(n)],
    }, index=idx)


def test_first_run_starts_paper_at_capital(tmp_path):
    # Paper account starts at the inception month at exactly `capital` (no
    # 20-year backtest masquerading as held assets).
    runner = DualMomentumRunner(reports_dir=tmp_path / "reports",
                                cfg=DualMomentumConfig(leverage=1.5, lookbacks=(1, 2, 3)),
                                capital=1_000_000.0)
    state = runner.run(monthly=_monthly())

    assert state["recommendation"]["asset"] == "QQQ"      # current pick (strongest)
    assert state["paper"]["equity"] == 1_000_000.0         # starts AT capital, no fake gains
    assert state["paper"]["months"] == 0
    assert state["backtest_reference"]["cagr_pct"] is not None  # reference shown separately
    saved = json.loads((tmp_path / "reports" / "runtime" / "dual_momentum.json").read_text())
    assert saved["inception_month"] == state["inception_month"]


def test_forward_paper_accrues_after_inception(tmp_path):
    # With an inception set in the past, the forward months accrue real returns.
    runner = DualMomentumRunner(reports_dir=tmp_path / "reports",
                                cfg=DualMomentumConfig(leverage=1.5, lookbacks=(1, 2, 3)),
                                capital=1_000_000.0)
    monthly = _monthly(n=24)
    state = runner.run(monthly=monthly, inception_override="2022-06")
    assert state["paper"]["months"] > 0
    assert state["paper"]["equity"] > 1_000_000.0          # uptrend forward -> grew
    assert len(state["equity_curve"]) >= 2


def test_runner_goes_to_cash_in_a_bear_universe(tmp_path):
    n = 24
    idx = pd.date_range("2022-01-31", periods=n, freq="ME")
    falling = pd.DataFrame({s: [100 * 0.97 ** i for i in range(n)] for s in
                            ("SPY", "QQQ", "EFA", "EEM", "GLD")}, index=idx)
    falling["TLT"] = [100 * 1.001 ** i for i in range(n)]
    runner = DualMomentumRunner(reports_dir=tmp_path / "reports",
                                cfg=DualMomentumConfig(leverage=1.5, lookbacks=(1, 2, 3)))
    state = runner.run(monthly=falling)
    assert state["recommendation"]["asset"] == "TLT"       # protected
    assert state["recommendation"]["is_risk_on"] is False
