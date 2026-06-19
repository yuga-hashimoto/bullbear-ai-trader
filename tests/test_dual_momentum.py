"""Dual-momentum strategy logic (synthetic prices, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.dual_momentum import (
    DualMomentumConfig,
    DualMomentumStrategy,
    performance,
)


def _monthly(series: dict[str, list[float]]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=len(next(iter(series.values()))), freq="ME")
    return pd.DataFrame(series, index=idx)


def test_picks_strongest_risky_asset():
    # QQQ rises fastest -> should be selected; lookbacks short for a small frame
    n = 15
    df = _monthly({
        "SPY": [100 * 1.01 ** i for i in range(n)],
        "QQQ": [100 * 1.05 ** i for i in range(n)],   # strongest momentum
        "EFA": [100 * 1.00 ** i for i in range(n)],
        "EEM": [100 * 0.99 ** i for i in range(n)],
        "GLD": [100 * 1.005 ** i for i in range(n)],
        "TLT": [100 * 1.002 ** i for i in range(n)],
    })
    strat = DualMomentumStrategy(DualMomentumConfig(lookbacks=(1, 2, 3)))
    rec = strat.recommend(df)
    assert rec.asset == "QQQ"
    assert rec.is_risk_on is True
    assert rec.leverage == 1.5
    assert rec.ranking[0][0] == "QQQ"        # ranked first


def test_absolute_momentum_filter_goes_to_safe_when_all_falling():
    n = 15
    df = _monthly({
        "SPY": [100 * 0.97 ** i for i in range(n)],
        "QQQ": [100 * 0.95 ** i for i in range(n)],
        "EFA": [100 * 0.96 ** i for i in range(n)],
        "EEM": [100 * 0.94 ** i for i in range(n)],
        "GLD": [100 * 0.98 ** i for i in range(n)],   # least-bad but still negative
        "TLT": [100 * 1.001 ** i for i in range(n)],
    })
    strat = DualMomentumStrategy(DualMomentumConfig(lookbacks=(1, 2, 3)))
    rec = strat.recommend(df)
    assert rec.asset == "TLT"                # stepped aside to bonds
    assert rec.is_risk_on is False
    assert rec.leverage == 1.0               # never leverage the safe asset


def test_leverage_amplifies_risky_return_net_of_costs():
    n = 18
    # QQQ steadily +2%/mo, everything else flat-ish so QQQ is always picked
    df = _monthly({
        "SPY": [100 * 1.001 ** i for i in range(n)],
        "QQQ": [100 * 1.02 ** i for i in range(n)],
        "EFA": [100 * 1.0 ** i for i in range(n)],
        "EEM": [100 * 1.0 ** i for i in range(n)],
        "GLD": [100 * 1.0 ** i for i in range(n)],
        "TLT": [100 * 1.0 ** i for i in range(n)],
    })
    p1 = performance(DualMomentumStrategy(
        DualMomentumConfig(leverage=1.0, lookbacks=(1, 2, 3))).backtest(df))
    p15 = performance(DualMomentumStrategy(
        DualMomentumConfig(leverage=1.5, lookbacks=(1, 2, 3))).backtest(df))
    assert p15["cagr"] > p1["cagr"]          # leverage boosts return in an uptrend


def test_no_lookahead_positions_are_shifted():
    n = 12
    df = _monthly({s: [100 * 1.01 ** i for i in range(n)] for s in
                   ("SPY", "QQQ", "EFA", "EEM", "GLD", "TLT")})
    pos = DualMomentumStrategy(DualMomentumConfig(lookbacks=(1, 2, 3))).positions(df)
    assert pd.isna(pos.iloc[0])              # first month has no prior decision
