"""Reproduce the dual-momentum validation (1.5x beats QQQ buy & hold).

Fetches daily prices for the universe via yfinance, resamples to month-end, runs
the strategy at several leverage levels, and prints CAGR / max drawdown / Sharpe
against QQQ and SPY buy & hold over the common period (incl. 2008/2020/2022).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.dual_momentum import (  # noqa: E402
    DualMomentumConfig,
    DualMomentumStrategy,
    performance,
)

UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "GLD", "TLT"]


def load_monthly() -> pd.DataFrame:
    import yfinance as yf

    cols = {}
    for s in UNIVERSE:
        px = yf.Ticker(s).history(period="max", interval="1d")["Close"].dropna()
        cols[s] = px.resample("ME").last()
    df = pd.DataFrame(cols).dropna()
    return df


def _hold(monthly: pd.DataFrame, sym: str) -> dict:
    return performance(monthly[sym].pct_change())


def main() -> None:
    monthly = load_monthly()
    print(f"common period: {monthly.index.min().date()} ~ {monthly.index.max().date()} "
          f"({len(monthly)} months)\n")
    for sym in ("QQQ", "SPY"):
        p = _hold(monthly, sym)
        print(f"  {sym} buy&hold : CAGR {p['cagr']*100:5.1f}%  MaxDD {p['max_drawdown']*100:6.1f}%"
              f"  Sharpe {p['sharpe']:.2f}")
    print()
    for lev in (1.0, 1.25, 1.5, 2.0):
        strat = DualMomentumStrategy(DualMomentumConfig(leverage=lev))
        p = performance(strat.backtest(monthly))
        print(f"  DualMom {lev:>4}x : CAGR {p['cagr']*100:5.1f}%  MaxDD {p['max_drawdown']*100:6.1f}%"
              f"  Sharpe {p['sharpe']:.2f}")

    rec = DualMomentumStrategy(DualMomentumConfig(leverage=1.5)).recommend(monthly)
    print(f"\n  今月の推奨: {rec.asset} を {rec.leverage}x で保有 "
          f"({'リスクオン' if rec.is_risk_on else '債券退避'}, momentum {rec.momentum:+.3f})")
    print("  ランキング:", ", ".join(f"{s} {m:+.2%}" for s, m in rec.ranking))


if __name__ == "__main__":
    main()
