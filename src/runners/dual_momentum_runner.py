"""Monthly dual-momentum runner: recommendation + paper equity tracking.

Designed to be run once a month (after month-end close). It fetches the universe,
recomputes the strategy's net equity curve up to the latest closed month (paper
track record), and reports what to hold for the coming month. State is persisted
to ``reports/runtime/dual_momentum.json`` for the dashboard / scheduling.

Read-only on the market: it never places real orders.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..strategy.dual_momentum import (
    DualMomentumConfig,
    DualMomentumStrategy,
    performance,
)
from ..utils.logging import get_logger

log = get_logger(__name__)

UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "GLD", "TLT"]
DEFAULT_CAPITAL = 1_000_000.0


def _fetch_monthly(universe: list[str]) -> pd.DataFrame:
    import yfinance as yf

    cols = {}
    for s in universe:
        px = yf.Ticker(s).history(period="max", interval="1d")["Close"].dropna()
        if len(px):
            cols[s] = px.resample("ME").last()
    return pd.DataFrame(cols).dropna()


class DualMomentumRunner:
    def __init__(self, reports_dir: str | Path = "reports",
                 cfg: DualMomentumConfig | None = None,
                 capital: float = DEFAULT_CAPITAL) -> None:
        self.cfg = cfg or DualMomentumConfig()
        self.strategy = DualMomentumStrategy(self.cfg)
        self.capital = capital
        self.state_path = Path(reports_dir) / "runtime" / "dual_momentum.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, monthly: pd.DataFrame | None = None) -> dict[str, Any]:
        if monthly is None:
            monthly = _fetch_monthly(list(self.cfg.risky) + [self.cfg.safe])
        if monthly.empty:
            raise ValueError("no price data for the dual-momentum universe")

        ret = self.strategy.backtest(monthly)
        positions = self.strategy.positions(monthly)
        equity = self.capital * float((1.0 + ret).prod())
        perf = performance(ret)
        rec = self.strategy.recommend(monthly)

        history = []
        for ts in ret.index[-12:]:
            history.append({
                "month": ts.strftime("%Y-%m"),
                "held": positions.get(ts) if isinstance(positions.get(ts), str) else None,
                "return_pct": round(float(ret.get(ts, 0.0)) * 100, 2),
            })

        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "as_of_month": ret.index[-1].strftime("%Y-%m"),
            "recommendation": {
                "asset": rec.asset, "leverage": rec.leverage,
                "is_risk_on": rec.is_risk_on, "momentum": round(rec.momentum, 4),
                "ranking": [{"symbol": s, "momentum_pct": round(m * 100, 2)} for s, m in rec.ranking],
            },
            "paper": {
                "capital": self.capital, "equity": round(equity, 2),
                "total_return_pct": round((equity / self.capital - 1) * 100, 2),
                "cagr_pct": round(perf["cagr"] * 100, 2),
                "max_drawdown_pct": round(perf["max_drawdown"] * 100, 2),
                "sharpe": round(perf["sharpe"], 2), "months": perf["months"],
            },
            "history": history,
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        tmp.replace(self.state_path)
        log.info("dual-momentum: hold %s @%sx, paper equity %.0f (%.1f%% since incept)",
                 rec.asset, rec.leverage, equity, state["paper"]["total_return_pct"])
        return state
