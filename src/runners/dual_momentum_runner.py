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

    def run(self, monthly: pd.DataFrame | None = None,
            inception_override: str | None = None) -> dict[str, Any]:
        if monthly is None:
            monthly = _fetch_monthly(list(self.cfg.risky) + [self.cfg.safe])
        if monthly.empty:
            raise ValueError("no price data for the dual-momentum universe")

        ret = self.strategy.backtest(monthly)          # full historical sim (reference)
        positions = self.strategy.positions(monthly)
        rec = self.strategy.recommend(monthly)
        latest = ret.index[-1]

        # Inception = when YOU started paper trading (set once, persisted). The
        # paper account starts at `capital` then and only counts FORWARD months —
        # not the 20-year backtest. The backtest is shown separately as reference.
        inception = self._inception(inception_override, latest)
        fwd = ret[ret.index > inception]                # strictly after start
        fwd_pos = positions[positions.index > inception]
        equity = self.capital * float((1.0 + fwd).prod()) if len(fwd) else self.capital
        fwd_perf = performance(fwd)
        bt_perf = performance(ret)

        # forward equity curve, anchored at capital on the inception month
        curve = [{"month": inception.strftime("%Y-%m"), "equity": round(self.capital, 2)}]
        eq = self.capital
        for ts, r in fwd.items():
            eq *= (1.0 + float(r))
            curve.append({"month": ts.strftime("%Y-%m"), "equity": round(eq, 2)})

        history = [{
            "month": ts.strftime("%Y-%m"),
            "held": fwd_pos.get(ts) if isinstance(fwd_pos.get(ts), str) else None,
            "return_pct": round(float(fwd.get(ts, 0.0)) * 100, 2),
        } for ts in fwd.index[-12:]]

        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "inception_month": inception.strftime("%Y-%m"),
            "as_of_month": latest.strftime("%Y-%m"),
            "recommendation": {
                "asset": rec.asset, "leverage": rec.leverage,
                "is_risk_on": rec.is_risk_on, "momentum": round(rec.momentum, 4),
                "ranking": [{"symbol": s, "momentum_pct": round(m * 100, 2)} for s, m in rec.ranking],
            },
            "paper": {                                  # FORWARD live paper (from inception)
                "capital": self.capital, "equity": round(equity, 2),
                "total_return_pct": round((equity / self.capital - 1) * 100, 2),
                "months": len(fwd),
                "cagr_pct": round(fwd_perf["cagr"] * 100, 2) if len(fwd) >= 12 else None,
                "max_drawdown_pct": round(fwd_perf["max_drawdown"] * 100, 2),
                "sharpe": round(fwd_perf["sharpe"], 2) if len(fwd) >= 6 else None,
            },
            "backtest_reference": {                     # 2005-now historical simulation
                "since": ret.index[0].strftime("%Y-%m"),
                "cagr_pct": round(bt_perf["cagr"] * 100, 2),
                "max_drawdown_pct": round(bt_perf["max_drawdown"] * 100, 2),
                "sharpe": round(bt_perf["sharpe"], 2), "months": bt_perf["months"],
            },
            "history": history,
            "equity_curve": curve,
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        tmp.replace(self.state_path)
        log.info("dual-momentum: hold %s @%sx | paper(forward) %.0f over %d mo (incept %s)",
                 rec.asset, rec.leverage, equity, len(fwd), inception.strftime("%Y-%m"))
        return state

    def _inception(self, override, latest):
        if override is not None:
            return pd.Timestamp(override)
        if self.state_path.exists():
            try:
                prev = json.loads(self.state_path.read_text()).get("inception_month")
                if prev:
                    return pd.Timestamp(prev) + pd.offsets.MonthEnd(0)
            except (OSError, json.JSONDecodeError):
                pass
        return latest  # first run: start the paper account now
