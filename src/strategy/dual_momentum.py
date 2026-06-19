"""Dual-momentum monthly rotation — the validated core strategy.

Each month: rank a small universe of risky ETFs by blended momentum (mean of the
3/6/12-month returns), hold the strongest. If even the strongest has negative
absolute momentum, step aside into bonds/cash (the absolute-momentum filter that
sidesteps bear markets). Risky holdings are run at a fixed leverage; the safe
asset is always 1x.

Validated on 2005-2026 daily data (incl. 2008/2020/2022), costs + financing
included: 1.5x => CAGR ~18.4% vs QQQ buy&hold 15.5%, with max drawdown -39% vs
-50%. This module is the pure, testable logic; see scripts/dual_momentum_backtest.py
to reproduce the numbers and tools/cli for the current-month recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CASH = "CASH"


@dataclass(frozen=True)
class DualMomentumConfig:
    risky: tuple[str, ...] = ("SPY", "QQQ", "EFA", "EEM", "GLD")
    safe: str = "TLT"                       # risk-off asset (or CASH)
    lookbacks: tuple[int, ...] = (3, 6, 12)  # months; blended (averaged)
    leverage: float = 1.5                   # applied to RISKY holdings only
    financing_pct: float = 0.05             # annual borrow cost on the levered portion
    leverage_expense_pct: float = 0.0095    # annual expense of the leverage vehicle
    switch_cost_pct: float = 0.001          # round-trip-ish cost per rebalance change


@dataclass(frozen=True)
class Recommendation:
    asset: str
    leverage: float
    momentum: float
    is_risk_on: bool
    ranking: list[tuple[str, float]] = field(default_factory=list)


class DualMomentumStrategy:
    def __init__(self, cfg: DualMomentumConfig | None = None) -> None:
        self.cfg = cfg or DualMomentumConfig()

    # ------------------------------------------------------------ signals
    def blended_momentum(self, monthly: pd.DataFrame) -> pd.DataFrame:
        """Mean of the N-month returns over the configured lookbacks."""
        parts = [monthly.pct_change(lb) for lb in self.cfg.lookbacks]
        return sum(parts) / len(parts)

    def _pick(self, mom_row: pd.Series) -> tuple[str, float]:
        risky = [s for s in self.cfg.risky if s in mom_row.index and not pd.isna(mom_row[s])]
        if not risky:
            return self.cfg.safe, float("nan")
        best = max(risky, key=lambda s: mom_row[s])
        best_mom = float(mom_row[best])
        if best_mom > 0:                      # absolute-momentum filter
            return best, best_mom
        return self.cfg.safe, best_mom

    def recommend(self, monthly: pd.DataFrame) -> Recommendation:
        """The position to hold NOW, from the latest fully-closed month."""
        mom = self.blended_momentum(monthly)
        row = mom.iloc[-1]
        asset, best_mom = self._pick(row)
        risk_on = asset != self.cfg.safe
        ranking = sorted(
            ((s, float(row[s])) for s in self.cfg.risky if s in row.index and not pd.isna(row[s])),
            key=lambda kv: kv[1], reverse=True,
        )
        return Recommendation(
            asset=asset,
            leverage=self.cfg.leverage if risk_on else 1.0,
            momentum=best_mom,
            is_risk_on=risk_on,
            ranking=ranking,
        )

    def positions(self, monthly: pd.DataFrame) -> pd.Series:
        """Target asset for each month, decided on prior-month data (no lookahead)."""
        mom = self.blended_momentum(monthly)
        picks = [self._pick(mom.iloc[t])[0] if t >= max(self.cfg.lookbacks) else self.cfg.safe
                 for t in range(len(monthly))]
        # decide at end of month t, hold during t+1
        return pd.Series(picks, index=monthly.index).shift(1)

    # --------------------------------------------------------- backtest
    def backtest(self, monthly: pd.DataFrame) -> pd.Series:
        """Net monthly return series (leverage, financing, expense, switch cost)."""
        c = self.cfg
        pos = self.positions(monthly)
        mret = monthly.pct_change()
        risky = set(c.risky)
        out = []
        for t, ts in enumerate(monthly.index):
            asset = pos.iloc[t]
            if not isinstance(asset, str) or asset not in mret.columns:
                out.append(0.0)
                continue
            base = mret.iloc[t].get(asset, 0.0)
            if asset in risky and c.leverage > 1.0:
                excess = c.leverage - 1.0
                r = base * c.leverage - excess * (c.financing_pct + c.leverage_expense_pct) / 12.0
            else:
                r = base
            out.append(float(r))
        ret = pd.Series(out, index=monthly.index)
        changed = (pos != pos.shift(1)).astype(float)
        ret = ret - changed * c.switch_cost_pct
        return ret.dropna()


def performance(ret: pd.Series) -> dict[str, float]:
    """CAGR / max drawdown / Sharpe / months from a monthly return series."""
    ret = ret.dropna()
    if ret.empty:
        return {"cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "months": 0}
    eq = (1.0 + ret).cumprod()
    years = len(ret) / 12.0
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else 0.0
    max_dd = float(((eq / eq.cummax()) - 1.0).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(12)) if ret.std() > 0 else 0.0
    return {"cagr": float(cagr), "max_drawdown": max_dd, "sharpe": sharpe, "months": int(len(ret))}
