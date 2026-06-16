"""Performance metrics for a backtest result.

Sharpe/Sortino are computed from *daily* returns (annualized with 252 trading
days). Trade-level stats come from the closed-trade list. Benchmarks are simple
buy-and-hold of a tradable ETF over the same window, plus a flat "cash" line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else 0.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min() * 100.0)


def _max_consecutive_losses(trades: list) -> int:
    best = cur = 0
    for t in trades:
        if t.net_pnl < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_metrics(result, interval_minutes: int) -> dict:
    equity = result.equity_curve
    trades = result.trades
    daily = result.daily_pnl

    initial = result.initial_cash
    final = float(equity.iloc[-1]) if not equity.empty else initial
    total_return = _safe_div(final - initial, initial) * 100.0

    # Daily returns for risk-adjusted ratios.
    if not daily.empty:
        daily_ret = daily["return_pct"].to_numpy() / 100.0
        n_days = max(len(daily_ret), 1)
    else:
        daily_ret = np.array([0.0])
        n_days = 1

    ann_return = ((final / initial) ** (TRADING_DAYS / n_days) - 1.0) * 100.0 if initial > 0 else 0.0

    mean_d = float(np.mean(daily_ret))
    std_d = float(np.std(daily_ret, ddof=1)) if len(daily_ret) > 1 else 0.0
    downside = daily_ret[daily_ret < 0]
    dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sharpe = _safe_div(mean_d, std_d) * np.sqrt(TRADING_DAYS)
    sortino = _safe_div(mean_d, dstd) * np.sqrt(TRADING_DAYS)

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)
    avg_win = _safe_div(gross_win, len(wins))
    avg_loss = _safe_div(gross_loss, len(losses))

    total_holding = sum(t.holding_minutes for t in trades)
    total_minutes = max(len(equity) * interval_minutes, 1)

    counters = getattr(result, "counters", {}) or {}
    agent_metrics = {
        "num_signals": counters.get("num_signals", 0),
        "no_trade_ratio": counters.get("no_trade_ratio", 0.0),
        "invalid_signals": counters.get("invalid_signals", 0),
        "rejected_signals": counters.get("rejected_signals", 0),
        "forced_exits": counters.get("forced_exits", 0),
    }

    return {**agent_metrics,
        "total_return_pct": round(total_return, 3),
        "annualized_return_pct": round(ann_return, 3),
        "max_drawdown_pct": round(max_drawdown(equity), 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "win_rate_pct": round(_safe_div(len(wins), len(trades)) * 100.0, 2),
        "profit_factor": round(_safe_div(gross_win, gross_loss), 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(_safe_div(avg_win, avg_loss), 3),
        "num_trades": len(trades),
        "avg_holding_minutes": round(_safe_div(total_holding, len(trades)), 1),
        "worst_day_pct": round(float(daily["return_pct"].min()) if not daily.empty else 0.0, 3),
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "exposure_time_pct": round(_safe_div(total_holding, total_minutes) * 100.0, 2),
        "final_equity": round(final, 2),
        "initial_cash": round(initial, 2),
    }


def buy_and_hold_return(close: pd.Series) -> float:
    """Total return % of holding ``close`` from first to last bar."""
    close = close.dropna()
    if len(close) < 2:
        return 0.0
    return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)


def benchmark_comparison(
    frames: dict[str, pd.DataFrame], benchmark_symbols: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {"cash": 0.0}
    for sym in benchmark_symbols:
        if sym in frames and not frames[sym].empty:
            out[f"buy_hold_{sym}"] = round(buy_and_hold_return(frames[sym]["close"]), 3)
    return out
