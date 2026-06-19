"""Offline lab to search for a profitable entry edge (train/test split).

Not part of the app — a research harness. Backtests entry-rule variants over the
saved feature matrix, reporting profit factor / Sharpe / win-rate on a TRAIN
slice and a held-out TEST slice so we don't fool ourselves with overfitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import BaseAgent
from src.agents.signal_schema import FAMILY_BEAR, FAMILY_BULL, no_trade_signal
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics
from src.config.settings import load_config
from src.data.store import load_features

CFG = load_config("config/default.yaml")
MATRIX = load_features(CFG)
SPLIT = "2026-05-26"  # ~last 3 weeks held out for validation


def _slice(df, start=None, end=None):
    out = df
    if start:
        out = out[out.index >= pd.Timestamp(start, tz=out.index.tz)]
    if end:
        out = out[out.index < pd.Timestamp(end, tz=out.index.tz)]
    return out


class _Agent(BaseAgent):
    name = "Lab"

    def __init__(self, rule):
        self.rule = rule

    def request_signal(self, context):
        return self.rule(context)


def _block(ctx, sym):
    return (context_symbols(ctx)).get(sym)


def context_symbols(ctx):
    return ctx.get("symbols", {})


def _mk(ctx, family, side, strength, conf):
    sym = (FAMILY_BULL if side == "BULL" else FAMILY_BEAR)[family]
    return {"timestamp": ctx["timestamp"], "agent_name": "Lab", "agent_version": "x",
            "target_family": family, "direction": "UP" if side == "BULL" else "DOWN",
            "action": "BUY_BULL" if side == "BULL" else "BUY_BEAR", "symbol": sym,
            "confidence": round(conf, 4), "reason": "lab", "features_used": {}}


def make_rule(**p):
    """Return an entry rule(context)->signal dict, parameterised."""
    def rule(ctx):
        best = None
        for family, sym in (("NASDAQ", "QQQ"), ("SEMICONDUCTOR", "SMH")):
            b = context_symbols(ctx).get(sym) or {}
            close, vwap = b.get("close"), b.get("vwap")
            rsi = b.get("rsi")
            atr = b.get("atr")
            r = b.get("returns", {})
            r1, r3, r6, r12 = r.get("1_bar"), r.get("3_bar"), r.get("6_bar"), r.get("12_bar")
            if None in (close, vwap, r3) or rsi is None:
                continue
            vwap_dev = (close - vwap) / vwap if vwap else 0.0
            # volatility gate
            if atr is not None and p.get("max_atr") and atr > p["max_atr"]:
                continue
            sig = _eval(p, close, vwap, vwap_dev, rsi, atr, r1, r3, r6, r12)
            if sig is None:
                continue
            side, strength = sig
            if best is None or strength > best[1]:
                best = (side, strength, family)
        if best is None:
            return no_trade_signal(ctx["timestamp"], "Lab", "no_edge").to_dict()
        side, strength, family = best
        conf = min(0.99, 0.66 + 50.0 * strength)
        return _mk(ctx, family, side, strength, conf)
    return rule


def _eval(p, close, vwap, vwap_dev, rsi, atr, r1, r3, r6, r12):
    mode = p.get("mode", "momentum")
    min_str = p.get("min_strength", 0.0)
    min_vwap = p.get("min_vwap_dev", 0.0)
    if mode == "momentum":
        side_f = p.get("side", "both")
        need_r1 = p.get("need_r1", False)
        # trend-following: above VWAP, rising, multi-timeframe agreement, RSI not exhausted
        agree_bull = r3 > 0 and (r12 is None or r12 > 0) and (not p.get("need_r6") or (r6 or 0) > 0) \
            and (not need_r1 or (r1 or 0) > 0)
        agree_bear = r3 < 0 and (r12 is None or r12 < 0) and (not p.get("need_r6") or (r6 or 0) < 0) \
            and (not need_r1 or (r1 or 0) < 0)
        if (side_f in ("both", "bull") and close > vwap and vwap_dev >= min_vwap and agree_bull
                and abs(r3) >= min_str and p.get("rsi_lo_bull", 0) < rsi < p.get("rsi_hi", 100)):
            return "BULL", abs(r3)
        if (side_f in ("both", "bear") and close < vwap and -vwap_dev >= min_vwap and agree_bear
                and abs(r3) >= min_str and p.get("rsi_lo", 0) < rsi < p.get("rsi_hi_bear", 100)):
            return "BEAR", abs(r3)
        return None
    if mode == "meanrev":
        # mean-reversion: oversold below VWAP -> expect bounce (bull); overbought above -> bear
        if close < vwap and rsi <= p.get("rsi_buy", 30) and abs(vwap_dev) >= min_vwap:
            return "BULL", abs(vwap_dev)
        if close > vwap and rsi >= p.get("rsi_sell", 70) and abs(vwap_dev) >= min_vwap:
            return "BEAR", abs(vwap_dev)
        return None
    return None


def run(label, rule, **slices):
    out = {}
    for name, (s, e) in slices.items():
        m = compute_metrics(BacktestEngine(CFG, _Agent(rule)).run(_slice(MATRIX, s, e)), 5)
        out[name] = m
    tr, te = out["train"], out["test"]
    print(f"{label:28} | TRAIN pf {tr['profit_factor']:.2f} sharpe {tr['sharpe_ratio']:6.2f} "
          f"win {tr['win_rate_pct']:4.1f}% n {tr['num_trades']:3d} ret {tr['total_return_pct']:6.2f}% "
          f"|| TEST pf {te['profit_factor']:.2f} sharpe {te['sharpe_ratio']:6.2f} "
          f"win {te['win_rate_pct']:4.1f}% n {te['num_trades']:3d} ret {te['total_return_pct']:6.2f}%")
    return out


SL = {"train": (None, SPLIT), "test": (SPLIT, None)}

if __name__ == "__main__":
    print("data:", MATRIX.index.min(), "->", MATRIX.index.max(), "| split", SPLIT)
    # baseline = current production rule
    run("V0 baseline(mom r3+vwap)", make_rule(mode="momentum"), **SL)
    run("V1 mom+r12 agree", make_rule(mode="momentum"), **SL)  # r12 agreement on by default
    run("V1b mom+r6+r12", make_rule(mode="momentum", need_r6=True), **SL)
    run("V2 mom+rsi<70", make_rule(mode="momentum", rsi_hi=70, rsi_hi_bear=100, rsi_lo=30), **SL)
    run("V3 mom+strength", make_rule(mode="momentum", min_strength=0.001, min_vwap_dev=0.0005), **SL)
    run("V4 mom+all filters", make_rule(mode="momentum", need_r6=True, rsi_hi=72, rsi_lo=28,
                                        min_strength=0.0008, min_vwap_dev=0.0003, max_atr=4.0), **SL)
    run("V5 meanrev rsi30/70", make_rule(mode="meanrev", rsi_buy=30, rsi_sell=70), **SL)
    print("--- round 2: refine around V4 ---")
    base = dict(mode="momentum", need_r6=True, min_strength=0.0008, min_vwap_dev=0.0003, max_atr=4.0)
    run("V4  base", make_rule(**base, rsi_hi=72, rsi_lo=28), **SL)
    run("V7  bull-only", make_rule(**base, side="bull", rsi_hi=72), **SL)
    run("V8  bear-only", make_rule(**base, side="bear", rsi_hi_bear=72), **SL)
    run("V9  +need_r1", make_rule(**base, need_r1=True, rsi_hi=72, rsi_lo=28), **SL)
    run("V10 rsi zone 45-70 bull", make_rule(**base, rsi_lo_bull=45, rsi_hi=70,
                                             rsi_lo=30, rsi_hi_bear=55), **SL)
    run("V11 stronger str+atr3", make_rule(mode="momentum", need_r6=True, min_strength=0.0015,
                                           min_vwap_dev=0.0006, max_atr=3.0, rsi_hi=72, rsi_lo=28), **SL)
    run("V12 bull-only+r1+zone", make_rule(mode="momentum", need_r6=True, need_r1=True, side="bull",
                                           min_strength=0.001, min_vwap_dev=0.0005, max_atr=3.5,
                                           rsi_lo_bull=48, rsi_hi=70), **SL)

    print("--- round 3: V9 stability across splits + micro-variations ---")
    v9 = dict(mode="momentum", need_r6=True, need_r1=True, min_strength=0.0008,
              min_vwap_dev=0.0003, max_atr=4.0, rsi_hi=72, rsi_lo=28)
    for sp in ("2026-05-12", "2026-05-19", "2026-06-02"):
        run(f"V9 @split {sp}", make_rule(**v9),
            train=(None, sp), test=(sp, None))
    run("V9a strength0.0005", make_rule(**{**v9, "min_strength": 0.0005}), **SL)
    run("V9b atr3.5", make_rule(**{**v9, "max_atr": 3.5}), **SL)
    run("V9c rsi 25-75", make_rule(**{**v9, "rsi_hi": 75, "rsi_lo": 25}), **SL)
