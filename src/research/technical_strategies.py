"""Deterministic technical strategy library for research and challenger seeding.

The implementations in this module are intentionally small, dependency-light and
causal.  They are not copied from any external project; they capture the useful
patterns from common Backtrader-style research libraries in this repository's own
Signal/Risk architecture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """Metadata for a rule strategy exposed through CLI/MCP."""

    name: str
    family: str
    description: str
    default_params: dict[str, float | int]

    def to_dict(self) -> dict:
        return asdict(self)


_REQUIRED = {"open", "high", "low", "close", "volume"}


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {missing}")
    out = df[["open", "high", "low", "close", "volume"]].copy()
    return out.sort_index()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return (atr / df["close"]).fillna(0.0)


def _confidence(score: pd.Series, *, floor: float = 0.55, scale: float = 20.0) -> pd.Series:
    conf = floor + score.abs().fillna(0.0).mul(scale)
    return conf.clip(lower=0.0, upper=0.98)


def _frame(score: pd.Series, *, reason: str, min_confidence: float = 0.56, scale: float = 20.0) -> pd.DataFrame:
    conf = _confidence(score, floor=min_confidence, scale=scale)
    direction = pd.Series("FLAT", index=score.index, dtype="object")
    direction = direction.mask(score > 0, "UP")
    direction = direction.mask(score < 0, "DOWN")
    direction = direction.mask(conf < min_confidence, "FLAT")
    return pd.DataFrame(
        {
            "score": score.fillna(0.0),
            "direction": direction,
            "confidence": conf.where(direction != "FLAT", 0.0),
            "reason": reason,
        }
    )


def _buy_hold(df: pd.DataFrame, **_: float | int) -> pd.DataFrame:
    score = pd.Series(0.01, index=df.index)
    return _frame(score, reason="buy_hold_baseline", min_confidence=0.66, scale=0.0)


def _sma_cross(df: pd.DataFrame, short: int = 10, long: int = 40, **_: float | int) -> pd.DataFrame:
    close = df["close"]
    short_ma = close.rolling(int(short), min_periods=int(short)).mean()
    long_ma = close.rolling(int(long), min_periods=int(long)).mean()
    score = (short_ma - long_ma) / close
    return _frame(score, reason=f"sma_cross short={short} long={long}", scale=35.0)


def _ema_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    **_: float | int,
) -> pd.DataFrame:
    close = df["close"]
    macd = close.ewm(span=int(fast), adjust=False).mean() - close.ewm(span=int(slow), adjust=False).mean()
    sig = macd.ewm(span=int(signal), adjust=False).mean()
    score = (macd - sig) / close
    return _frame(score, reason=f"macd fast={fast} slow={slow} signal={signal}", scale=90.0)


def _rsi_reversion(
    df: pd.DataFrame,
    period: int = 14,
    lower: int = 30,
    upper: int = 70,
    **_: float | int,
) -> pd.DataFrame:
    rsi = _rsi(df["close"], int(period))
    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(rsi < float(lower), (float(lower) - rsi) / 100.0)
    raw = raw.mask(rsi > float(upper), -((rsi - float(upper)) / 100.0))
    return _frame(raw, reason=f"rsi_reversion period={period} lower={lower} upper={upper}", scale=9.0)


def _rsi_momentum(
    df: pd.DataFrame,
    period: int = 14,
    lower: int = 42,
    upper: int = 58,
    **_: float | int,
) -> pd.DataFrame:
    rsi = _rsi(df["close"], int(period))
    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(rsi > float(upper), (rsi - float(upper)) / 100.0)
    raw = raw.mask(rsi < float(lower), -((float(lower) - rsi) / 100.0))
    return _frame(raw, reason=f"rsi_momentum period={period} lower={lower} upper={upper}", scale=8.0)


def _bollinger_reversion(
    df: pd.DataFrame,
    period: int = 20,
    stdev: float = 2.0,
    **_: float | int,
) -> pd.DataFrame:
    close = df["close"]
    mid = close.rolling(int(period), min_periods=int(period)).mean()
    band = close.rolling(int(period), min_periods=int(period)).std() * float(stdev)
    upper = mid + band
    lower = mid - band
    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(close < lower, (lower - close) / close)
    raw = raw.mask(close > upper, -((close - upper) / close))
    return _frame(raw, reason=f"bollinger_reversion period={period} stdev={stdev}", scale=60.0)


def _bollinger_breakout(
    df: pd.DataFrame,
    period: int = 20,
    stdev: float = 2.0,
    **_: float | int,
) -> pd.DataFrame:
    close = df["close"]
    mid = close.rolling(int(period), min_periods=int(period)).mean()
    band = close.rolling(int(period), min_periods=int(period)).std() * float(stdev)
    upper = mid + band
    lower = mid - band
    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(close > upper, (close - upper) / close)
    raw = raw.mask(close < lower, -((lower - close) / close))
    return _frame(raw, reason=f"bollinger_breakout period={period} stdev={stdev}", scale=75.0)


def _momentum(df: pd.DataFrame, lookback: int = 12, threshold: float = 0.001, **_: float | int) -> pd.DataFrame:
    ret = df["close"].pct_change(int(lookback))
    score = ret.where(ret.abs() >= float(threshold), 0.0)
    return _frame(score, reason=f"momentum lookback={lookback} threshold={threshold}", scale=18.0)


def _turtle_breakout(df: pd.DataFrame, entry: int = 20, exit: int = 10, **_: float | int) -> pd.DataFrame:
    close = df["close"]
    high = df["high"].rolling(int(entry), min_periods=int(entry)).max().shift(1)
    low = df["low"].rolling(int(entry), min_periods=int(entry)).min().shift(1)
    atr = _atr_pct(df, int(exit)).replace(0.0, pd.NA)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(close > high, ((close - high) / close) / atr)
    raw = raw.mask(close < low, -(((low - close) / close) / atr))
    return _frame(raw.fillna(0.0), reason=f"turtle_breakout entry={entry} exit={exit}", scale=0.7)


def _vcp_breakout(
    df: pd.DataFrame,
    lookback: int = 30,
    contraction: int = 10,
    volume_ratio: float = 1.05,
    **_: float | int,
) -> pd.DataFrame:
    close = df["close"]
    vol = close.pct_change().rolling(int(contraction), min_periods=int(contraction)).std()
    prior_vol = close.pct_change().rolling(int(lookback), min_periods=int(lookback)).std()
    high = close.rolling(int(lookback), min_periods=int(lookback)).max().shift(1)
    avg_volume = df["volume"].rolling(int(lookback), min_periods=int(lookback)).mean()
    contracted = vol < prior_vol
    volume_confirmed = df["volume"] > avg_volume * float(volume_ratio)
    raw = pd.Series(0.0, index=df.index)
    breakout = (close > high) & contracted & volume_confirmed
    raw = raw.mask(breakout, (close - high) / close)
    return _frame(raw, reason=f"vcp_breakout lookback={lookback} contraction={contraction}", scale=85.0)


StrategyFunc = Callable[..., pd.DataFrame]

_SPECS: dict[str, StrategySpec] = {
    "buy_hold": StrategySpec("buy_hold", "baseline", "Always proposes the bull ETF for the selected family.", {}),
    "sma_cross": StrategySpec("sma_cross", "trend", "Short/long moving-average cross on the decision asset.", {"short": 10, "long": 40}),
    "macd": StrategySpec("macd", "trend", "EMA MACD histogram direction on the decision asset.", {"fast": 12, "slow": 26, "signal": 9}),
    "rsi_reversion": StrategySpec("rsi_reversion", "mean_reversion", "RSI oversold/overbought mean-reversion.", {"period": 14, "lower": 30, "upper": 70}),
    "rsi_momentum": StrategySpec("rsi_momentum", "momentum", "RSI regime-following momentum.", {"period": 14, "lower": 42, "upper": 58}),
    "bollinger_reversion": StrategySpec("bollinger_reversion", "mean_reversion", "Fade closes outside Bollinger bands.", {"period": 20, "stdev": 2.0}),
    "bollinger_breakout": StrategySpec("bollinger_breakout", "breakout", "Follow closes outside Bollinger bands.", {"period": 20, "stdev": 2.0}),
    "momentum": StrategySpec("momentum", "momentum", "N-bar return direction with a deadband threshold.", {"lookback": 12, "threshold": 0.001}),
    "turtle_breakout": StrategySpec("turtle_breakout", "breakout", "Donchian-channel breakout normalized by ATR%.", {"entry": 20, "exit": 10}),
    "vcp_breakout": StrategySpec("vcp_breakout", "breakout", "Volatility-contraction breakout with volume confirmation.", {"lookback": 30, "contraction": 10, "volume_ratio": 1.05}),
}

_FUNCTIONS: dict[str, StrategyFunc] = {
    "buy_hold": _buy_hold,
    "sma_cross": _sma_cross,
    "macd": _ema_macd,
    "rsi_reversion": _rsi_reversion,
    "rsi_momentum": _rsi_momentum,
    "bollinger_reversion": _bollinger_reversion,
    "bollinger_breakout": _bollinger_breakout,
    "momentum": _momentum,
    "turtle_breakout": _turtle_breakout,
    "vcp_breakout": _vcp_breakout,
}


def list_strategy_specs() -> list[StrategySpec]:
    """Return all built-in rule strategy definitions."""
    return list(_SPECS.values())


def strategy_names() -> list[str]:
    return sorted(_SPECS)


def generate_signal_frame(
    ohlcv: pd.DataFrame,
    strategy_name: str,
    params: Mapping[str, float | int] | None = None,
) -> pd.DataFrame:
    """Generate per-bar directional research signals for one decision asset.

    Returns a DataFrame indexed like ``ohlcv`` with: ``score``, ``direction``
    (UP/DOWN/FLAT), ``confidence`` and ``reason``.  The caller is responsible for
    converting the direction into this repo's Signal JSON / Risk Engine flow.
    """
    name = strategy_name.lower()
    if name not in _FUNCTIONS:
        raise ValueError(f"unknown strategy: {strategy_name!r}; available={strategy_names()}")
    df = _validate_ohlcv(ohlcv)
    merged_params = dict(_SPECS[name].default_params)
    merged_params.update(dict(params or {}))
    out = _FUNCTIONS[name](df, **merged_params)
    out["strategy"] = name
    return out
