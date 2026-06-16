"""Technical indicators — all strictly causal (no look-ahead).

Every function uses only the current bar and prior bars. Rolling/EWM windows
look backward; nothing references a future row. This is the foundation of the
"no future leak" guarantee tested in ``tests/test_no_leak.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, periods: int) -> pd.Series:
    """Simple return over ``periods`` bars (uses past only)."""
    return close.pct_change(periods=periods)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def ma_deviation(close: pd.Series, window: int) -> pd.Series:
    """Percent deviation of price from its moving average."""
    m = sma(close, window)
    return (close - m) / m


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd_diff(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - signal_line


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return atr(high, low, close, window) / close


def volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling std of bar returns (annualization-agnostic, per-bar)."""
    return close.pct_change().rolling(window, min_periods=window).std()


def volume_change(volume: pd.Series, window: int = 20) -> pd.Series:
    avg = volume.rolling(window, min_periods=window).mean()
    return (volume - avg) / avg.replace(0.0, np.nan)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP reset each trading day (causal within the session)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    day = pd.Series(df.index.date, index=df.index)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0.0, np.nan)
    return cum_pv / cum_vol


def vwap_deviation(df: pd.DataFrame) -> pd.Series:
    vwap = session_vwap(df)
    return (df["close"] - vwap) / vwap


def intraday_change(df: pd.DataFrame) -> pd.Series:
    """Return from the session's opening price to the current close."""
    day = pd.Series(df.index.date, index=df.index)
    day_open = df["open"].groupby(day).transform("first")
    return (df["close"] - day_open) / day_open


def overnight_gap(df: pd.DataFrame) -> pd.Series:
    """Gap between today's session open and the prior session's close.

    Causal: uses today's *opening* price (known at the session start) and the
    *previous* day's close (fully known). Today's later prices never enter, so
    the value is stable under truncation. The first day has no prior close ->
    NaN.
    """
    day = pd.Series(df.index.date, index=df.index)
    daily_open = df["open"].groupby(day).first()
    daily_close = df["close"].groupby(day).last()
    prev_close = daily_close.shift(1)
    gap_per_day = (daily_open - prev_close) / prev_close
    return day.map(gap_per_day)


def dist_from_high(close: pd.Series, window: int = 20) -> pd.Series:
    roll_high = close.rolling(window, min_periods=1).max()
    return (close - roll_high) / roll_high


def dist_from_low(close: pd.Series, window: int = 20) -> pd.Series:
    roll_low = close.rolling(window, min_periods=1).min()
    return (close - roll_low) / roll_low
