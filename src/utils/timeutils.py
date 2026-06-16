"""Timezone and US-market-session helpers.

All intraday data is normalized to a single exchange timezone (default
America/New_York) so that session filtering and "minutes since open / before
close" logic is unambiguous regardless of where the backtest runs.
"""
from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd


def to_exchange_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Return ``index`` localized/converted to the exchange timezone.

    Naive timestamps are assumed to already be in exchange local time.
    """
    zone = ZoneInfo(tz)
    if index.tz is None:
        return index.tz_localize(zone)
    return index.tz_convert(zone)


def parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def session_mask(
    index: pd.DatetimeIndex, open_str: str, close_str: str
) -> pd.Series:
    """Boolean mask selecting bars within the regular cash session.

    A bar timestamped at its *open* is included while ``open <= t < close``.
    """
    open_t = parse_hhmm(open_str)
    close_t = parse_hhmm(close_str)
    times = index.time
    mask = [(t >= open_t) and (t < close_t) for t in times]
    return pd.Series(mask, index=index)


def minutes_since_open(ts: pd.Timestamp, open_str: str) -> float:
    open_t = parse_hhmm(open_str)
    open_dt = ts.normalize() + pd.Timedelta(hours=open_t.hour, minutes=open_t.minute)
    return (ts - open_dt).total_seconds() / 60.0


def minutes_to_close(ts: pd.Timestamp, close_str: str) -> float:
    close_t = parse_hhmm(close_str)
    close_dt = ts.normalize() + pd.Timedelta(
        hours=close_t.hour, minutes=close_t.minute
    )
    return (close_dt - ts).total_seconds() / 60.0
