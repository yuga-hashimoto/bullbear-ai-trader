"""Bar-boundary scheduling helpers.

The runner sleeps to the *next bar boundary* (not a fixed offset) so that a slow
iteration never causes drift or duplicate processing. Bar boundaries are aligned
to the session-relative grid (09:30 + k*interval).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..market.sessions import to_zone

INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900}


def interval_to_seconds(interval: str) -> int:
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"unsupported interval: {interval}")
    return INTERVAL_SECONDS[interval]


def bar_floor(now: datetime, interval_seconds: int, tz: str) -> datetime:
    """Floor ``now`` to the start of its bar, aligned to the top of the hour.

    5m/15m/1m all divide an hour evenly, so flooring within the hour yields a
    grid that includes 09:30, 09:35, ... for the regular session.
    """
    now = to_zone(now, tz)
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    elapsed = (now - hour_start).total_seconds()
    k = int(elapsed // interval_seconds)
    return hour_start + timedelta(seconds=k * interval_seconds)


def next_bar_boundary(now: datetime, interval_seconds: int, tz: str) -> datetime:
    return bar_floor(now, interval_seconds, tz) + timedelta(seconds=interval_seconds)


def seconds_until(target: datetime, now: datetime, tz: str) -> float:
    return max(0.0, (to_zone(target, tz) - to_zone(now, tz)).total_seconds())
