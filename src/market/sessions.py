"""Market session primitives.

All datetimes here are **timezone-aware**. Naive datetimes are rejected — the
whole system reasons in the exchange timezone (America/New_York) so that DST and
day-of-week logic is unambiguous regardless of where the process runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

import pandas as pd


class MarketState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    HOLIDAY = "holiday"
    EARLY_CLOSE = "early_close"   # an early-close day, currently within session
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"


REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


def require_aware(dt: datetime) -> datetime:
    """Guard: reject naive datetimes (no tzinfo)."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("naive datetime is not allowed; pass a tz-aware datetime")
    return dt


def to_zone(dt: datetime, tz: str) -> datetime:
    return require_aware(dt).astimezone(ZoneInfo(tz))


@dataclass(frozen=True)
class TradingSession:
    """A single trading day's regular-session window (tz-aware)."""

    session_date: date
    open_dt: datetime
    close_dt: datetime
    is_early_close: bool = False

    def contains(self, now: datetime) -> bool:
        return self.open_dt <= require_aware(now) < self.close_dt

    def minutes_since_open(self, now: datetime) -> float:
        return (require_aware(now) - self.open_dt).total_seconds() / 60.0

    def minutes_to_close(self, now: datetime) -> float:
        return (self.close_dt - require_aware(now)).total_seconds() / 60.0
