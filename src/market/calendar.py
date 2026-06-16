"""US market calendar (America/New_York), swappable implementation.

Primary: ``pandas_market_calendars`` (XNYS) — handles holidays and early closes
accurately. Fallback: a static NYSE-like calendar derived from pandas holiday
rules, used only if the library is unavailable.

All public methods take/return **tz-aware** datetimes in the exchange timezone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

from ..utils.logging import get_logger
from .sessions import (
    EARLY_CLOSE,
    REGULAR_CLOSE,
    REGULAR_OPEN,
    MarketState,
    TradingSession,
    to_zone,
)

log = get_logger(__name__)
_MAX_SCAN_DAYS = 15


class MarketCalendar(ABC):
    def __init__(self, tz: str = "America/New_York") -> None:
        self.tz = tz
        self.zone = ZoneInfo(tz)

    # -- subclass hooks ------------------------------------------------------
    @abstractmethod
    def is_trading_day(self, d: date) -> bool: ...

    @abstractmethod
    def is_market_holiday(self, d: date) -> bool: ...

    @abstractmethod
    def is_early_close_day(self, d: date) -> bool: ...

    def _close_time(self, d: date) -> time:
        return EARLY_CLOSE if self.is_early_close_day(d) else REGULAR_CLOSE

    # -- derived API ---------------------------------------------------------
    def session_for_date(self, d: date) -> TradingSession | None:
        if not self.is_trading_day(d):
            return None
        open_dt = datetime.combine(d, REGULAR_OPEN, tzinfo=self.zone)
        close_dt = datetime.combine(d, self._close_time(d), tzinfo=self.zone)
        return TradingSession(d, open_dt, close_dt, self.is_early_close_day(d))

    def is_market_open(self, now: datetime) -> bool:
        now = to_zone(now, self.tz)
        sess = self.session_for_date(now.date())
        return bool(sess and sess.contains(now))

    def next_market_open(self, now: datetime) -> datetime:
        now = to_zone(now, self.tz)
        for i in range(_MAX_SCAN_DAYS):
            d = (now + pd.Timedelta(days=i)).date()
            sess = self.session_for_date(d)
            if sess and sess.open_dt > now:
                return sess.open_dt
        raise RuntimeError("no market open found within scan window")

    def next_market_close(self, now: datetime) -> datetime:
        now = to_zone(now, self.tz)
        for i in range(_MAX_SCAN_DAYS):
            d = (now + pd.Timedelta(days=i)).date()
            sess = self.session_for_date(d)
            if sess and sess.close_dt > now:
                return sess.close_dt
        raise RuntimeError("no market close found within scan window")

    def classify_state(self, now: datetime) -> MarketState:
        now = to_zone(now, self.tz)
        d = now.date()
        sess = self.session_for_date(d)
        if sess is None:
            return MarketState.HOLIDAY if self.is_market_holiday(d) else MarketState.CLOSED
        if now < sess.open_dt:
            return MarketState.PRE_MARKET
        if now >= sess.close_dt:
            return MarketState.AFTER_HOURS
        return MarketState.EARLY_CLOSE if sess.is_early_close else MarketState.OPEN


class PandasMarketCalendar(MarketCalendar):
    """XNYS calendar backed by pandas_market_calendars."""

    def __init__(self, name: str = "XNYS", tz: str = "America/New_York") -> None:
        super().__init__(tz)
        import pandas_market_calendars as mcal

        self.name = name
        self._cal = mcal.get_calendar(name)

    @lru_cache(maxsize=8)
    def _year_schedule(self, year: int) -> dict[date, time]:
        sched = self._cal.schedule(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
        out: dict[date, time] = {}
        for ts, row in sched.iterrows():
            close_et = row["market_close"].tz_convert(self.zone)
            out[ts.date()] = close_et.time()
        return out

    def is_trading_day(self, d: date) -> bool:
        return d in self._year_schedule(d.year)

    def is_market_holiday(self, d: date) -> bool:
        # A weekday that is not a trading day == a holiday.
        return d.weekday() < 5 and d not in self._year_schedule(d.year)

    def is_early_close_day(self, d: date) -> bool:
        ct = self._year_schedule(d.year).get(d)
        return ct is not None and ct < REGULAR_CLOSE

    def _close_time(self, d: date) -> time:
        return self._year_schedule(d.year).get(d, REGULAR_CLOSE)


class StaticUSMarketCalendar(MarketCalendar):
    """Fallback NYSE-like calendar from pandas holiday rules (no early closes
    beyond a small known set). Used only if pandas_market_calendars is missing."""

    def __init__(self, tz: str = "America/New_York") -> None:
        super().__init__(tz)
        from pandas.tseries.holiday import (
            AbstractHolidayCalendar,
            GoodFriday,
            Holiday,
            USLaborDay,
            USMartinLutherKingJr,
            USMemorialDay,
            USPresidentsDay,
            USThanksgivingDay,
            nearest_workday,
        )

        class _NYSE(AbstractHolidayCalendar):
            rules = [
                Holiday("NewYears", month=1, day=1, observance=nearest_workday),
                USMartinLutherKingJr,
                USPresidentsDay,
                GoodFriday,
                USMemorialDay,
                Holiday("Juneteenth", month=6, day=19, observance=nearest_workday,
                        start_date="2021-01-01"),
                Holiday("Independence", month=7, day=4, observance=nearest_workday),
                USLaborDay,
                USThanksgivingDay,
                Holiday("Christmas", month=12, day=25, observance=nearest_workday),
            ]

        self._hol_cal = _NYSE()

    @lru_cache(maxsize=8)
    def _holidays(self, year: int) -> set[date]:
        days = self._hol_cal.holidays(start=f"{year}-01-01", end=f"{year}-12-31")
        return {d.date() for d in days}

    @lru_cache(maxsize=8)
    def _early_closes(self, year: int) -> set[date]:
        # Common NYSE early closes: day after Thanksgiving, Christmas Eve, July 3.
        out: set[date] = set()
        tg = [d for d in self._holidays(year)]  # not directly usable; compute below
        # Friday after the 4th Thursday of November.
        nov = pd.date_range(f"{year}-11-01", f"{year}-11-30")
        thursdays = [d for d in nov if d.weekday() == 3]
        if len(thursdays) >= 4:
            out.add((thursdays[3] + pd.Timedelta(days=1)).date())
        for mmdd in ((12, 24), (7, 3)):
            try:
                d = date(year, *mmdd)
                if d.weekday() < 5:
                    out.add(d)
            except ValueError:
                pass
        return out

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self._holidays(d.year)

    def is_market_holiday(self, d: date) -> bool:
        return d.weekday() < 5 and d in self._holidays(d.year)

    def is_early_close_day(self, d: date) -> bool:
        return self.is_trading_day(d) and d in self._early_closes(d.year)


def make_calendar(name: str = "XNYS", tz: str = "America/New_York") -> MarketCalendar:
    """Factory: prefer pandas_market_calendars, fall back to the static calendar."""
    try:
        import pandas_market_calendars  # noqa: F401

        return PandasMarketCalendar(name=name, tz=tz)
    except Exception as exc:  # noqa: BLE001
        log.warning("pandas_market_calendars unavailable (%s); using static calendar", exc)
        return StaticUSMarketCalendar(tz=tz)
