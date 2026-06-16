"""Market calendar: open / closed / holiday / early-close classification."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.market.calendar import make_calendar
from src.market.sessions import MarketState

ET = ZoneInfo("America/New_York")


def _dt(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_regular_hours_open():
    cal = make_calendar()
    assert cal.is_market_open(_dt(2024, 3, 13, 10, 0))      # Wed mid-session
    assert cal.classify_state(_dt(2024, 3, 13, 10, 0)) == MarketState.OPEN


def test_premarket_and_afterhours_closed():
    cal = make_calendar()
    assert not cal.is_market_open(_dt(2024, 3, 13, 8, 0))
    assert cal.classify_state(_dt(2024, 3, 13, 8, 0)) == MarketState.PRE_MARKET
    assert cal.classify_state(_dt(2024, 3, 13, 17, 0)) == MarketState.AFTER_HOURS


def test_weekend_closed_not_holiday():
    cal = make_calendar()
    assert cal.classify_state(_dt(2024, 3, 16, 12, 0)) == MarketState.CLOSED  # Saturday
    assert not cal.is_market_open(_dt(2024, 3, 16, 12, 0))


def test_holiday():
    cal = make_calendar()
    assert cal.is_market_holiday(datetime(2024, 1, 1, tzinfo=ET).date())
    assert cal.classify_state(_dt(2024, 1, 1, 12, 0)) == MarketState.HOLIDAY
    assert cal.is_market_holiday(datetime(2024, 7, 4, tzinfo=ET).date())


def test_early_close_day():
    cal = make_calendar()
    # 2024-07-03 is a 1pm early close on NYSE.
    assert cal.is_early_close_day(datetime(2024, 7, 3, tzinfo=ET).date())
    assert cal.classify_state(_dt(2024, 7, 3, 12, 0)) == MarketState.EARLY_CLOSE
    assert cal.classify_state(_dt(2024, 7, 3, 14, 0)) == MarketState.AFTER_HOURS


def test_next_open_and_close_are_aware_and_future():
    cal = make_calendar()
    now = _dt(2024, 3, 16, 12, 0)  # Saturday
    nxt = cal.next_market_open(now)
    assert nxt.tzinfo is not None and nxt > now
    assert cal.next_market_close(now) > nxt
