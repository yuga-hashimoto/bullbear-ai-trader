"""Market data feeds for the live-style runners.

A feed returns the most recent bars (per symbol) observable *as of* ``now`` —
never future bars. The runner derives ``last_bar_time`` from the returned data
to detect staleness. Feeds are swappable like :class:`DataSource`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from ..data.clean import clean_ohlcv
from ..data.synthetic import SyntheticDataSource
from ..data.yfinance_source import YFinanceDataSource
from ..config.settings import Config
from ..market.sessions import to_zone
from ..runners.scheduler import bar_floor, interval_to_seconds


class MarketDataFeed(ABC):
    @abstractmethod
    def fetch_recent(
        self, symbols: list[str], interval: str, now: datetime
    ) -> dict[str, pd.DataFrame]:
        """Return cleaned OHLCV per symbol, only bars with index <= current bar."""
        raise NotImplementedError


class SyntheticLiveFeed(MarketDataFeed):
    """Deterministic synthetic feed (offline). Bars are generated up to the
    current bar boundary so the runner can operate with no network."""

    def __init__(self, tz: str = "America/New_York", seed: int = 42,
                 lookback_days: int = 7) -> None:
        self.tz = tz
        self.lookback_days = lookback_days
        self._src = SyntheticDataSource(seed=seed, tz=tz)

    def fetch_recent(self, symbols, interval, now):
        now = to_zone(now, self.tz)
        cutoff = bar_floor(now, interval_to_seconds(interval), self.tz)
        start = (now - pd.Timedelta(days=self.lookback_days)).date().isoformat()
        end = (now + pd.Timedelta(days=1)).date().isoformat()
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self._src.fetch(sym, interval, start, end)
            df = df[df.index <= cutoff]
            out[sym] = clean_ohlcv(df, self.tz, "09:30", "16:00")
        return out


class YFinanceLiveFeed(MarketDataFeed):
    """Recent real market bars from Yahoo, excluding the still-forming bar."""

    def __init__(self, tz: str = "America/New_York", lookback_days: int = 7) -> None:
        self.tz = tz
        self.lookback_days = lookback_days
        self._src = YFinanceDataSource(tz=tz)

    def fetch_recent(self, symbols, interval, now):
        now = to_zone(now, self.tz)
        interval_seconds = interval_to_seconds(interval)
        cutoff = bar_floor(now, interval_seconds, self.tz) - pd.Timedelta(
            seconds=interval_seconds
        )
        start = (now - pd.Timedelta(days=self.lookback_days)).date().isoformat()
        end = (now + pd.Timedelta(days=1)).date().isoformat()
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self._src.fetch(symbol, interval, start, end)
            frame = frame[frame.index <= cutoff]
            out[symbol] = clean_ohlcv(frame, self.tz, "09:30", "16:00")
        return out


def make_live_feed(cfg: Config) -> MarketDataFeed:
    if cfg.data_source == "synthetic":
        return SyntheticLiveFeed(tz=cfg.runner.timezone, seed=cfg.backtest.random_seed)
    if cfg.data_source == "yfinance":
        return YFinanceLiveFeed(tz=cfg.runner.timezone)
    raise NotImplementedError(
        f"runner feed for data_source={cfg.data_source!r} is not implemented"
    )


class FrozenFramesFeed(MarketDataFeed):
    """Serves pre-supplied frames sliced to ``now`` (for tests / replay)."""

    def __init__(self, frames: dict[str, pd.DataFrame], tz: str = "America/New_York") -> None:
        self.frames = frames
        self.tz = tz

    def fetch_recent(self, symbols, interval, now):
        now = to_zone(now, self.tz)
        cutoff = bar_floor(now, interval_to_seconds(interval), self.tz)
        return {
            sym: self.frames[sym][self.frames[sym].index <= cutoff]
            for sym in symbols if sym in self.frames
        }
