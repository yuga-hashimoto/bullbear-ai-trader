"""yfinance OHLCV adapter.

Note: Yahoo only serves a limited intraday history (e.g. ~60 days for 5m bars).
For longer backtests, switch ``data_source`` to a provider that retains more
history, or persist incremental pulls to the local store.
"""
from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger
from ..utils.timeutils import to_exchange_tz
from .base import INTERVAL_MAP, OHLCV_COLUMNS, DataSource

log = get_logger(__name__)


class YFinanceDataSource(DataSource):
    def __init__(self, tz: str = "America/New_York") -> None:
        self.tz = tz

    def fetch(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "yfinance is not installed. `pip install yfinance` or use "
                "data_source: synthetic."
            ) from exc

        yf_interval = INTERVAL_MAP[interval]
        log.info("yfinance fetch %s %s %s..%s", symbol, yf_interval, start, end)
        raw = yf.download(
            tickers=symbol,
            interval=yf_interval,
            start=start,
            end=end,
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            log.warning("yfinance returned no data for %s", symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # yfinance may return a MultiIndex column frame for a single ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = raw.rename(columns=rename)
        df = df[[c for c in OHLCV_COLUMNS if c in df.columns]].copy()
        df.index = to_exchange_tz(pd.DatetimeIndex(df.index), self.tz)
        df.index.name = "timestamp"
        return df
