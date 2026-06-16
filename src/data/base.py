"""Data source interface — the swappable boundary for OHLCV retrieval.

Concrete adapters (yfinance, moomoo, synthetic) implement :class:`DataSource`.
Everything downstream depends only on this interface, so the data origin can be
changed via config without touching feature/model/backtest code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# Canonical OHLCV column names used everywhere downstream.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Supported bar intervals -> pandas/yfinance interval string.
INTERVAL_MAP = {"1m": "1m", "5m": "5m", "15m": "15m"}


class DataSource(ABC):
    """Abstract OHLCV provider.

    Implementations must return a DataFrame indexed by a tz-aware
    ``DatetimeIndex`` (exchange timezone) with the columns in
    :data:`OHLCV_COLUMNS`. Returning an empty frame is allowed when no data is
    available; callers handle that gracefully.
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch raw OHLCV bars for ``symbol`` in ``[start, end)``."""
        raise NotImplementedError
