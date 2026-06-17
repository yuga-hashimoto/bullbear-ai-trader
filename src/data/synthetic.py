"""Synthetic OHLCV generator.

Deterministic, offline data source used for tests and for demoing the full
pipeline without any network access. Leveraged ETFs are simulated as roughly
3x daily moves of their underlying so that cross-asset relationships exist and
the strategy logic has something real to chase.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .base import DataSource

# Rough daily-leverage factor relative to a chosen underlying.
_LEVERAGE = {
    "TQQQ": ("QQQ", +3.0),
    "SQQQ": ("QQQ", -3.0),
    "SOXL": ("SMH", +3.0),
    "SOXS": ("SMH", -3.0),
}


def _stable_bucket(text: str, modulo: int) -> int:
    """Stable per-symbol bucket; unlike Python hash(), reproducible per process."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


class SyntheticDataSource(DataSource):
    """Generates reproducible 5m-style bars via a seeded random walk."""

    def __init__(self, seed: int = 42, tz: str = "America/New_York") -> None:
        self.seed = seed
        self.tz = tz
        self._underlying_returns: dict[str, pd.Series] = {}

    def _session_index(self, interval: str, start: str, end: str) -> pd.DatetimeIndex:
        step = {"1m": "1min", "5m": "5min", "15m": "15min"}[interval]
        # Build NAIVE timestamps first, then localize once. Localizing per-day
        # tz-aware stamps and concatenating produces mixed UTC offsets across a
        # DST boundary, which pandas refuses to combine.
        days = pd.bdate_range(start=start, end=end)  # naive business days
        stamps: list[pd.Timestamp] = []
        for day in days:
            open_t = day.normalize() + pd.Timedelta(hours=9, minutes=30)
            close_t = day.normalize() + pd.Timedelta(hours=16)
            stamps.extend(pd.date_range(open_t, close_t, freq=step, inclusive="left"))
        idx = pd.DatetimeIndex(stamps)
        return idx.tz_localize(self.tz)

    def _returns_for(self, symbol: str, idx: pd.DatetimeIndex) -> np.ndarray:
        """Per-bar log returns, sharing the underlying path for leveraged ETFs."""
        base, mult = _LEVERAGE.get(symbol, (symbol, 1.0))
        if base not in self._underlying_returns:
            rng = np.random.default_rng(self.seed + _stable_bucket(base, 10_000))
            # mild intraday drift + noise
            r = rng.normal(0.0, 0.0008, size=len(idx))
            self._underlying_returns[base] = pd.Series(r, index=idx)
        underlying = self._underlying_returns[base].reindex(idx).fillna(0.0).to_numpy()
        rng2 = np.random.default_rng(self.seed + _stable_bucket(symbol, 10_000))
        idiosyncratic = rng2.normal(0.0, 0.0003, size=len(idx))
        return underlying * mult + idiosyncratic

    def fetch(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        idx = self._session_index(interval, start, end)
        if len(idx) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rets = self._returns_for(symbol, idx)
        price0 = 50.0 + _stable_bucket(symbol, 50)
        close = price0 * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[price0], close[:-1]])
        rng = np.random.default_rng(self.seed + _stable_bucket(symbol + "hl", 10_000))
        wiggle = np.abs(rng.normal(0.0, 0.0004, size=len(idx))) * close
        high = np.maximum(open_, close) + wiggle
        low = np.minimum(open_, close) - wiggle
        volume = rng.integers(10_000, 100_000, size=len(idx)).astype(float)
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
