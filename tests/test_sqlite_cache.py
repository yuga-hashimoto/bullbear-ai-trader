from __future__ import annotations

import pandas as pd

from src.data.sqlite_cache import SQLiteOHLCVCache, symbols_in_cache


def test_sqlite_cache_roundtrip(tmp_path):
    idx = pd.date_range("2026-01-01 14:30", periods=3, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100, 200, 300],
        },
        index=idx,
    )
    cache = SQLiteOHLCVCache(tmp_path / "market.sqlite")

    assert cache.write("SOXL", "5m", df) == 3
    loaded = cache.read("SOXL", "5m")

    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert len(loaded) == 3
    assert float(loaded.iloc[-1]["close"]) == 3.05
    assert symbols_in_cache(cache.status()) == ["SOXL"]
