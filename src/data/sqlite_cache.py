"""SQLite OHLCV cache for repeated local research runs.

Parquet/CSV files remain the primary artifact format.  This cache is optional and
is meant to speed up repeated fetch/build cycles and provide a single queryable
local market-data database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from .store import safe_symbol

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (symbol, interval, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval_time
ON ohlcv(symbol, interval, timestamp);
"""


class SQLiteOHLCVCache:
    """Tiny SQLite-backed OHLCV cache."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def write(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"cannot cache OHLCV without columns: {missing}")
        payload = df[required].copy().sort_index()
        payload = payload.reset_index().rename(columns={payload.index.name or "index": "timestamp"})
        if "timestamp" not in payload.columns:
            payload = payload.rename(columns={payload.columns[0]: "timestamp"})
        payload["timestamp"] = pd.to_datetime(payload["timestamp"], utc=True).map(lambda x: x.isoformat())
        payload.insert(0, "interval", interval)
        payload.insert(0, "symbol", symbol)
        rows = list(payload[["symbol", "interval", "timestamp", *required]].itertuples(index=False, name=None))
        with self._connect() as conn:
            conn.execute("DELETE FROM ohlcv WHERE symbol = ? AND interval = ?", (symbol, interval))
            conn.executemany(
                """
                INSERT OR REPLACE INTO ohlcv
                (symbol, interval, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def read(self, symbol: str, interval: str) -> pd.DataFrame:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
            ORDER BY timestamp
        """
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, interval))
        if df.empty:
            raise FileNotFoundError(f"no SQLite OHLCV cache for {symbol} {interval} at {self.db_path}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    def status(self) -> list[dict[str, object]]:
        query = """
            SELECT symbol, interval, COUNT(*) AS rows, MIN(timestamp) AS start, MAX(timestamp) AS end
            FROM ohlcv
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [
            {"symbol": symbol, "safe_symbol": safe_symbol(symbol), "interval": interval, "rows": rows_, "start": start, "end": end}
            for symbol, interval, rows_, start, end in rows
        ]


def write_many(cache: SQLiteOHLCVCache, interval: str, frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Write multiple symbol frames and return row counts."""
    return {symbol: cache.write(symbol, interval, frame) for symbol, frame in frames.items()}


def symbols_in_cache(status_rows: Iterable[dict[str, object]]) -> list[str]:
    return sorted({str(row["symbol"]) for row in status_rows})
