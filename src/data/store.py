"""Local OHLCV/feature persistence and the data-source factory.

Files are stored as Parquet when possible (falling back to CSV) under the
configured ``raw_dir`` / ``features_dir``. Symbols are filename-sanitized so
that tickers like ``^VIX`` map to a safe path.

An optional SQLite OHLCV cache can also be enabled with:

    storage:
      sqlite_enabled: true
      sqlite_db: data/cache/market_data.sqlite

Parquet/CSV remains the canonical artifact; SQLite is a query/cache layer.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config.settings import Config
from ..utils.logging import get_logger
from .base import DataSource

log = get_logger(__name__)


def safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "_idx_").replace("/", "_").replace(":", "_")


def _path(directory: Path, symbol: str, interval: str, ext: str) -> Path:
    return directory / f"{safe_symbol(symbol)}_{interval}.{ext}"


def _write(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        df.to_parquet(path)
        return path
    except Exception:  # pragma: no cover - parquet engine missing
        path = base.with_suffix(".csv")
        df.to_csv(path)
        return path


def _read(base: Path) -> pd.DataFrame:
    pq = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        df = pd.read_csv(csv, index_col=0)
        # Parse via UTC to avoid mixed-offset object indexes across DST, then
        # let downstream cleaning convert to the exchange timezone.
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    raise FileNotFoundError(f"no stored data at {pq} or {csv}")


def _sqlite_enabled(cfg: Config) -> bool:
    return bool((cfg.raw.get("storage", {}) or {}).get("sqlite_enabled", False))


def _sqlite_db_path(cfg: Config) -> str:
    storage = dict(cfg.raw.get("storage", {}) or {})
    return str(storage.get("sqlite_db", "data/cache/market_data.sqlite"))


def save_raw(cfg: Config, symbol: str, df: pd.DataFrame) -> Path:
    base = _path(cfg.path("raw_dir"), symbol, cfg.interval, "parquet").with_suffix("")
    path = _write(df, base)
    if _sqlite_enabled(cfg):
        try:
            from .sqlite_cache import SQLiteOHLCVCache

            rows = SQLiteOHLCVCache(_sqlite_db_path(cfg)).write(symbol, cfg.interval, df)
            log.info("cached %s rows for %s %s in SQLite", rows, symbol, cfg.interval)
        except Exception as exc:  # cache must not break canonical file persistence
            log.warning("SQLite cache write failed for %s: %s", symbol, exc)
    return path


def load_raw(cfg: Config, symbol: str) -> pd.DataFrame:
    base = _path(cfg.path("raw_dir"), symbol, cfg.interval, "parquet").with_suffix("")
    try:
        return _read(base)
    except FileNotFoundError:
        if not _sqlite_enabled(cfg):
            raise
        from .sqlite_cache import SQLiteOHLCVCache

        return SQLiteOHLCVCache(_sqlite_db_path(cfg)).read(symbol, cfg.interval)


def save_features(cfg: Config, df: pd.DataFrame, name: str = "features") -> Path:
    base = cfg.path("features_dir") / f"{name}_{cfg.interval}"
    return _write(df, base)


def load_features(cfg: Config, name: str = "features") -> pd.DataFrame:
    base = cfg.path("features_dir") / f"{name}_{cfg.interval}"
    return _read(base)


def sqlite_cache_status(cfg: Config) -> list[dict[str, object]]:
    from .sqlite_cache import SQLiteOHLCVCache

    return SQLiteOHLCVCache(_sqlite_db_path(cfg)).status()


def make_data_source(cfg: Config) -> DataSource:
    """Factory selecting the configured data source (the swap point)."""
    if cfg.data_source == "synthetic":
        from .synthetic import SyntheticDataSource

        return SyntheticDataSource(seed=cfg.backtest.random_seed, tz=cfg.timezone)
    if cfg.data_source == "yfinance":
        from .yfinance_source import YFinanceDataSource

        return YFinanceDataSource(tz=cfg.timezone)
    if cfg.data_source == "moomoo":
        from .moomoo_source import MoomooDataSource

        return MoomooDataSource(tz=cfg.timezone)
    raise ValueError(f"unknown data_source: {cfg.data_source}")
