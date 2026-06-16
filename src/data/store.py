"""Local OHLCV/feature persistence and the data-source factory.

Files are stored as Parquet when possible (falling back to CSV) under the
configured ``raw_dir`` / ``features_dir``. Symbols are filename-sanitized so
that tickers like ``^VIX`` map to a safe path.
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


def save_raw(cfg: Config, symbol: str, df: pd.DataFrame) -> Path:
    base = _path(cfg.path("raw_dir"), symbol, cfg.interval, "parquet").with_suffix("")
    return _write(df, base)


def load_raw(cfg: Config, symbol: str) -> pd.DataFrame:
    base = _path(cfg.path("raw_dir"), symbol, cfg.interval, "parquet").with_suffix("")
    return _read(base)


def save_features(cfg: Config, df: pd.DataFrame, name: str = "features") -> Path:
    base = cfg.path("features_dir") / f"{name}_{cfg.interval}"
    return _write(df, base)


def load_features(cfg: Config, name: str = "features") -> pd.DataFrame:
    base = cfg.path("features_dir") / f"{name}_{cfg.interval}"
    return _read(base)


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
