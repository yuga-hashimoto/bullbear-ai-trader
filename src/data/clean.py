"""OHLCV cleaning: timezone, duplicates, missing values, session filtering.

Cleaning is deliberately conservative and *causal* — we never fill a bar using
future information. Forward-fill is allowed for tiny gaps inside a session;
rows still missing essential price data are dropped.
"""
from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger
from ..utils.timeutils import session_mask, to_exchange_tz
from .base import OHLCV_COLUMNS

log = get_logger(__name__)


def clean_ohlcv(
    df: pd.DataFrame,
    tz: str,
    session_open: str,
    session_close: str,
    max_gap_fill: int = 1,
) -> pd.DataFrame:
    """Return a cleaned, regular-session-only OHLCV frame.

    Steps: normalize tz -> sort -> drop duplicate timestamps -> restrict to the
    regular cash session -> small causal gap fill -> drop remaining NaNs.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out.index = to_exchange_tz(pd.DatetimeIndex(out.index), tz)
    out.index.name = "timestamp"
    out = out[[c for c in OHLCV_COLUMNS if c in out.columns]]

    out = out.sort_index()
    # Keep the last observation for duplicate timestamps.
    out = out[~out.index.duplicated(keep="last")]

    # Restrict to the regular cash session (drops pre/post-market & overnight).
    mask = session_mask(out.index, session_open, session_close)
    out = out[mask.to_numpy()]

    # Small, causal forward-fill for isolated gaps; never backfill.
    if max_gap_fill > 0:
        out = out.ffill(limit=max_gap_fill)

    before = len(out)
    out = out.dropna(subset=[c for c in ["open", "high", "low", "close"] if c in out])
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0.0)
    dropped = before - len(out)
    if dropped:
        log.info("clean_ohlcv dropped %d incomplete rows", dropped)

    return out
