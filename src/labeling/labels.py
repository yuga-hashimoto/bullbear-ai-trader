"""Triple-class direction labels (UP / FLAT / DOWN) for decision symbols.

The label at bar ``t`` is derived from the *future* return of the decision
symbol over ``horizon_bars``. Two safeguards keep this honest:

  * The future window is confined to the *same trading day* — we never label a
    bar using a price from the next session (no overnight leak), consistent with
    the day-trading mandate.
  * ``futret`` / ``label`` columns are the only place future data appears. They
    are training targets and are stripped from the model's feature inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.settings import Config

# Integer class encoding.
DOWN, FLAT, UP = 0, 1, 2
CLASS_NAMES = {DOWN: "DOWN", FLAT: "FLAT", UP: "UP"}


def label_col(symbol: str) -> str:
    return f"label__{symbol}"


def futret_col(symbol: str) -> str:
    return f"futret__{symbol}"


def future_return_same_day(close: pd.Series, horizon: int) -> pd.Series:
    """Return over the next ``horizon`` bars, NaN if it would cross days.

    ``close`` must be indexed by a tz-aware DatetimeIndex.
    """
    future_close = close.shift(-horizon)
    day = pd.Series(close.index.date, index=close.index)
    future_day = day.shift(-horizon)
    same_day = day.to_numpy() == future_day.to_numpy()
    fut_ret = future_close / close - 1.0
    return fut_ret.where(pd.Series(same_day, index=close.index))


def make_labels(close: pd.Series, cfg: Config) -> tuple[pd.Series, pd.Series]:
    """Return ``(label_series, future_return_series)`` for one decision symbol."""
    lab_cfg = cfg.labeling
    fut_ret = future_return_same_day(close, lab_cfg.horizon_bars)
    labels = pd.Series(np.nan, index=close.index, dtype="float64")
    labels[fut_ret > lab_cfg.up_threshold] = UP
    labels[fut_ret < lab_cfg.down_threshold] = DOWN
    labels[(fut_ret <= lab_cfg.up_threshold) & (fut_ret >= lab_cfg.down_threshold)] = FLAT
    return labels, fut_ret


def attach_labels(
    matrix: pd.DataFrame, frames: dict[str, pd.DataFrame], cfg: Config
) -> pd.DataFrame:
    """Attach label/future-return columns for each distinct decision symbol."""
    out = matrix.copy()
    decision_symbols = {str(spec["decision"]) for spec in cfg.instruments.values()}
    for sym in decision_symbols:
        if sym not in frames or frames[sym].empty:
            continue
        close = frames[sym]["close"].reindex(out.index)
        labels, fut_ret = make_labels(close, cfg)
        out[label_col(sym)] = labels
        out[futret_col(sym)] = fut_ret
    return out
