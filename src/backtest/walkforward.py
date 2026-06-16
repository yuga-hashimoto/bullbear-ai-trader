"""Walk-forward validation windows.

Generates rolling (train, test) date windows so the model is always evaluated
on data *after* its training period. This avoids the optimistic bias of a single
fixed split and is the recommended way to validate a time-series strategy.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_windows(
    start: str,
    end: str,
    train_days: int,
    test_days: int,
    step_days: int,
    tz: str = "America/New_York",
) -> list[Window]:
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz)
    windows: list[Window] = []
    cursor = start_ts
    while True:
        tr_start = cursor
        tr_end = tr_start + pd.Timedelta(days=train_days)
        te_start = tr_end
        te_end = te_start + pd.Timedelta(days=test_days)
        if te_start >= end_ts:
            break
        windows.append(
            Window(tr_start, tr_end, te_start, min(te_end, end_ts))
        )
        cursor = cursor + pd.Timedelta(days=step_days)
    return windows
