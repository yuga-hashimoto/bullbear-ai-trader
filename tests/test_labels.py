"""Label generation correctness."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import load_config
from src.labeling.labels import DOWN, FLAT, UP, future_return_same_day, make_labels


def _intraday_index(n: int) -> pd.DatetimeIndex:
    start = pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    return pd.date_range(start, periods=n, freq="5min")


def test_future_return_does_not_cross_day():
    # Two short days back to back.
    idx = pd.DatetimeIndex(
        list(_intraday_index(3))
        + list(pd.date_range(pd.Timestamp("2024-01-03 09:30", tz="America/New_York"),
                             periods=3, freq="5min"))
    )
    close = pd.Series(np.arange(1, 7, dtype=float), index=idx)
    fr = future_return_same_day(close, horizon=2)
    # Last two bars of day 1 cannot see into day 2 -> NaN.
    assert np.isnan(fr.iloc[1])
    assert np.isnan(fr.iloc[2])
    # First bar of day 1 looks 2 bars ahead within the same day.
    assert abs(fr.iloc[0] - (3.0 / 1.0 - 1.0)) < 1e-12


def test_thresholds_map_to_classes():
    cfg = load_config("config/synthetic.yaml")
    idx = _intraday_index(5)
    # Build close so that the 1-bar-ahead return is clearly up / down / flat.
    close = pd.Series([100.0, 100.5, 100.0, 100.0005, 100.0], index=idx)
    # horizon 1 for a clean check.
    object.__setattr__(cfg.labeling, "horizon_bars", 1)
    labels, fr = make_labels(close, cfg)
    assert labels.iloc[0] == UP      # +0.5% > up_threshold
    assert labels.iloc[1] == DOWN    # -0.5% < down_threshold
    assert labels.iloc[3] == FLAT    # ~0 within band
