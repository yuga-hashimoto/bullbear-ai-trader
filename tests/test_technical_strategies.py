from __future__ import annotations

import pandas as pd

from src.research.technical_strategies import generate_signal_frame, list_strategy_specs, strategy_names


def _ohlcv(rows: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 09:30", periods=rows, freq="5min", tz="America/New_York")
    close = pd.Series([100 + i * 0.2 for i in range(rows)], index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000,
        },
        index=idx,
    )


def test_strategy_registry_contains_expected_baselines():
    names = {spec.name for spec in list_strategy_specs()}
    assert {"sma_cross", "macd", "rsi_reversion", "bollinger_breakout", "turtle_breakout"} <= names
    assert strategy_names() == sorted(names)


def test_generate_signal_frame_has_expected_columns_and_no_future_shape_change():
    df = _ohlcv()
    signals = generate_signal_frame(df, "sma_cross", {"short": 5, "long": 15})

    assert list(signals.index) == list(df.index)
    assert {"score", "direction", "confidence", "reason", "strategy"} <= set(signals.columns)
    assert set(signals["direction"].unique()) <= {"UP", "DOWN", "FLAT"}
    assert signals["confidence"].between(0.0, 0.98).all()
