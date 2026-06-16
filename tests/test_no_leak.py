"""Feature engineering must not use future data.

Property: a feature value at bar ``i`` computed from the full history must equal
the value at bar ``i`` computed from the truncated history ``[:i+1]``. If any
indicator peeked ahead, truncation would change the value.
"""
from __future__ import annotations

import numpy as np

from src.features.builder import build_symbol_features, feature_columns


def test_features_are_causal(frames):
    sym = "TQQQ"
    df = frames[sym]
    full = build_symbol_features(df, sym)
    cols = feature_columns(full)
    assert cols, "expected feature columns"

    # Check several interior indices, well past warmup.
    for i in (60, 90, 120):
        if i >= len(df):
            continue
        truncated = build_symbol_features(df.iloc[: i + 1], sym)
        a = full[cols].iloc[i]
        b = truncated[cols].iloc[i]
        for c in cols:
            va, vb = a[c], b[c]
            if np.isnan(va) and np.isnan(vb):
                continue
            assert abs(va - vb) < 1e-9, f"{c} leaked future info at bar {i}"


def test_no_feature_column_is_constant_label_leak(labeled_matrix):
    # Sanity: label/future columns exist but are NOT among feature columns.
    feat = set(feature_columns(labeled_matrix))
    leaked = [c for c in labeled_matrix.columns if c.startswith(("label__", "futret__")) and c in feat]
    assert not leaked, f"label/future columns leaked into features: {leaked}"
