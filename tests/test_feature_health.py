from __future__ import annotations

import dataclasses
import json

import pandas as pd

from src.features.builder import (
    build_symbol_features,
    feature_columns,
    prepare_feature_matrix,
)
from src.pipeline import build_features


def test_zero_volume_symbol_skips_volume_dependent_features(frames):
    frame = frames["QQQ"].copy()
    frame["volume"] = 0.0

    features = build_symbol_features(frame, "^VIX")

    assert "feat__^VIX__vwap_dev" not in features
    assert "feat__^VIX__volchg_20" not in features


def test_prepare_feature_matrix_drops_only_all_nan_features(labeled_matrix):
    matrix = labeled_matrix.copy()
    matrix["feat__^VIX__vwap_dev"] = float("nan")

    prepared, health = prepare_feature_matrix(matrix)

    assert not prepared.empty
    assert "feat__^VIX__vwap_dev" not in prepared.columns
    assert "feat__^VIX__vwap_dev" in health["dropped_all_nan_features"]
    assert all(prepared[feature_columns(prepared)].notna().all())


def test_build_features_writes_health_report(cfg, frames, tmp_path, monkeypatch):
    paths = {
        **cfg.paths,
        "raw_dir": str(tmp_path / "raw"),
        "features_dir": str(tmp_path / "features"),
    }
    cfg2 = dataclasses.replace(cfg, paths=paths)
    monkeypatch.setattr("src.pipeline._load_frames", lambda _cfg, _symbols: frames)

    build_features(cfg2)

    report_path = cfg2.path("features_dir") / "feature_health_report.json"
    report = json.loads(report_path.read_text())
    assert report["rows"] > 0
    assert report["feature_count_before"] >= report["feature_count_after"]
    assert isinstance(report["nan_ratio_by_feature"], dict)
