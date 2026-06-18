from __future__ import annotations

import dataclasses
import json

from src.ops.doctor import build_readiness


def test_readiness_reports_live_disabled_and_missing_data(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    paths = {
        **cfg.paths,
        "features_dir": str(tmp_path / "features"),
        "reports_dir": str(tmp_path / "reports"),
    }
    cfg2 = dataclasses.replace(cfg, paths=paths)

    result = build_readiness(cfg2)

    assert result["live_trading"]["enabled"] is False
    assert result["paper"]["ready"] is False
    codes = {item["code"] for item in result["checks"] if not item["ok"]}
    assert "features_missing" in codes


def test_readiness_is_json_serializable(cfg, tmp_path):
    paths = {**cfg.paths, "reports_dir": str(tmp_path / "reports")}
    result = build_readiness(dataclasses.replace(cfg, paths=paths))

    json.dumps(result)
