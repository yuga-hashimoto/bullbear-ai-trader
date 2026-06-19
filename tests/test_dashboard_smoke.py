"""Dual-momentum dashboard renders from a persisted state file.

Skipped automatically when streamlit is not installed (viewer-only dependency).
"""
from __future__ import annotations

import json

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = "src/reports/dashboard.py"


def _write_state(reports_dir):
    runtime = reports_dir / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "dual_momentum.json").write_text(json.dumps({
        "updated_at": "2026-06-19T15:06:15+00:00",
        "as_of_month": "2026-06",
        "recommendation": {
            "asset": "EEM", "leverage": 1.5, "is_risk_on": True, "momentum": 0.35,
            "ranking": [{"symbol": "EEM", "momentum_pct": 35.0},
                        {"symbol": "QQQ", "momentum_pct": 27.9},
                        {"symbol": "GLD", "momentum_pct": 4.9}],
        },
        "paper": {"capital": 1_000_000.0, "equity": 38_718_976.0,
                  "total_return_pct": 3771.9, "cagr_pct": 18.38,
                  "max_drawdown_pct": -38.73, "sharpe": 0.79, "months": 248},
        "history": [{"month": "2026-05", "held": "EEM", "return_pct": 10.5},
                    {"month": "2026-06", "held": "EEM", "return_pct": 5.3}],
        "equity_curve": [{"month": "2026-05", "equity": 36_800_000.0},
                         {"month": "2026-06", "equity": 38_718_976.0}],
    }))


def test_dashboard_renders_from_state(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    _write_state(reports_dir)
    monkeypatch.setenv("BULLBEAR_REPORTS_DIR", str(reports_dir))
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception, at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "EEM" in rendered                  # current holding surfaced
    assert "+3,771.9%" in rendered or "3771.9" in rendered


def test_dashboard_handles_missing_state(tmp_path, monkeypatch):
    monkeypatch.setenv("BULLBEAR_REPORTS_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception, at.exception      # shows a friendly "waiting" message
