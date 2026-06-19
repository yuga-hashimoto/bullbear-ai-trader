from __future__ import annotations

from src.config.settings import load_config


def test_default_config_matches_jpy_risk_contract():
    cfg = load_config("config/default.yaml")

    assert cfg.account.base_currency == "JPY"
    assert cfg.account.initial_capital_jpy == 1_000_000
    assert cfg.risk.max_loss_per_trade_jpy == 40_000
    assert cfg.risk.max_daily_loss_jpy == 60_000
    assert cfg.risk.max_drawdown_pct == 15.0
    assert cfg.risk.max_portfolio_risk_pct == 5.0
    assert cfg.risk.allow_overnight_positions is True
    assert cfg.runner.vendor_delay_seconds == 600


def test_default_test_window_overlaps_saved_features():
    cfg = load_config("config/default.yaml")

    assert cfg.start_date == "2026-04-20"
    assert cfg.test_start == "2026-06-01"
    assert cfg.test_end == "2026-06-16"


def test_synthetic_config_uses_isolated_paths():
    cfg = load_config("config/synthetic.yaml")

    assert cfg.path("raw_dir").as_posix().startswith("data_synthetic/")
    assert cfg.path("features_dir").as_posix().startswith("data_synthetic/")
    assert cfg.path("signals_dir").as_posix().startswith("data_synthetic/")
    assert cfg.path("artifacts_dir").as_posix().startswith("artifacts_synthetic")
    assert cfg.path("reports_dir").as_posix().startswith("reports_synthetic")
