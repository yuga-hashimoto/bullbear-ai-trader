"""Live-trading safety gate must be closed by default."""
from __future__ import annotations

import dataclasses

import pytest

from src.config.settings import (
    LiveTradingDisabledError,
    assert_live_trading_allowed,
    load_config,
)
from src.brokers.moomoo import MoomooBroker


def test_default_config_disables_live():
    cfg = load_config("config/default.yaml")
    assert cfg.live_trading_enabled is False


def test_assert_blocks_without_all_three(monkeypatch):
    cfg = load_config("config/default.yaml")
    # Even with env + flag, config flag is false -> blocked.
    monkeypatch.setenv("BULLBEAR_ALLOW_LIVE", "1")
    with pytest.raises(LiveTradingDisabledError):
        assert_live_trading_allowed(cfg, explicit_flag=True)


def test_assert_blocks_when_env_missing(monkeypatch):
    cfg = load_config("config/default.yaml")
    cfg = dataclasses.replace(cfg, live_trading_enabled=True)
    monkeypatch.delenv("BULLBEAR_ALLOW_LIVE", raising=False)
    with pytest.raises(LiveTradingDisabledError):
        assert_live_trading_allowed(cfg, explicit_flag=True)


def test_assert_passes_with_all_three(monkeypatch):
    cfg = load_config("config/default.yaml")
    cfg = dataclasses.replace(cfg, live_trading_enabled=True)
    monkeypatch.setenv("BULLBEAR_ALLOW_LIVE", "1")
    # Should NOT raise when all three independent switches are set.
    assert_live_trading_allowed(cfg, explicit_flag=True)


def test_moomoo_broker_refuses_to_construct_by_default():
    cfg = load_config("config/default.yaml")
    with pytest.raises(LiveTradingDisabledError):
        MoomooBroker(cfg, allow_live=True)  # env + config still block
