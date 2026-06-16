"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.base import BaseAgent
from src.agents.signal_schema import no_trade_signal
from src.config.settings import load_config
from src.data.clean import clean_ohlcv
from src.data.synthetic import SyntheticDataSource
from src.features.builder import build_feature_matrix
from src.labeling.labels import attach_labels
from src.models.base import DirectionModel


@pytest.fixture(scope="session")
def cfg():
    return load_config("config/synthetic.yaml")


@pytest.fixture(scope="session")
def frames(cfg):
    src = SyntheticDataSource(seed=7, tz=cfg.timezone)
    out = {}
    for sym in cfg.all_symbols:
        raw = src.fetch(sym, cfg.interval, "2024-01-02", "2024-01-12")
        out[sym] = clean_ohlcv(raw, cfg.timezone, cfg.session_open, cfg.session_close)
    return out


@pytest.fixture(scope="session")
def labeled_matrix(cfg, frames):
    m = build_feature_matrix(frames, cfg)
    return attach_labels(m, frames, cfg)


class ConstantModel(DirectionModel):
    """Returns a fixed class distribution for every row (test double)."""

    def __init__(self, proba=(0.0, 0.0, 1.0)):
        # order matches classes_ = [DOWN, FLAT, UP]
        self._proba = np.array(proba, dtype=float)
        self.classes_ = [0, 1, 2]

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile(self._proba, (len(X), 1))

    def save(self, path):  # pragma: no cover - not used
        pass

    @classmethod
    def load(cls, path):  # pragma: no cover - not used
        return cls()


class ConstantSignalAgent(BaseAgent):
    """Test agent that emits the same signal every bar."""

    name = "ConstantSignalAgent"

    def __init__(self, action="NO_TRADE", symbol=None, family="MARKET",
                 direction="FLAT", confidence=0.0):
        self._action = action
        self._symbol = symbol
        self._family = family
        self._direction = direction
        self._confidence = confidence

    def request_signal(self, context):
        ts = context["timestamp"]
        if self._action == "NO_TRADE":
            return no_trade_signal(ts, self.name).to_dict()
        return {
            "timestamp": ts,
            "agent_name": self.name,
            "target_family": self._family,
            "direction": self._direction,
            "action": self._action,
            "symbol": self._symbol,
            "confidence": self._confidence,
        }
