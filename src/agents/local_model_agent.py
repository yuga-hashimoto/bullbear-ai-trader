"""LocalModelAgent — wraps the in-repo ML models as an Agent.

The LightGBM / sklearn direction models are NO LONGER on the main decision path;
they are simply one Agent implementation among others. This keeps the trained
models usable (and benchmarkable against external agents) without making the
repo responsible for the trading intelligence.

It precomputes per-bar predictions in :meth:`prepare` and emits a standard
Signal per bar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config.settings import Config
from ..models.base import Prediction
from ..models.factory import load_model
from ..utils.logging import get_logger
from .base import BaseAgent
from .signal_schema import FAMILY_BEAR, FAMILY_BULL, no_trade_signal

log = get_logger(__name__)

# Decision symbol -> agent target family.
DECISION_FAMILY = {"QQQ": "NASDAQ", "SMH": "SEMICONDUCTOR", "SPY": "MARKET"}


class LocalModelAgent(BaseAgent):
    name = "LocalModelAgent"
    version = "1.0.0"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.models = self._load_models(cfg)
        self._preds: dict[str, dict[str, Prediction]] = {}  # ts -> {sym: pred}

    @staticmethod
    def _decision_symbols(cfg: Config) -> list[str]:
        seen: dict[str, None] = {}
        for spec in cfg.instruments.values():
            seen.setdefault(str(spec["decision"]), None)
        return list(seen.keys())

    def _load_models(self, cfg: Config) -> dict:
        models = {}
        for sym in self._decision_symbols(cfg):
            path = cfg.path("artifacts_dir") / f"model_{sym.replace('^', '_idx_')}.pkl"
            if Path(path).exists():
                models[sym] = load_model(cfg, str(path))
        if not models:
            raise RuntimeError(
                "LocalModelAgent found no trained models. Run `train` first."
            )
        return models

    def prepare(self, matrix: pd.DataFrame, feat_cols: list[str]) -> None:
        X = matrix[feat_cols]
        per_symbol = {sym: model.predict(X) for sym, model in self.models.items()}
        ts_index = [t.isoformat() for t in matrix.index]
        self._preds = {}
        for i, ts in enumerate(ts_index):
            self._preds[ts] = {sym: preds[i] for sym, preds in per_symbol.items()}

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        ts = context["timestamp"]
        preds = self._preds.get(ts, {})
        best_sym = None
        best_pred: Prediction | None = None
        for sym, pred in preds.items():
            if pred.direction == "FLAT":
                continue
            if best_pred is None or pred.confidence > best_pred.confidence:
                best_sym, best_pred = sym, pred

        if best_pred is None or best_sym not in DECISION_FAMILY:
            return no_trade_signal(ts, self.name, reason="model FLAT/low").to_dict()

        family = DECISION_FAMILY[best_sym]
        if family not in FAMILY_BULL:  # MARKET has no tradable ETF mapping
            return no_trade_signal(ts, self.name, reason="no tradable family").to_dict()

        if best_pred.direction == "UP":
            action, symbol = "BUY_BULL", FAMILY_BULL[family]
        else:
            action, symbol = "BUY_BEAR", FAMILY_BEAR[family]
        return {
            "timestamp": ts,
            "agent_name": self.name,
            "agent_version": self.version,
            "target_family": family,
            "direction": best_pred.direction,
            "action": action,
            "symbol": symbol,
            "confidence": round(float(best_pred.confidence), 4),
            "expected_holding_minutes": 30,
            "reason": f"{best_sym} model direction={best_pred.direction}",
            "risk_notes": [],
            "features_used": {"model": self.cfg.model_type, "decision_symbol": best_sym},
            "raw_response": {"proba": best_pred.proba},
        }
