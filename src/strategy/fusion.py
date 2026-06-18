"""Fuse script-generated candidates with non-ordering AI analysis."""
from __future__ import annotations

from copy import deepcopy

import pandas as pd

from ..agents.analysis_schema import MarketAnalysis, MarketAnalysisError
from ..agents.signal_schema import no_trade_signal


class SignalFusion:
    def __init__(self, ai_weight: float = 0.2, conflict_threshold: float = 0.65) -> None:
        if not 0.0 <= ai_weight <= 0.3:
            raise ValueError("ai_weight must be in [0, 0.3]")
        self.ai_weight = ai_weight
        self.conflict_threshold = conflict_threshold

    def fuse(self, numeric: dict, analysis_data: dict | None, now: str) -> dict:
        if numeric.get("action") not in {"BUY_BULL", "BUY_BEAR"}:
            return deepcopy(numeric)
        if not analysis_data:
            return deepcopy(numeric)
        try:
            analysis = MarketAnalysis.from_dict(analysis_data)
        except MarketAnalysisError:
            return deepcopy(numeric)
        if not analysis.is_valid_at(pd.Timestamp(now)):
            return deepcopy(numeric)
        if analysis.target_family not in {numeric.get("target_family"), "MARKET"}:
            return deepcopy(numeric)

        numeric_direction = numeric.get("direction")
        if (
            analysis.direction not in {"FLAT", numeric_direction}
            and analysis.confidence >= self.conflict_threshold
        ):
            return no_trade_signal(now, "SignalFusion", "ai_conflict").to_dict()
        if analysis.direction == numeric_direction:
            result = deepcopy(numeric)
            boost = self.ai_weight * analysis.confidence
            result["confidence"] = round(min(0.99, float(result["confidence"]) + boost), 4)
            result["agent_name"] = "NumericPlusOpenCode"
            result.setdefault("features_used", {})
            result["features_used"]["ai_confirmation"] = analysis.to_dict()
            return result
        return deepcopy(numeric)
