"""Strict, non-ordering market analysis returned by OpenCode."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


class MarketAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class MarketAnalysis:
    timestamp: str
    valid_until: str
    target_family: str
    direction: str
    confidence: float
    thesis: str
    invalidation: str
    risk_factors: list[str]
    source_news_ids: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketAnalysis":
        if not isinstance(data, dict):
            raise MarketAnalysisError("analysis must be an object")
        forbidden = {"action", "symbol", "quantity", "order_type", "stop_price"}
        present = sorted(forbidden.intersection(data))
        if present:
            raise MarketAnalysisError(f"order fields are forbidden: {present}")
        required = {
            "timestamp", "valid_until", "target_family", "direction",
            "confidence", "thesis", "invalidation", "risk_factors",
            "source_news_ids",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise MarketAnalysisError(f"missing fields: {missing}")
        try:
            analysis = cls(
                timestamp=str(data["timestamp"]),
                valid_until=str(data["valid_until"]),
                target_family=str(data["target_family"]).upper(),
                direction=str(data["direction"]).upper(),
                confidence=float(data["confidence"]),
                thesis=str(data["thesis"]),
                invalidation=str(data["invalidation"]),
                risk_factors=[str(x) for x in data["risk_factors"]],
                source_news_ids=[str(x) for x in data["source_news_ids"]],
            )
            start = pd.Timestamp(analysis.timestamp)
            end = pd.Timestamp(analysis.valid_until)
        except (TypeError, ValueError) as exc:
            raise MarketAnalysisError(f"malformed analysis: {exc}") from exc
        if analysis.target_family not in {"NASDAQ", "SEMICONDUCTOR", "MARKET"}:
            raise MarketAnalysisError("invalid target_family")
        if analysis.direction not in {"UP", "DOWN", "FLAT"}:
            raise MarketAnalysisError("invalid direction")
        if not 0.0 <= analysis.confidence <= 1.0:
            raise MarketAnalysisError("confidence must be in [0,1]")
        if end <= start:
            raise MarketAnalysisError("valid_until must be after timestamp")
        if not analysis.source_news_ids:
            raise MarketAnalysisError("source_news_ids must not be empty")
        return analysis

    def is_valid_at(self, now: pd.Timestamp) -> bool:
        return pd.Timestamp(self.timestamp) <= now <= pd.Timestamp(self.valid_until)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
