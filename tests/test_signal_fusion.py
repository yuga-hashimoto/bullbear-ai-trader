from __future__ import annotations

from src.strategy.fusion import SignalFusion


NUMERIC_UP = {
    "timestamp": "2026-06-18T10:00:00-04:00",
    "agent_name": "NumericAgent",
    "target_family": "NASDAQ",
    "direction": "UP",
    "action": "BUY_BULL",
    "symbol": "TQQQ",
    "confidence": 0.7,
}


def _analysis(direction: str, valid_until="2026-06-18T10:30:00-04:00"):
    return {
        "timestamp": "2026-06-18T10:00:00-04:00",
        "valid_until": valid_until,
        "target_family": "NASDAQ",
        "direction": direction,
        "confidence": 0.8,
        "thesis": "test",
        "invalidation": "test",
        "risk_factors": [],
        "source_news_ids": ["n1"],
    }


def test_numeric_signal_operates_without_ai():
    result = SignalFusion().fuse(NUMERIC_UP, None, NUMERIC_UP["timestamp"])

    assert result["action"] == "BUY_BULL"
    assert result["confidence"] == 0.7


def test_ai_confirmation_can_only_strengthen_numeric_candidate():
    result = SignalFusion(ai_weight=0.2).fuse(
        NUMERIC_UP, _analysis("UP"), NUMERIC_UP["timestamp"]
    )

    assert result["action"] == "BUY_BULL"
    assert result["confidence"] > NUMERIC_UP["confidence"]


def test_ai_conflict_rejects_numeric_candidate():
    result = SignalFusion().fuse(
        NUMERIC_UP, _analysis("DOWN"), NUMERIC_UP["timestamp"]
    )

    assert result["action"] == "NO_TRADE"
    assert result["reason"] == "ai_conflict"


def test_expired_ai_is_ignored():
    result = SignalFusion().fuse(
        NUMERIC_UP,
        _analysis("DOWN", valid_until="2026-06-18T10:05:00-04:00"),
        "2026-06-18T10:06:00-04:00",
    )

    assert result["action"] == "BUY_BULL"


def test_ai_cannot_create_order_without_numeric_candidate():
    neutral = {
        "timestamp": NUMERIC_UP["timestamp"],
        "agent_name": "NumericAgent",
        "target_family": "MARKET",
        "direction": "FLAT",
        "action": "NO_TRADE",
        "symbol": None,
        "confidence": 0.0,
    }

    result = SignalFusion().fuse(neutral, _analysis("UP"), neutral["timestamp"])

    assert result["action"] == "NO_TRADE"
