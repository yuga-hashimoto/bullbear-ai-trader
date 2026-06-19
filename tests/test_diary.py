from __future__ import annotations

import json

from src.reports.diary import format_diary_event
from src.reports.loader import load_diary_events


def test_heartbeat_is_hidden_from_diary():
    assert format_diary_event({
        "timestamp": "2026-06-18T14:45:58+00:00",
        "event": "HEARTBEAT",
    }) is None


def test_buy_signal_is_explained_in_japanese_and_jst():
    item = format_diary_event({
        "timestamp": "2026-06-18T10:45:00-04:00",
        "event": "AGENT_SIGNAL",
        "action": "BUY_BULL",
        "symbol": "TQQQ",
        "confidence": 0.75,
        "reason": "numeric close-vwap and momentum candidate",
    })

    assert item == {
        "time": "06/18 23:45 JST",
        "icon": "🧠",
        "message": (
            "TQQQを買う候補。上昇を予想、自信度75%。"
            "価格が当日の平均売買価格（VWAP）を上回り、"
            "短期の値動きも上向いているため"
        ),
    }


def test_risk_rejection_explains_why_order_was_not_sent():
    item = format_diary_event({
        "timestamp": "2026-06-18T10:45:00-04:00",
        "event": "RISK_DECISION",
        "action": "BUY_BULL",
        "symbol": "TQQQ",
        "decision": "REJECT",
        "rejection_reason": "position_open",
        "current_position": "SOXL",
    })

    assert item["message"] == (
        "TQQQの新規購入を見送り。現在SOXLを保有しているため"
    )


def test_closed_trade_reports_profit_and_exit_reason():
    item = format_diary_event({
        "timestamp": "2026-06-18T14:40:01+00:00",
        "event": "POSITION_CLOSED",
        "symbol": "SOXL",
        "net_pnl": 20.13,
        "reason": "take_profit",
    })

    assert item["message"] == "SOXLを売却。損益 +$20.13。利益確定条件に到達"


def test_diary_loader_skips_noise_before_applying_limit(tmp_path):
    runtime = tmp_path / "reports" / "runtime"
    runtime.mkdir(parents=True)
    events = [
        {
            "timestamp": "2026-06-18T10:40:00-04:00",
            "event": "AGENT_SIGNAL",
            "action": "NO_TRADE",
            "confidence": 0.0,
            "reason": "no_numeric_edge",
        },
        *[
            {
                "timestamp": "2026-06-18T14:40:01+00:00",
                "event": "HEARTBEAT",
            }
            for _ in range(100)
        ],
    ]
    (runtime / "paper_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )

    diary = load_diary_events(tmp_path / "reports", limit=10)

    assert len(diary) == 1
    assert "今回は取引しない" in diary[0]["message"]
