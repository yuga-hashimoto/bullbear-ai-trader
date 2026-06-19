"""Convert low-level PaperRunner events into a readable Japanese diary."""
from __future__ import annotations

from typing import Any

import pandas as pd


def _value(event: dict[str, Any], key: str, default: Any = None) -> Any:
    value = event.get(key, default)
    return default if pd.isna(value) else value


def _time_jst(value: Any) -> str:
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("Asia/Tokyo").strftime("%m/%d %H:%M JST")
    except (TypeError, ValueError):
        return "時刻不明"


def translate_reason(reason: str | None) -> str:
    translations = {
        "numeric close-vwap and momentum candidate": (
            "価格が当日の平均売買価格（VWAP）を上回り、"
            "短期の値動きも上向いているため"
        ),
        "no_numeric_edge": "売買する根拠が基準に達しなかったため",
        "ai_conflict": "数値分析とAI分析の方向が一致しなかったため",
        "no_new_news": "新しい重要ニュースがないため",
        "timeout": "AI分析が時間内に完了しなかったため",
        "agent_error": "AI分析でエラーが発生したため",
    }
    if not reason:
        return "理由は特に記録されていません"
    return translations.get(reason, f"システム判定: {reason}")


def _rejection_reason(event: dict[str, Any]) -> str:
    reason = str(_value(event, "rejection_reason", ""))
    current = _value(event, "current_position")
    translations = {
        "position_open": (
            f"現在{current}を保有しているため"
            if current
            else "すでに別の銘柄を保有しているため"
        ),
        "symbol_cooldown": "同じ銘柄を売却した直後の待機時間中のため",
        "max_trades_per_day": "本日の取引回数上限に達したため",
        "max_consecutive_losses": "連続損失後の安全停止条件に達したため",
        "no_trade_first_minutes": "市場開始直後の取引禁止時間中のため",
        "no_new_entry_last_minutes": "市場終了前の新規購入禁止時間中のため",
        "low_confidence": "予測の自信度が基準未満のため",
        "daily_stop": "本日の損失停止条件に達したため",
    }
    return translations.get(reason, f"安全判定: {reason or '理由不明'}")


def format_diary_event(event: dict[str, Any]) -> dict[str, str] | None:
    event_type = str(_value(event, "event", ""))
    timestamp = _time_jst(_value(event, "timestamp"))
    symbol = _value(event, "symbol")

    if event_type == "AGENT_SIGNAL":
        action = str(_value(event, "action", "NO_TRADE"))
        confidence = float(_value(event, "confidence", 0.0))
        reason = translate_reason(str(_value(event, "reason", "")))
        if action == "NO_TRADE":
            message = f"今回は取引しない。{reason}"
        elif action == "BUY_BULL":
            message = (
                f"{symbol}を買う候補。上昇を予想、自信度{confidence:.0%}。{reason}"
            )
        elif action == "BUY_BEAR":
            message = (
                f"{symbol}を買う候補。下落局面を予想、自信度{confidence:.0%}。{reason}"
            )
        else:
            message = f"{symbol or '保有銘柄'}の売却を検討。{reason}"
        return {"time": timestamp, "icon": "🧠", "message": message}

    if event_type == "RISK_DECISION":
        decision = str(_value(event, "decision", ""))
        action = str(_value(event, "action", ""))
        if decision == "NO_ACTION":
            return None
        if decision == "ACCEPT":
            message = f"{symbol}の購入を安全チェックで許可"
        elif decision == "REJECT":
            subject = (
                f"{symbol}の新規購入"
                if action in {"BUY_BULL", "BUY_BEAR"}
                else f"{symbol or '取引'}"
            )
            message = f"{subject}を見送り。{_rejection_reason(event)}"
        else:
            message = f"{symbol or '保有銘柄'}の売却を実行する判断"
        return {"time": timestamp, "icon": "🛡️", "message": message}

    if event_type == "PAPER_FILL":
        shares = _value(event, "shares", 0)
        price = float(_value(event, "price", 0.0))
        return {
            "time": timestamp,
            "icon": "🛒",
            "message": f"{symbol}を{shares:g}株購入。約定価格 ${price:,.2f}",
        }

    if event_type in {"POSITION_CLOSED", "FORCE_EXIT"}:
        pnl = float(_value(event, "net_pnl", 0.0))
        reason = str(_value(event, "reason", ""))
        reasons = {
            "take_profit": "利益確定条件に到達",
            "stop_loss": "損切り条件に到達",
            "trailing_stop": "価格が直近高値から下落したため利益保護",
            "max_holding_time": "最大保有時間に到達",
            "agent_exit": "AIが売却を提案",
            "force_close_eod": "市場終了前の強制決済",
            "force_close_early": "短縮取引日の終了前に決済",
        }
        explanation = reasons.get(reason, f"終了理由: {reason or '不明'}")
        return {
            "time": timestamp,
            "icon": "💰",
            "message": f"{symbol}を売却。損益 {pnl:+.2f}ドル。{explanation}".replace(
                f"{pnl:+.2f}ドル", f"{'+' if pnl >= 0 else '-'}${abs(pnl):.2f}"
            ),
        }

    if event_type == "MARKET_OPEN":
        return {"time": timestamp, "icon": "🔔", "message": "米国市場が開始"}
    if event_type == "RUNNER_STARTED":
        return {"time": timestamp, "icon": "▶️", "message": "ペーパー取引システムを起動"}
    if event_type == "RUNNER_STOPPED":
        return {"time": timestamp, "icon": "⏹️", "message": "ペーパー取引システムを停止"}
    if event_type == "MARKET_DATA_STALE":
        return {
            "time": timestamp,
            "icon": "⚠️",
            "message": "市場データの更新が遅れているため取引を見送り",
        }
    if event_type == "AGENT_ERROR":
        reason = _value(event, "reason") or _value(event, "message", "不明")
        return {
            "time": timestamp,
            "icon": "⚠️",
            "message": f"AI分析エラー。{translate_reason(str(reason))}",
        }
    if event_type == "DAILY_STOP":
        return {
            "time": timestamp,
            "icon": "🛑",
            "message": "安全上限に達したため本日の新規取引を停止",
        }

    return None
