"""Evidence-based readiness checks for research, paper, and live operation."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import Config
from ..data.store import load_features
from ..pipeline import _slice
from ..runners.base import runtime_dir
from ..runners.heartbeat import RuntimeWriter


def _check(code: str, ok: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"code": code, "ok": bool(ok), "severity": severity, "detail": detail}


def _has_opencode_key() -> bool:
    if os.getenv("OPENCODE_GO_API_KEY") or os.getenv("OPENCODE_API_KEY"):
        return True
    return Path("/Users/yu-ga/.hermes/.env").exists()


def build_readiness(cfg: Config) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    features_ok = False
    try:
        matrix = load_features(cfg)
        test = _slice(matrix, cfg.test_start, cfg.test_end, cfg.timezone)
        features_ok = not test.empty
        checks.append(_check(
            "features_window",
            features_ok,
            (
                f"{len(test)} rows in {cfg.test_start}..{cfg.test_end}"
                if features_ok
                else "configured test window has no feature rows"
            ),
        ))
    except FileNotFoundError as exc:
        checks.append(_check("features_missing", False, str(exc)))

    key_ok = cfg.agent.type != "external" or _has_opencode_key()
    checks.append(_check(
        "opencode_key",
        key_ok,
        "configured" if key_ok else "external analysis selected but API key is unavailable",
    ))
    live_disabled = not (
        cfg.live_trading_enabled or cfg.trading.allow_live_orders
    )
    checks.append(_check(
        "live_disabled",
        live_disabled,
        "real-money order switches are disabled" if live_disabled else "live switch enabled",
    ))

    hb = RuntimeWriter(runtime_dir(cfg)).read_heartbeat()
    heartbeat_detail = "no heartbeat yet"
    heartbeat_ok = False
    if hb and hb.get("timestamp"):
        try:
            age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(str(hb["timestamp"])).astimezone(timezone.utc)
            ).total_seconds()
            heartbeat_ok = age <= max(cfg.runner.heartbeat_interval_seconds * 3, 120)
            heartbeat_detail = f"heartbeat age {round(age, 1)} seconds"
        except ValueError:
            heartbeat_detail = "heartbeat timestamp is invalid"
    checks.append(_check(
        "paper_heartbeat",
        heartbeat_ok,
        heartbeat_detail,
        severity="warning",
    ))

    paper_ready = features_ok and key_ok and live_disabled
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "ready": paper_ready,
            "running": heartbeat_ok,
        },
        "live_trading": {
            "enabled": not live_disabled,
            "eligible": False,
            "reason": "requires six months paper evidence and explicit reviewed live integration",
        },
        "checks": checks,
    }
