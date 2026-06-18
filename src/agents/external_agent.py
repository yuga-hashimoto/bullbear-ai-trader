"""OpenCode adapter for natural-language analysis, never direct orders."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..config.settings import AgentConfig
from ..utils.logging import get_logger
from .analysis_schema import MarketAnalysis
from .base import BaseAgent
from .news_store import NewsStateStore
from .signal_schema import no_trade_signal

log = get_logger(__name__)
SUPPORTED_TRANSPORTS = {"http"}


class ExternalAgentNotConfiguredError(RuntimeError):
    pass


class ExternalAgentAdapter(BaseAgent):
    """Fetch news and return a validated analysis inside a NO_TRADE signal."""

    name = "OpenCodeAnalysisAgent"
    version = "2.0.0"

    def __init__(self, agent_cfg: AgentConfig, state_dir: str | Path = "reports/runtime") -> None:
        self.cfg = agent_cfg
        self.transport = (agent_cfg.transport or "http").lower()
        if self.transport not in SUPPORTED_TRANSPORTS:
            raise ValueError(f"unsupported external transport: {self.transport!r}")
        self.endpoint = agent_cfg.endpoint or "https://opencode.ai/zen/go/v1"
        self.timeout_seconds = agent_cfg.timeout_seconds
        self.model = agent_cfg.model or "deepseek-v4-flash"
        self.api_key = self._load_api_key()
        if not self.api_key:
            raise ExternalAgentNotConfiguredError(
                "OpenCode analysis requires OPENCODE_GO_API_KEY or OPENCODE_API_KEY"
            )
        self.news_store = NewsStateStore(state_dir)

    def _load_api_key(self) -> str | None:
        key = os.getenv("OPENCODE_GO_API_KEY") or os.getenv("OPENCODE_API_KEY")
        if key:
            return key
        hermes_env = Path("/Users/yu-ga/.hermes/.env")
        if not hermes_env.exists():
            return None
        try:
            for line in hermes_env.open(encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() in {"OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"}:
                    return value.strip().strip("'\"")
        except OSError as exc:
            log.warning("failed to read OpenCode key file: %s", exc)
        return None

    def _fetch_news(self, current_time: pd.Timestamp) -> list[dict[str, Any]]:
        now_utc = pd.Timestamp.now(tz="UTC")
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize("America/New_York")
        if abs((current_time.tz_convert("UTC") - now_utc).total_seconds()) >= 600:
            return []
        return self._fetch_live_news()

    def _fetch_live_news(self) -> list[dict[str, Any]]:
        import yfinance as yf

        out: list[dict[str, Any]] = []
        for ticker in ("QQQ", "SMH"):
            try:
                for item in yf.Ticker(ticker).news or []:
                    content = item.get("content", {})
                    news_id = item.get("id") or content.get("id")
                    if not news_id:
                        continue
                    out.append({
                        "id": str(news_id),
                        "title": content.get("title", ""),
                        "summary": content.get("summary", ""),
                        "pubDate": content.get("pubDate", ""),
                    })
            except Exception as exc:  # noqa: BLE001
                log.warning("news fetch failed for %s: %s", ticker, exc)
        return out

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        timestamp = str(context.get("timestamp", ""))
        current_time = pd.Timestamp(timestamp)
        news = [
            item for item in self._fetch_news(current_time)
            if not self.news_store.is_seen(str(item["id"]))
        ]
        if not news:
            return no_trade_signal(timestamp, self.name, "no_new_news").to_dict()

        analysis = self._request_analysis(context, news)
        ids = [str(item["id"]) for item in news]
        if set(analysis.source_news_ids) != set(ids):
            raise ValueError("analysis source_news_ids do not match supplied news")
        self.news_store.mark_seen(ids)
        self.news_store.save_analysis(analysis.to_dict())

        result = no_trade_signal(timestamp, self.name, "analysis_only").to_dict()
        result["agent_version"] = self.version
        result["features_used"] = {
            "analysis_direction": analysis.direction,
            "analysis_confidence": analysis.confidence,
            "analysis_valid_until": analysis.valid_until,
        }
        result["raw_response"] = {"analysis": analysis.to_dict()}
        return result

    def _request_analysis(
        self, context: dict[str, Any], news: list[dict[str, Any]]
    ) -> MarketAnalysis:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Analyze market news. Never produce an order, action, symbol, "
                        "quantity, or price. Return only the requested JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "context": context,
                        "news": news,
                        "required_schema": {
                            "timestamp": context["timestamp"],
                            "valid_until": "ISO-8601 later than timestamp",
                            "target_family": "NASDAQ|SEMICONDUCTOR|MARKET",
                            "direction": "UP|DOWN|FLAT",
                            "confidence": "0..1",
                            "thesis": "string",
                            "invalidation": "string",
                            "risk_factors": ["string"],
                            "source_news_ids": [item["id"] for item in news],
                        },
                    }, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        response = requests.post(
            f"{self.endpoint.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if not choices:
            raise ValueError("OpenCode returned no choices")
        content = choices[0].get("message", {}).get("content", "").strip()
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return MarketAnalysis.from_dict(json.loads(content))

    def safe_request(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.request_signal(context)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenCode analysis failed closed: %s", exc)
            return no_trade_signal(
                context["timestamp"], self.name, reason=f"analysis_error: {exc}"
            ).to_dict()
