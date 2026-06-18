"""ExternalAgentAdapter — functional bridge to OpenCode API (DeepSeek).

Connects to the OpenCode API (or compatible ChatCompletions endpoint) to request
trading signals dynamically using market context. Key loading from environment
variable (OPENCODE_GO_API_KEY) or ~/.hermes/.env is supported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
import requests

from ..config.settings import AgentConfig
from ..utils.logging import get_logger
from .base import BaseAgent
from .signal_schema import no_trade_signal

log = get_logger(__name__)

SUPPORTED_TRANSPORTS = {"http", "command", "mcp", "file", "stdio", "websocket"}


class ExternalAgentNotConfiguredError(RuntimeError):
    """Raised when an external agent is requested without an API key."""


class ExternalAgentAdapter(BaseAgent):
    name = "OpenCodeAgent"
    version = "1.0.0"

    def __init__(self, agent_cfg: AgentConfig) -> None:
        self.cfg = agent_cfg
        self.transport = (agent_cfg.transport or "http").lower()
        self.endpoint = agent_cfg.endpoint or "https://opencode.ai/zen/go/v1"
        self.timeout_seconds = agent_cfg.timeout_seconds
        self.model = agent_cfg.model or "deepseek-v4-flash"
        
        if self.transport not in SUPPORTED_TRANSPORTS:
            raise ValueError(f"unsupported transport: {self.transport!r}")
            
        self.api_key = self._load_api_key()
        if not self.api_key:
            raise ExternalAgentNotConfiguredError(
                "ExternalAgentAdapter requires OPENCODE_GO_API_KEY environment variable "
                "or configured in ~/.hermes/.env. It will not attempt any connection otherwise."
            )
        
        # Track seen news IDs to avoid double-processing
        self.seen_news_ids: set[str] = set()
        
        # Fallback local agent when no news triggers LLM
        from .mock_agent import MockAgent
        self.fallback_agent = MockAgent()

    def _load_api_key(self) -> str | None:
        # 1. Check direct env variable
        key = os.getenv("OPENCODE_GO_API_KEY") or os.getenv("OPENCODE_API_KEY")
        if key:
            return key
        
        # 2. Fallback: Parse ~/.hermes/.env
        hermes_env = Path("/Users/yu-ga/.hermes/.env")
        if hermes_env.exists():
            try:
                with hermes_env.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() in ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"):
                                return v.strip().strip("'\"")
            except Exception as e:
                log.warning("Failed to read ~/.hermes/.env: %s", e)
                
        return None

    def _fetch_news(self, current_time: pd.Timestamp) -> list[dict[str, Any]]:
        # Detect if we are running in real-time or historical backtest.
        # If current_time is within 10 minutes of real UTC time, it's live/paper.
        now_utc = pd.Timestamp.now(tz="UTC")
        try:
            current_time_utc = current_time.tz_convert("UTC")
        except TypeError:
            # Assume local tz is America/New_York if timezone naive
            current_time_utc = current_time.tz_localize("America/New_York").tz_convert("UTC")
            
        is_live = abs((current_time_utc - now_utc).total_seconds()) < 600
        
        if is_live:
            return self._fetch_live_news()
        else:
            # No dummy news in backtests (avoids unwanted API calls)
            return []

    def _fetch_live_news(self) -> list[dict[str, Any]]:
        import yfinance as yf
        tickers = ["QQQ", "SMH"]
        news_items = []
        for t in tickers:
            try:
                raw_news = yf.Ticker(t).news
                if not raw_news:
                    continue
                for n in raw_news:
                    content = n.get("content", {})
                    news_id = n.get("id") or content.get("id")
                    if not news_id:
                        continue
                    news_items.append({
                        "id": news_id,
                        "title": content.get("title", ""),
                        "summary": content.get("summary", ""),
                        "pubDate": content.get("pubDate", ""),
                    })
            except Exception as e:
                log.warning("Failed to fetch live news for %s: %s", t, e)
        return news_items

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        timestamp = context.get("timestamp", "")
        try:
            current_time = pd.Timestamp(timestamp)
        except Exception as exc:
            log.warning("Failed to parse context timestamp %s: %s", timestamp, exc)
            current_time = pd.Timestamp.now()

        # 1. Fetch news
        news_list = self._fetch_news(current_time)
        
        # 2. Filter out already processed news
        new_news = [n for n in news_list if n["id"] not in self.seen_news_ids]
        
        # 3. If no new news, do not call OpenCode API. Fallback to local agent.
        if not new_news:
            log.info("No new news events for timestamp %s. Skipping OpenCode API and falling back to local agent.", timestamp)
            return self.fallback_agent.request_signal(context)

        log.info("Found %d new news events. Requesting signal from OpenCode API.", len(new_news))
        
        context_json = json.dumps(context, indent=2, ensure_ascii=False)
        news_json = json.dumps(new_news, indent=2, ensure_ascii=False)
        
        prompt = f"""Current Market Context (JSON):
{context_json}

Recent News Event(s) (JSON):
{news_json}

Your task is to analyze the market context and the recent news events, and make a day-trading decision for this bar.
This decision is triggered specifically by the occurrence of these new news events.

You must return a raw JSON object strictly conforming to the following fields:

Required Fields:
- timestamp: "{timestamp}" (exactly matching the timestamp in context)
- agent_name: "OpenCodeAgent"
- target_family: "NASDAQ" | "SEMICONDUCTOR" | "MARKET"
- direction: "UP" | "DOWN" | "FLAT"
- action: "BUY_BULL" | "BUY_BEAR" | "NO_TRADE" | "EXIT"
- symbol: "TQQQ" | "SOXL" | "SQQQ" | "SOXS" | null
  - If action is "BUY_BULL", symbol must be "TQQQ" (for NASDAQ) or "SOXL" (for SEMICONDUCTOR).
  - If action is "BUY_BEAR", symbol must be "SQQQ" (for NASDAQ) or "SOXS" (for SEMICONDUCTOR).
  - If action is "NO_TRADE", symbol must be null.
  - If action is "EXIT", symbol must be "TQQQ", "SOXL", "SQQQ", "SOXS", or null.
- confidence: float between 0.0 and 1.0 (how confident you are in this decision)
- expected_holding_minutes: int (estimated holding duration in minutes, >= 0)
- reason: string (brief explanation of your decision based on the news and market context)
- risk_notes: array of strings (brief explanation of potential risks)
- features_used: object (any features or news aspects you paid attention to)

Output only the raw JSON object. Do not include markdown code block (like ```json ... ```) or any other text."""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a day-trading AI agent. You analyze the market context and generate trading signals. You must return your decision in raw JSON format strictly matching the provided schema, with no other text, markdown blocks, or explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        base_url = self.endpoint.rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        log.info("Requesting signal from OpenCode API at %s for timestamp %s", url, timestamp)
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
        except Exception as exc:
            log.error("Failed to connect to OpenCode API: %s", exc)
            raise RuntimeError(f"OpenCode API connection error: {exc}") from exc

        try:
            res_data = response.json()
            choices = res_data.get("choices", [])
            if not choices:
                raise ValueError("No choices returned from model API")
            
            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                raise ValueError("Empty content returned from model API")
            
            # Remove markdown JSON wrapping if present
            if content.startswith("```"):
                lines = content.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```"):
                    if lines[-1].strip() == "```":
                        content = "\n".join(lines[1:-1]).strip()
                    else:
                        content = "\n".join(lines[1:]).strip()
            
            signal_dict = json.loads(content)
            signal_dict["agent_name"] = self.name
            signal_dict["agent_version"] = self.version
            
            # Successfully generated signal from news, add these news IDs to seen list
            for n in new_news:
                self.seen_news_ids.add(n["id"])
                
            return signal_dict
            
        except Exception as exc:
            log.error("Failed to parse OpenCode API response content: %s", exc)
            log.debug("Raw API Response: %s", response.text)
            raise ValueError(f"OpenCode API malformed response: {exc}") from exc

    def safe_request(self, context: dict[str, Any]) -> dict[str, Any]:
        """Request a signal, falling back to NO_TRADE on any failure."""
        try:
            return self.request_signal(context)
        except Exception as exc:  # noqa: BLE001 - external boundary
            log.warning("external agent failed; NO_TRADE fallback: %s", exc)
            return no_trade_signal(context["timestamp"], self.name, reason=f"fallback: {exc}").to_dict()

