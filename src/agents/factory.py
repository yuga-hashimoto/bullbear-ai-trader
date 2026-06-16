"""Agent factory — selects the agent implementation (the swap point).

Resolution order: explicit ``agent_type`` arg, then ``cfg.agent.type``.
External agents that are unconfigured fall back to MockAgent only when
``cfg.agent.fallback_to_no_trade`` is set; otherwise the error propagates so a
misconfiguration never silently changes behavior.
"""
from __future__ import annotations

from pathlib import Path

from ..config.settings import Config
from ..utils.logging import get_logger
from .base import BaseAgent

log = get_logger(__name__)


def make_agent(
    cfg: Config,
    agent_type: str | None = None,
    signal_file: str | Path | None = None,
) -> BaseAgent:
    kind = (agent_type or cfg.agent.type or "mock").lower()

    if kind == "mock":
        from .mock_agent import MockAgent

        return MockAgent()

    if kind == "replay":
        from .replay_agent import ReplayAgent

        path = signal_file or cfg.agent.signal_file
        if not path:
            raise ValueError("replay agent requires --signals / agent.signal_file")
        return ReplayAgent(path)

    if kind == "local_model":
        from .local_model_agent import LocalModelAgent

        return LocalModelAgent(cfg)

    if kind == "external":
        from .external_agent import (
            ExternalAgentAdapter,
            ExternalAgentNotConfiguredError,
        )

        try:
            return ExternalAgentAdapter(cfg.agent)
        except ExternalAgentNotConfiguredError:
            if cfg.agent.fallback_to_no_trade:
                from .mock_agent import MockAgent

                log.warning("external agent not configured; falling back to MockAgent")
                return MockAgent()
            raise

    raise ValueError(f"unknown agent type: {kind!r}")
