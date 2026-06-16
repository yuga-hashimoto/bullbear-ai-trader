"""ExternalAgentAdapter — future bridge to OpenClaw / HermesAgent.

This is a SKELETON. It defines the transport options and the request shape, but
does not perform real I/O yet. Critically, it never connects on its own: an
endpoint must be explicitly configured. If unconfigured it raises (or the
factory falls back to MockAgent when ``fallback_to_no_trade`` semantics allow).

Planned transports (selected via config ``agent.type`` / ``agent.endpoint``):
  * http        — POST context JSON to an HTTP endpoint
  * command     — spawn a local command, write context to stdin, read signal
  * mcp         — call an MCP tool
  * file        — file-based IPC (write context, poll for signal)
  * stdio       — long-lived subprocess exchanging JSON lines
  * websocket   — bidirectional streaming
"""
from __future__ import annotations

from typing import Any

from ..config.settings import AgentConfig
from ..utils.logging import get_logger
from .base import BaseAgent
from .signal_schema import no_trade_signal

log = get_logger(__name__)

SUPPORTED_TRANSPORTS = {"http", "command", "mcp", "file", "stdio", "websocket"}


class ExternalAgentNotConfiguredError(RuntimeError):
    """Raised when an external agent is requested without an endpoint."""


class ExternalAgentAdapter(BaseAgent):
    name = "ExternalAgent"
    version = "0.0.0-skeleton"

    def __init__(self, agent_cfg: AgentConfig) -> None:
        self.cfg = agent_cfg
        self.transport = (agent_cfg.transport or "http").lower()
        self.endpoint = agent_cfg.endpoint
        self.timeout_seconds = agent_cfg.timeout_seconds
        if self.transport not in SUPPORTED_TRANSPORTS:
            raise ValueError(f"unsupported transport: {self.transport!r}")
        # Never auto-connect: an endpoint must be explicitly provided.
        if not self.endpoint:
            raise ExternalAgentNotConfiguredError(
                "ExternalAgentAdapter requires agent.endpoint to be set. "
                "It will not attempt any connection otherwise."
            )

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        # Intentionally not implemented in this phase. The transport wiring to
        # OpenClaw / HermesAgent must be built and reviewed before enabling.
        raise NotImplementedError(
            f"ExternalAgentAdapter transport {self.transport!r} is not "
            "implemented yet. Use --agent mock or --agent replay for now."
        )

    def safe_request(self, context: dict[str, Any]) -> dict[str, Any]:
        """Request a signal, falling back to NO_TRADE on any failure."""
        try:
            return self.request_signal(context)
        except Exception as exc:  # noqa: BLE001 - external boundary
            log.warning("external agent failed; NO_TRADE fallback: %s", exc)
            return no_trade_signal(context["timestamp"], self.name, reason=f"fallback: {exc}").to_dict()
