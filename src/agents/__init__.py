"""Agent layer: external trade-decision agents behind a standard Signal schema.

Agents only PROPOSE trades via a validated Signal. They never place orders; the
Risk Engine is the authority and can reject any proposal.
"""
from .base import BaseAgent
from .factory import make_agent
from .signal_schema import Signal, SignalValidationError, no_trade_signal

__all__ = [
    "BaseAgent",
    "make_agent",
    "Signal",
    "SignalValidationError",
    "no_trade_signal",
]
