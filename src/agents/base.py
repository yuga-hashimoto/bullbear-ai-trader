"""Agent interface.

An agent proposes trades by returning a standardized signal. It NEVER places
orders. The flow is always:

    build_context -> request_signal -> parse_response -> validate_signal
    -> (Risk Engine) -> (Backtest Execution)

Concrete agents override :meth:`request_signal` (and optionally
:meth:`prepare`). Default ``build_context`` / ``parse_response`` /
``validate_signal`` implementations cover the common case.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .context import ContextInputs, build_agent_context
from .signal_schema import Signal


class BaseAgent(ABC):
    name: str = "BaseAgent"
    version: str = "0.0.0"

    # -- optional batch precompute hook -------------------------------------
    def prepare(self, matrix: pd.DataFrame, feat_cols: list[str]) -> None:
        """Optional: precompute per-bar state before the backtest loop.

        Default is a no-op. ``LocalModelAgent`` uses this to batch model
        inference; ``ReplayAgent`` uses it to index its signal file.
        """
        return None

    # -- interface ----------------------------------------------------------
    def build_context(self, inputs: ContextInputs) -> dict[str, Any]:
        """Turn the engine's observable state into the agent context dict."""
        return build_agent_context(inputs)

    @abstractmethod
    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return a RAW signal dict for this bar (not yet validated)."""
        raise NotImplementedError

    def parse_response(self, raw: dict[str, Any]) -> Signal:
        """Parse a raw signal dict into a :class:`Signal`."""
        return Signal.from_dict(raw)

    def validate_signal(self, signal: Signal) -> Signal:
        """Structurally validate the signal (raises SignalValidationError)."""
        return signal.validate()
