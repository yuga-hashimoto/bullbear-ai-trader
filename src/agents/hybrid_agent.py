"""Script-owned numeric orders with optional OpenCode analysis fusion."""
from __future__ import annotations

from typing import Any

from ..strategy.fusion import SignalFusion
from ..strategy.numeric import NumericSignalStrategy
from .base import BaseAgent
from .external_agent import ExternalAgentAdapter


class HybridAnalysisAgent(BaseAgent):
    name = "NumericPlusOpenCode"
    version = "1.0.0"

    def __init__(
        self,
        analysis_agent: ExternalAgentAdapter,
        numeric: NumericSignalStrategy | None = None,
        fusion: SignalFusion | None = None,
        strategy_cfg: object | None = None,
    ) -> None:
        self.analysis_agent = analysis_agent
        self.numeric = numeric or (
            NumericSignalStrategy.from_config(strategy_cfg)
            if strategy_cfg is not None else NumericSignalStrategy()
        )
        self.fusion = fusion or SignalFusion()

    def request_signal(self, context: dict[str, Any]) -> dict[str, Any]:
        numeric_signal = self.numeric.signal(context)
        analysis_signal = self.analysis_agent.safe_request(context)
        analysis = (analysis_signal.get("raw_response") or {}).get("analysis")
        return self.fusion.fuse(numeric_signal, analysis, context["timestamp"])
