"""BacktestRunner — thin wrapper that runs a one-shot historical backtest.

Provided for symmetry with PaperRunner/LiveRunner. It processes the stored
feature matrix in one pass via the existing pipeline and then exits.
"""
from __future__ import annotations

from ..config.settings import Config
from ..pipeline import backtest
from .base import BaseRunner


class BacktestRunner(BaseRunner):
    name = "backtest"

    def __init__(self, cfg: Config, agent_type: str | None = None,
                 signal_file: str | None = None) -> None:
        super().__init__(cfg)
        self.agent_type = agent_type
        self.signal_file = signal_file

    def run(self) -> dict:
        return backtest(self.cfg, self.agent_type, self.signal_file)
