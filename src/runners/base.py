"""Runner base + stop-flag coordination.

A runner drives the per-bar trade loop. ``stop-runner`` writes a stop flag file
that the running process picks up on its next loop and exits cleanly. The same
flag lets tests request a clean stop deterministically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..config.settings import Config
from .heartbeat import RuntimeWriter

STOP_FLAG = "stop.flag"


def runtime_dir(cfg: Config) -> Path:
    return cfg.path("reports_dir") / "runtime"


def request_stop(cfg: Config) -> Path:
    """Write the stop flag (used by the `stop-runner` CLI)."""
    d = runtime_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    flag = d / STOP_FLAG
    flag.write_text("stop")
    return flag


def clear_stop_flag(cfg: Config) -> None:
    flag = runtime_dir(cfg) / STOP_FLAG
    if flag.exists():
        flag.unlink()


def stop_requested(cfg: Config) -> bool:
    return (runtime_dir(cfg) / STOP_FLAG).exists()


class BaseRunner(ABC):
    name = "base"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.writer = RuntimeWriter(runtime_dir(cfg))
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def should_stop(self) -> bool:
        return self._stopped or stop_requested(self.cfg)

    @abstractmethod
    def run(self) -> None: ...
