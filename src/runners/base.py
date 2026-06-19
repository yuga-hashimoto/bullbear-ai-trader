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

DISABLE_FLAG = "disable.flag"


def runtime_dir(cfg: Config) -> Path:
    return cfg.path("reports_dir") / "runtime"


def request_stop(cfg: Config) -> Path:
    """Persistently disable the runner until an explicit start request."""
    d = runtime_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    flag = d / DISABLE_FLAG
    flag.write_text("disabled")
    return flag


def request_start(cfg: Config) -> Path:
    """Clear persistent disable state and return the flag path."""
    flag = runtime_dir(cfg) / DISABLE_FLAG
    if flag.exists():
        flag.unlink()
    return flag


def clear_stop_flag(cfg: Config) -> None:
    """Backward-compatible alias for explicit resume."""
    request_start(cfg)


def runner_disabled(cfg: Config) -> bool:
    return (runtime_dir(cfg) / DISABLE_FLAG).exists()


def stop_requested(cfg: Config) -> bool:
    return runner_disabled(cfg)


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
