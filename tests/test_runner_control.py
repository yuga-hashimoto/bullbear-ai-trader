from __future__ import annotations

import dataclasses

from src.cli import build_parser
from src.runners.base import (
    request_start,
    request_stop,
    runner_disabled,
)


def _cfg(cfg, tmp_path):
    return dataclasses.replace(
        cfg,
        paths={**cfg.paths, "reports_dir": str(tmp_path / "reports")},
    )


def test_stop_persists_until_explicit_start(cfg, tmp_path):
    cfg2 = _cfg(cfg, tmp_path)

    flag = request_stop(cfg2)

    assert flag.name == "disable.flag"
    assert runner_disabled(cfg2)
    request_start(cfg2)
    assert not runner_disabled(cfg2)


def test_cli_supports_start_runner():
    args = build_parser().parse_args(
        ["start-runner", "--config", "config/default.yaml"]
    )

    assert args.command == "start-runner"


def test_launchd_wrapper_waits_while_disabled():
    script = open("deploy/run-paper-launchd.sh").read()

    assert "disable.flag" in script
    assert "while [[ -f \"$DISABLE_FLAG\" ]]" in script


def test_demo_prefers_project_virtualenv():
    script = open("scripts/run_demo.sh").read()

    assert 'PY="${PYTHON:-.venv/bin/python}"' in script
