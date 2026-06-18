#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="${CONFIG:-config/default.yaml}"

"$PYTHON" -m src.cli doctor --config "$CONFIG"
"$PYTHON" -m src.cli runner-status --config "$CONFIG" || true
