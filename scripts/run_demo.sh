#!/usr/bin/env bash
# End-to-end offline demo using synthetic data + the Mock agent (no network).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
CFG="config/synthetic.yaml"
REPORTS_DIR=$($PY -c "from src.config.settings import load_config; print(load_config('$CFG').paths['reports_dir'])")
SIGNALS_DIR=$($PY -c "from src.config.settings import load_config; print(load_config('$CFG').paths['signals_dir'])")

echo "== fetch (synthetic) =="
$PY -m src.cli fetch-data --config "$CFG"
echo "== build features =="
$PY -m src.cli build-features --config "$CFG"

echo "== backtest with MockAgent =="
$PY -m src.cli backtest --config "$CFG" --agent mock

# Use the signals just produced as a replay source (round-trip check).
LATEST_RUN=$($PY -c "import json;print(json.load(open('$REPORTS_DIR/latest.json'))['run_id'])")
mkdir -p "$SIGNALS_DIR"
cp "$REPORTS_DIR/runs/${LATEST_RUN}/agent_signals.jsonl" "$SIGNALS_DIR/sample.jsonl"
echo "== validate recorded signals =="
$PY -m src.cli validate-signals --signals "$SIGNALS_DIR/sample.jsonl"

echo "== backtest with ReplayAgent (replaying MockAgent's signals) =="
$PY -m src.cli backtest --config "$CFG" --agent replay --signals "$SIGNALS_DIR/sample.jsonl"

echo "== list runs =="
$PY -m src.cli list-runs --config "$CFG"

echo
echo "Runs written under $REPORTS_DIR/runs/. View them with:"
echo "    BULLBEAR_REPORTS_DIR=$REPORTS_DIR streamlit run src/reports/dashboard.py"
