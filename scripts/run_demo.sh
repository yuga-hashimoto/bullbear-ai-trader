#!/usr/bin/env bash
# End-to-end offline demo using synthetic data + the Mock agent (no network).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
CFG="config/synthetic.yaml"

echo "== fetch (synthetic) =="
$PY -m src.cli fetch-data --config "$CFG"
echo "== build features =="
$PY -m src.cli build-features --config "$CFG"

echo "== backtest with MockAgent =="
$PY -m src.cli backtest --config "$CFG" --agent mock

# Use the signals just produced as a replay source (round-trip check).
LATEST_RUN=$($PY -c "import json;print(json.load(open('reports/latest.json'))['run_id'])")
mkdir -p data/signals
cp "reports/runs/${LATEST_RUN}/agent_signals.jsonl" data/signals/sample.jsonl
echo "== validate recorded signals =="
$PY -m src.cli validate-signals --signals data/signals/sample.jsonl

echo "== backtest with ReplayAgent (replaying MockAgent's signals) =="
$PY -m src.cli backtest --config "$CFG" --agent replay --signals data/signals/sample.jsonl

echo "== list runs =="
$PY -m src.cli list-runs --config "$CFG"

echo
echo "Runs written under reports/runs/. View them with:"
echo "    BULLBEAR_REPORTS_DIR=reports streamlit run src/reports/dashboard.py"
