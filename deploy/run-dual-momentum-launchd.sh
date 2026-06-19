#!/bin/zsh
set -u

# Monthly dual-momentum rebalance: recompute the paper equity curve and the
# recommendation for the coming month (which ETF to hold, or bonds if risk-off).
# Read-only on the market — never places real orders.

REPO="/Volumes/MOVESPEED/Documents/GitHub/bullbear-ai-trader"
PYTHON="$REPO/.venv/bin/python"
CONFIG="$REPO/config/default.yaml"

tries=0
while [[ ! -d "$REPO" || ! -x "$PYTHON" || ! -f "$CONFIG" ]]; do
  sleep 10
  tries=$((tries + 1))
  [[ $tries -ge 60 ]] && exit 0
done

cd "$REPO" || exit 78
exec "$PYTHON" -m src.cli run-dual-momentum --config "$CONFIG" --leverage 1.5
