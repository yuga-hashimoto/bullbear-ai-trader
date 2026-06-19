#!/bin/zsh
set -u

# Daily evolution "judge": after the US close, score challengers on their live
# shadow track records, retire the weakest losing DNA, keep the winners, and
# spawn fresh mutations. All DNA + track records are persisted to the history DB
# (reports/evolution/history.db). Champion promotion stays strict/policy-gated.

REPO="/Volumes/MOVESPEED/Documents/GitHub/bullbear-ai-trader"
PYTHON="$REPO/.venv/bin/python"
CONFIG="$REPO/config/default.yaml"

# Repo lives on an external volume; wait for it to mount before running.
tries=0
while [[ ! -d "$REPO" || ! -x "$PYTHON" || ! -f "$CONFIG" ]]; do
  sleep 10
  tries=$((tries + 1))
  [[ $tries -ge 60 ]] && exit 0   # give up after ~10 min; try again next day
done

cd "$REPO" || exit 78
exec "$PYTHON" -m src.cli run-live-evolution --config "$CONFIG" --env paper
