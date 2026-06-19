#!/bin/zsh
set -u

REPO="/Volumes/MOVESPEED/Documents/GitHub/bullbear-ai-trader"
PYTHON="$REPO/.venv/bin/python"
CONFIG="$REPO/config/default.yaml"
DISABLE_FLAG="$REPO/reports/runtime/disable.flag"

# The repository is on an external volume. At login, launchd may start before
# macOS has mounted it. Keep this internal-disk launcher alive until the full
# runtime is available.
while [[ ! -d "$REPO" || ! -x "$PYTHON" || ! -f "$CONFIG" ]]; do
  sleep 10
done

while [[ -f "$DISABLE_FLAG" ]]; do
  sleep 10
done

cd "$REPO" || exit 78
exec "$PYTHON" -m src.cli run-paper \
  --config "$CONFIG" \
  --agent external
