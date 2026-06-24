# ai-trader patterns integrated into bullbear-ai-trader

This document records the safe parts imported from the `whchien/ai-trader` idea set.
No external GPL implementation code is copied here; the implementation is native to
this repository and keeps the existing safety architecture: Agent proposes Signal
JSON, Risk Engine is authoritative, and live orders remain disabled.

## What was adopted

### 1. Built-in technical strategy library

Added `src/research/technical_strategies.py` with deterministic baseline strategies:

- `buy_hold`
- `sma_cross`
- `macd`
- `rsi_reversion`
- `rsi_momentum`
- `bollinger_reversion`
- `bollinger_breakout`
- `momentum`
- `turtle_breakout`
- `vcp_breakout`

These strategies return directional research signals on the decision asset
(`QQQ` or `SMH`). They do not place orders.

### 2. RuleStrategyAgent

Added `src/agents/rule_agent.py` and registered `--agent rule`.

The RuleStrategyAgent converts a research direction into the existing Signal JSON:

- `UP` -> `BUY_BULL` -> `TQQQ` or `SOXL`
- `DOWN` -> `BUY_BEAR` -> `SQQQ` or `SOXS`
- `FLAT` -> `NO_TRADE`
- opposite target while holding -> `EXIT`

The signal still goes through schema validation, Risk Engine validation, execution
costs, sizing, cooldowns, stop logic and report logging.

### 3. Strategy lab CLI

New commands:

```bash
python -m src.cli list-strategies
python -m src.cli quick-strategy --config config/default.yaml --strategy sma_cross --family SEMICONDUCTOR
python -m src.cli quick-strategy --config config/default.yaml --strategy turtle_breakout --family auto --param entry=20 --param exit=10
python -m src.cli strategy-sweep --config config/default.yaml --family auto
python -m src.cli backtest --config config/default.yaml --agent rule
```

`quick-strategy` and `strategy-sweep` write normal run outputs under `reports/runs/<run_id>/`.
That means the dashboard and `compare-runs` can compare these rule baselines against external-agent runs.

### 4. Optional SQLite OHLCV cache

Added `src/data/sqlite_cache.py` and integrated it into `src/data/store.py`.

Enable it in YAML:

```yaml
storage:
  sqlite_enabled: true
  sqlite_db: data/cache/market_data.sqlite
```

Parquet/CSV remains the canonical artifact. SQLite is used as an optional cache and query layer.

Inspect it with:

```bash
python -m src.cli sqlite-cache-status --config config/default.yaml
```

### 5. Optional MCP server

Added `src/mcp/server.py`.

Install FastMCP only when using the MCP server:

```bash
pip install fastmcp
python -m src.mcp.server
```

Exposed tools:

- `list_strategies`
- `fetch_market_data`
- `build_feature_matrix`
- `run_backtest`
- `quick_strategy`
- `cache_status`

All tools call the same safe pipeline. They cannot bypass live-trading safety.

## Recommended next use

1. Run normal data setup:

```bash
python -m src.cli fetch-data --config config/default.yaml
python -m src.cli build-features --config config/default.yaml
```

2. Run all rule baselines:

```bash
python -m src.cli strategy-sweep --config config/default.yaml --family auto
```

3. Compare the best rule run against the existing champion/external-agent run:

```bash
python -m src.cli compare-runs --config config/default.yaml --run-ids <rule_run_id> <champion_run_id>
```

4. Use the best rule strategy as a Challenger seed instead of treating it as a final live system.
