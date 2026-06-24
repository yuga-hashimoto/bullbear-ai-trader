"""Optional FastMCP server for bullbear-ai-trader research workflows.

Install FastMCP separately when you want to expose these tools to Claude,
ChatGPT Apps SDK bridges or LocalAnt-style local connectors:

    pip install fastmcp
    python -m src.mcp.server

The tools are backtest/research only. They call the same pipeline, Agent and Risk
Engine used by the CLI and cannot place real orders.
"""
from __future__ import annotations

from typing import Any

from ..config.settings import load_config
from ..data.store import sqlite_cache_status
from ..pipeline import backtest, build_features, fetch_data
from ..research.technical_strategies import list_strategy_specs


def _configure_rule_agent(cfg, strategy: str, family: str, params: dict[str, Any] | None) -> None:
    raw = cfg.raw.setdefault("rule_agent", {})
    raw["strategy"] = strategy
    raw["family"] = family
    raw["params"] = dict(params or {})


def list_strategies() -> list[dict[str, Any]]:
    """Return available deterministic strategy baselines."""
    return [s.to_dict() for s in list_strategy_specs()]


def fetch_market_data(
    config: str = "config/default.yaml",
    symbols: list[str] | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    """Fetch OHLCV data through the configured read-only data source."""
    from dataclasses import replace

    cfg = load_config(config)
    if interval:
        cfg = replace(cfg, interval=interval)
    return {"fetched": fetch_data(cfg, symbols)}


def build_feature_matrix(config: str = "config/default.yaml") -> dict[str, str]:
    """Build the causal feature matrix from stored OHLCV data."""
    cfg = load_config(config)
    return {"features": str(build_features(cfg))}


def run_backtest(
    config: str = "config/default.yaml",
    agent: str | None = None,
    signals: str | None = None,
) -> dict[str, Any]:
    """Run a normal backtest using any supported safe agent type."""
    cfg = load_config(config)
    return backtest(cfg, agent, signals)


def quick_strategy(
    config: str = "config/default.yaml",
    strategy: str = "sma_cross",
    family: str = "auto",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one built-in rule strategy through the normal Risk Engine."""
    cfg = load_config(config)
    _configure_rule_agent(cfg, strategy, family, params)
    return backtest(cfg, "rule", None)


def cache_status(config: str = "config/default.yaml") -> dict[str, Any]:
    """Inspect the optional SQLite OHLCV cache."""
    cfg = load_config(config)
    return {"cache": sqlite_cache_status(cfg)}


def _build_mcp():
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("FastMCP is not installed. Run: pip install fastmcp") from exc

    mcp = FastMCP("bullbear-ai-trader")
    mcp.tool()(list_strategies)
    mcp.tool()(fetch_market_data)
    mcp.tool()(build_feature_matrix)
    mcp.tool()(run_backtest)
    mcp.tool()(quick_strategy)
    mcp.tool()(cache_status)
    return mcp


def main() -> None:
    _build_mcp().run()


if __name__ == "__main__":
    main()
