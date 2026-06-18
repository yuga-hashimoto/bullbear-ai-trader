"""Configuration loading, validation and the live-trading safety gate.

The config is loaded from YAML into typed dataclasses. Validation fails fast on
bad input. The most important function here is :func:`assert_live_trading_allowed`,
which is the single choke point that all real-order code paths must pass through.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --- Typed sub-configs ------------------------------------------------------
@dataclass(frozen=True)
class LabelingConfig:
    horizon_bars: int = 6
    up_threshold: float = 0.0015
    down_threshold: float = -0.0015


@dataclass(frozen=True)
class StrategyConfig:
    max_concurrent_positions: int = 1
    expected_return_weight: float = 1.0


@dataclass(frozen=True)
class AgentConfig:
    """How signals are produced. Decision-making lives OUTSIDE this repo."""

    type: str = "mock"            # mock | replay | external | local_model
    endpoint: str | None = None
    signal_file: str | None = None
    transport: str = "http"       # http | command | mcp | file | stdio | websocket
    timeout_seconds: int = 30
    fallback_to_no_trade: bool = True
    model: str = "deepseek-v4-flash"


@dataclass(frozen=True)
class TradingConfig:
    """Live-trading switches. All default to the safe (disabled) state."""

    live_trading_enabled: bool = False
    require_manual_live_unlock: bool = True
    allow_live_orders: bool = False
    broker: str = "backtest"      # backtest | paper | moomoo


@dataclass(frozen=True)
class RunnerConfig:
    """Continuous-runner settings (PaperRunner / future LiveRunner)."""

    mode: str = "paper"                      # backtest | paper | live
    interval: str = "5m"
    timezone: str = "America/New_York"
    regular_session_only: bool = True
    extended_hours_enabled: bool = False
    poll_interval_seconds: int = 10
    align_to_bar_boundary: bool = True
    prevent_duplicate_bar_processing: bool = True
    stale_data_threshold_seconds: int = 180
    max_agent_latency_seconds: int = 30
    max_data_errors_before_stop: int = 3
    max_agent_errors_before_stop: int = 3
    heartbeat_interval_seconds: int = 30


@dataclass(frozen=True)
class MarketConfig:
    calendar: str = "XNYS"
    session_timezone: str = "America/New_York"
    regular_open: str = "09:30"
    regular_close: str = "16:00"
    early_close_force_exit_minutes_before_close: int = 5


@dataclass(frozen=True)
class AccountConfig:
    base_currency: str = "JPY"
    initial_capital_jpy: float = 1_000_000.0
    usd_jpy_rate: float = 150.0


@dataclass(frozen=True)
class RiskConfig:
    confidence_threshold: float = 0.65
    max_loss_per_trade_pct: float = 0.7
    take_profit_pct: float = 1.2
    trailing_stop_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_trades_per_day: int = 3
    max_consecutive_losses: int = 2
    max_holding_minutes: int = 120
    no_trade_first_minutes: int = 10
    no_new_entry_last_minutes: int = 30
    force_close_minutes_before_close: int = 5
    max_spread_pct: float = 0.15
    max_atr_pct: float = 5.0
    min_bars_between_same_symbol: int = 3
    max_loss_per_trade_jpy: float = 10_000.0
    max_daily_loss_jpy: float = 50_000.0
    max_drawdown_pct: float = 15.0
    max_portfolio_risk_pct: float = 3.0
    overnight_gap_risk_pct: float = 5.0
    allow_overnight_positions: bool = False


@dataclass(frozen=True)
class CostConfig:
    commission_per_share: float = 0.0
    commission_pct: float = 0.0005
    min_commission: float = 0.0
    slippage_pct: float = 0.02
    spread_pct: float = 0.03


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100000.0
    position_sizing: str = "fixed_fraction"
    fraction_per_trade: float = 1.0
    fill_model: str = "next_bar_open"
    random_seed: int = 42


@dataclass(frozen=True)
class Config:
    symbols: list[str]
    context_symbols: list[str]
    benchmark_symbols: list[str]
    instruments: dict[str, dict[str, Any]]
    data_source: str
    interval: str
    start_date: str
    end_date: str
    timezone: str
    session_open: str
    session_close: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    walk_forward: dict[str, Any]
    model_type: str
    model_params: dict[str, Any]
    labeling: LabelingConfig
    strategy: StrategyConfig
    agent: AgentConfig
    trading: TradingConfig
    runner: RunnerConfig
    market: MarketConfig
    account: AccountConfig
    risk: RiskConfig
    costs: CostConfig
    backtest: BacktestConfig
    live_trading_enabled: bool
    broker: str
    paths: dict[str, str]
    log_level: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- convenience path accessors --
    def path(self, key: str) -> Path:
        return Path(self.paths[key])

    @property
    def all_symbols(self) -> list[str]:
        """Tradable + context symbols, de-duplicated, order preserved."""
        seen: dict[str, None] = {}
        for s in [*self.symbols, *self.context_symbols]:
            seen.setdefault(s, None)
        return list(seen.keys())


def _validate(cfg: Config) -> None:
    if cfg.interval not in {"1m", "5m", "15m"}:
        raise ValueError(f"unsupported interval: {cfg.interval}")
    if cfg.data_source not in {"yfinance", "moomoo", "synthetic"}:
        raise ValueError(f"unsupported data_source: {cfg.data_source}")
    if not cfg.symbols:
        raise ValueError("symbols must not be empty")
    if cfg.labeling.up_threshold <= 0 or cfg.labeling.down_threshold >= 0:
        raise ValueError("up_threshold must be > 0 and down_threshold < 0")
    if not (0.0 <= cfg.risk.confidence_threshold <= 1.0):
        raise ValueError("risk.confidence_threshold must be in [0, 1]")
    if cfg.strategy.max_concurrent_positions < 1:
        raise ValueError("max_concurrent_positions must be >= 1")
    if cfg.account.base_currency != "JPY":
        raise ValueError("account.base_currency must be JPY")
    if cfg.account.initial_capital_jpy <= 0 or cfg.account.usd_jpy_rate <= 0:
        raise ValueError("account capital and FX rate must be positive")
    if cfg.risk.max_loss_per_trade_jpy <= 0 or cfg.risk.max_daily_loss_jpy <= 0:
        raise ValueError("JPY risk limits must be positive")
    if not (0 < cfg.risk.max_drawdown_pct < 100):
        raise ValueError("risk.max_drawdown_pct must be in (0, 100)")
    if not (0 < cfg.risk.max_portfolio_risk_pct <= cfg.risk.max_drawdown_pct):
        raise ValueError("portfolio risk must be positive and no larger than max drawdown")
    if cfg.agent.type not in {"mock", "replay", "external", "local_model"}:
        raise ValueError(f"unknown agent.type: {cfg.agent.type}")
    for sym, spec in cfg.instruments.items():
        if "decision" not in spec or "direction" not in spec:
            raise ValueError(f"instrument {sym} needs 'decision' and 'direction'")
        if int(spec["direction"]) not in (-1, 1):
            raise ValueError(f"instrument {sym} direction must be +1 or -1")


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # trading section with backward-compatible top-level fallback.
    trading_raw = dict(raw.get("trading", {}))
    live = bool(trading_raw.get("live_trading_enabled", raw.get("live_trading_enabled", False)))
    broker = trading_raw.get("broker", raw.get("broker", "backtest"))
    trading = TradingConfig(
        live_trading_enabled=live,
        require_manual_live_unlock=bool(trading_raw.get("require_manual_live_unlock", True)),
        allow_live_orders=bool(trading_raw.get("allow_live_orders", False)),
        broker=broker,
    )

    cfg = Config(
        symbols=list(raw["symbols"]),
        context_symbols=list(raw.get("context_symbols", [])),
        benchmark_symbols=list(raw.get("benchmark_symbols", [])),
        instruments=dict(raw.get("instruments", {})),
        data_source=raw.get("data_source", "yfinance"),
        interval=raw.get("interval", "5m"),
        start_date=str(raw["start_date"]),
        end_date=str(raw["end_date"]),
        timezone=raw.get("timezone", "America/New_York"),
        session_open=raw.get("session_open", "09:30"),
        session_close=raw.get("session_close", "16:00"),
        train_start=str(raw.get("train_start", raw["start_date"])),
        train_end=str(raw.get("train_end", raw["end_date"])),
        test_start=str(raw.get("test_start", raw["start_date"])),
        test_end=str(raw.get("test_end", raw["end_date"])),
        walk_forward=dict(raw.get("walk_forward", {})),
        model_type=raw.get("model_type", "lightgbm"),
        model_params=dict(raw.get("model_params", {})),
        labeling=LabelingConfig(**raw.get("labeling", {})),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        agent=AgentConfig(**raw.get("agent", {})),
        trading=trading,
        runner=RunnerConfig(**raw.get("runner", {})),
        market=MarketConfig(**raw.get("market", {})),
        account=AccountConfig(**raw.get("account", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        costs=CostConfig(**raw.get("costs", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        live_trading_enabled=live,
        broker=broker,
        paths=dict(
            raw.get(
                "paths",
                {
                    "raw_dir": "data/raw",
                    "features_dir": "data/features",
                    "artifacts_dir": "artifacts",
                    "reports_dir": "reports",
                },
            )
        ),
        log_level=raw.get("log_level", "INFO"),
        raw=raw,
    )
    _validate(cfg)
    return cfg


# --- LIVE TRADING SAFETY GATE -----------------------------------------------
ALLOW_LIVE_ENV = "BULLBEAR_ALLOW_LIVE"


class LiveTradingDisabledError(RuntimeError):
    """Raised when a real-order path is reached without full authorization."""


def assert_live_trading_allowed(cfg: Config, explicit_flag: bool = False) -> None:
    """The single choke point for enabling real orders.

    Requires ALL THREE independent switches to be set, by design:
      1. ``cfg.live_trading_enabled is True`` (config file)
      2. env var ``BULLBEAR_ALLOW_LIVE == "1"`` (deployment environment)
      3. ``explicit_flag is True`` (explicit CLI/API argument)

    ``trading.require_manual_live_unlock`` (default true) documents that the
    explicit flag must be supplied by a human. Any backtest or paper path must
    NEVER call this. If reached without all three, it raises and refuses.
    """
    reasons: list[str] = []
    if not cfg.live_trading_enabled:
        reasons.append("config.live_trading_enabled is false")
    if os.environ.get(ALLOW_LIVE_ENV) != "1":
        reasons.append(f"env {ALLOW_LIVE_ENV} != '1'")
    if not explicit_flag:
        reasons.append("explicit authorization flag not provided")
    if reasons:
        raise LiveTradingDisabledError(
            "Live trading is disabled. Unmet conditions: " + "; ".join(reasons)
        )
