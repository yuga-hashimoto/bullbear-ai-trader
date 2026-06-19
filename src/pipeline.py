"""High-level pipeline steps wired together for the CLI.

    fetch_data -> build_features -> (train, optional) -> backtest

Decision-making is delegated to an Agent (mock / replay / external / local_model).
``train`` remains only as a helper for the optional LocalModelAgent — it is NOT
on the main decision path. All steps are backtest/research only; nothing here
can place a real order.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .agents.factory import make_agent
from .backtest.engine import BacktestEngine
from .backtest.metrics import benchmark_comparison, compute_metrics
from .config.settings import Config
from .data.clean import clean_ohlcv
from .data.store import load_features, load_raw, make_data_source, save_features, save_raw
from .features.builder import (
    build_feature_matrix,
    feature_columns,
    prepare_feature_matrix,
    price_col,
)
from .labeling.labels import attach_labels, label_col
from .models.base import DirectionModel
from .models.factory import make_model
from .reports.report import write_reports
from .reports.runs import new_run_id, run_dir, save_run
from .utils.logging import get_logger

log = get_logger(__name__)

_INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15}


# --- step 1: fetch ----------------------------------------------------------
def fetch_data(cfg: Config, symbols: list[str] | None = None) -> dict[str, int]:
    source = make_data_source(cfg)
    symbols = symbols or cfg.all_symbols
    counts: dict[str, int] = {}
    for sym in symbols:
        try:
            df = source.fetch(sym, cfg.interval, cfg.start_date, cfg.end_date)
        except NotImplementedError:
            raise
        except Exception as exc:
            log.warning("fetch failed for %s: %s", sym, exc)
            counts[sym] = 0
            continue
        if df.empty:
            log.warning("no data for %s", sym)
            counts[sym] = 0
            continue
        save_raw(cfg, sym, df)
        counts[sym] = len(df)
        log.info("saved %s rows for %s", len(df), sym)
    return counts


# --- step 2: features + labels ---------------------------------------------
def _load_frames(cfg: Config, symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            raw = load_raw(cfg, sym)
        except FileNotFoundError:
            log.warning("raw data missing for %s; skipping", sym)
            continue
        frames[sym] = clean_ohlcv(raw, cfg.timezone, cfg.session_open, cfg.session_close)
    return frames


def build_features(cfg: Config) -> Path:
    frames = _load_frames(cfg, cfg.all_symbols)
    if not frames:
        raise RuntimeError("no raw data found; run fetch-data first")
    matrix = build_feature_matrix(frames, cfg)
    matrix = attach_labels(matrix, frames, cfg)
    matrix, health = prepare_feature_matrix(matrix)
    path = save_features(cfg, matrix)
    health_path = cfg.path("features_dir") / "feature_health_report.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(json.dumps(health, indent=2, sort_keys=True))
    log.info("features saved: %s rows x %s cols -> %s", *matrix.shape, path)
    return path


# --- step 3: train (LocalModelAgent helper only) ----------------------------
def _slice(df: pd.DataFrame, start: str, end: str, tz: str) -> pd.DataFrame:
    lo = pd.Timestamp(start, tz=tz)
    hi = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1)
    return df[(df.index >= lo) & (df.index < hi)]


def _decision_symbols(cfg: Config) -> list[str]:
    seen: dict[str, None] = {}
    for spec in cfg.instruments.values():
        seen.setdefault(str(spec["decision"]), None)
    return list(seen.keys())


def train(cfg: Config) -> dict[str, Path]:
    matrix = load_features(cfg)
    feat_cols = feature_columns(matrix)
    train_df = _slice(matrix, cfg.train_start, cfg.train_end, cfg.timezone)
    artifacts: dict[str, Path] = {}
    for sym in _decision_symbols(cfg):
        lcol = label_col(sym)
        if lcol not in train_df.columns:
            continue
        sub = train_df[[*feat_cols, lcol]].dropna()
        if sub.empty or sub[lcol].nunique() < 2:
            log.warning("insufficient training data for %s; skipping", sym)
            continue
        model: DirectionModel = make_model(cfg)
        model.fit(sub[feat_cols], sub[lcol])
        out = cfg.path("artifacts_dir") / f"model_{sym.replace('^', '_idx_')}.pkl"
        model.save(str(out))
        artifacts[sym] = out
        log.info("trained model for %s on %s rows -> %s", sym, len(sub), out)
    if not artifacts:
        raise RuntimeError("no models trained; check labels and date ranges")
    return artifacts


# --- step 4: backtest -------------------------------------------------------
def _close_frames_from_matrix(cfg: Config, test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in cfg.benchmark_symbols:
        col = price_col(sym, "close")
        if col in test_df.columns:
            out[sym] = pd.DataFrame({"close": test_df[col]}).dropna()
    return out


def backtest(
    cfg: Config,
    agent_type: str | None = None,
    signal_file: str | None = None,
) -> dict:
    matrix = load_features(cfg)
    test_df = _slice(matrix, cfg.test_start, cfg.test_end, cfg.timezone)
    if test_df.empty:
        raise RuntimeError("test slice is empty; check test_start/test_end")

    agent = make_agent(cfg, agent_type, signal_file)
    engine = BacktestEngine(cfg, agent)
    result = engine.run(test_df)

    interval_min = _INTERVAL_MIN[cfg.interval]
    metrics = compute_metrics(result, interval_min)
    bench = benchmark_comparison(_close_frames_from_matrix(cfg, test_df), cfg.benchmark_symbols)

    run_id = new_run_id()
    d = run_dir(cfg, run_id)
    report_paths = write_reports(
        d, metrics, bench, result.trades_frame, result.daily_pnl,
        title=f"Backtest {cfg.test_start}..{cfg.test_end} ({cfg.interval}) agent={agent.name}",
        counters=result.counters,
    )
    overview = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": cfg.symbols,
        "period": {"start": cfg.test_start, "end": cfg.test_end},
        "interval": cfg.interval,
        "agent_type": agent.name,
        "initial_cash": cfg.backtest.initial_cash,
        "total_return_pct": metrics["total_return_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "win_rate_pct": metrics["win_rate_pct"],
        "profit_factor": metrics["profit_factor"],
        "num_trades": metrics["num_trades"],
        "no_trade_ratio": metrics["no_trade_ratio"],
        "rejected_signals": metrics["rejected_signals"],
        "forced_exits": metrics["forced_exits"],
    }
    save_run(cfg, run_id, result, metrics, bench, overview, report_paths)
    log.info("backtest run %s done: %s", run_id, metrics)
    return {
        "run_id": run_id,
        "run_dir": str(d),
        "metrics": metrics,
        "benchmark": bench,
        "counters": result.counters,
    }


def run_pipeline(
    cfg: Config,
    agent_type: str | None = None,
    signal_file: str | None = None,
) -> dict:
    fetch_data(cfg)
    build_features(cfg)
    if (agent_type or cfg.agent.type) == "local_model":
        train(cfg)
    return backtest(cfg, agent_type, signal_file)
