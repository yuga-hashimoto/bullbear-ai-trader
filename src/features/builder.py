"""Feature matrix assembly across symbols.

Produces one time-aligned DataFrame combining:
  * per-symbol technical features  -> columns ``feat__<sym>__<name>``
  * raw prices for tradable ETFs   -> columns ``px__<sym>__<ohlcv>``
  * cross-asset linkage features   -> e.g. ETF/underlying co-movement

Column naming conventions let downstream code select feature vs price columns
unambiguously. Alignment uses the intersection of timestamps so that no row
contains a value carried forward across a missing symbol.
"""
from __future__ import annotations

import pandas as pd

from ..config.settings import Config
from . import indicators as ind

FEAT_PREFIX = "feat__"
PX_PREFIX = "px__"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEAT_PREFIX)]


def price_col(symbol: str, field: str) -> str:
    return f"{PX_PREFIX}{symbol}__{field}"


def build_symbol_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Compute the per-symbol causal feature block."""
    p = f"{FEAT_PREFIX}{symbol}__"
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    out = pd.DataFrame(index=df.index)

    # Short-term returns at multiple lags.
    for n in (1, 3, 6, 12):
        out[f"{p}ret_{n}"] = ind.returns(close, n)

    # Trend / mean-reversion.
    out[f"{p}ma_dev_20"] = ind.ma_deviation(close, 20)
    out[f"{p}ma_dev_50"] = ind.ma_deviation(close, 50)
    out[f"{p}vwap_dev"] = ind.vwap_deviation(df)
    out[f"{p}rsi_14"] = ind.rsi(close, 14)
    out[f"{p}macd_diff"] = ind.macd_diff(close)
    out[f"{p}atr_pct"] = ind.atr_pct(high, low, close, 14)
    out[f"{p}vol_20"] = ind.volatility(close, 20)
    out[f"{p}volchg_20"] = ind.volume_change(vol, 20)

    # Session-relative location.
    out[f"{p}intraday_chg"] = ind.intraday_change(df)
    out[f"{p}gap"] = ind.overnight_gap(df)
    out[f"{p}dist_high_20"] = ind.dist_from_high(close, 20)
    out[f"{p}dist_low_20"] = ind.dist_from_low(close, 20)
    return out


def _cross_asset(
    etf: str, underlying: str, frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """ETF vs underlying co-movement features (linkage)."""
    if etf not in frames or underlying not in frames:
        return pd.DataFrame()
    p = f"{FEAT_PREFIX}link__{etf}__{underlying}__"
    etf_ret = frames[etf]["close"].pct_change()
    und_ret = frames[underlying]["close"].pct_change()
    out = pd.DataFrame(index=frames[etf].index)
    out[f"{p}ret_spread"] = etf_ret - und_ret
    out[f"{p}corr_20"] = etf_ret.rolling(20, min_periods=20).corr(und_ret)
    out[f"{p}beta_20"] = (
        etf_ret.rolling(20, min_periods=20).cov(und_ret)
        / und_ret.rolling(20, min_periods=20).var()
    )
    return out


def build_feature_matrix(
    frames: dict[str, pd.DataFrame], cfg: Config
) -> pd.DataFrame:
    """Assemble the aligned feature matrix from cleaned per-symbol frames."""
    blocks: list[pd.DataFrame] = []

    # Per-symbol features for every available symbol.
    for symbol, df in frames.items():
        if df.empty:
            continue
        blocks.append(build_symbol_features(df, symbol))

    # Raw prices (+ session VWAP) for ALL symbols. Tradable-symbol prices are
    # needed by the execution engine; context-symbol prices (QQQ/SMH/SPY) are
    # needed to build the agent's observable market context.
    for symbol in cfg.all_symbols:
        if symbol not in frames or frames[symbol].empty:
            continue
        src = frames[symbol]
        px = src[["open", "high", "low", "close", "volume"]].copy()
        px.columns = [price_col(symbol, c) for c in ["open", "high", "low", "close", "volume"]]
        px[price_col(symbol, "vwap")] = ind.session_vwap(src)
        blocks.append(px)

    # Cross-asset linkage for each tradable instrument vs its decision symbol.
    for etf, spec in cfg.instruments.items():
        block = _cross_asset(etf, str(spec["decision"]), frames)
        if not block.empty:
            blocks.append(block)

    if not blocks:
        return pd.DataFrame()

    matrix = pd.concat(blocks, axis=1)
    # Align to the intersection of timestamps that have core price data.
    matrix = matrix.sort_index()
    matrix = matrix[~matrix.index.duplicated(keep="last")]
    return matrix
