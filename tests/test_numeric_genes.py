"""V9 entry genes are config-driven and tunable by challengers."""
from __future__ import annotations

import dataclasses

from src.config.settings import StrategyConfig
from src.evolution import guardrails as gr
from src.evolution.champion import apply_patch
from src.strategy.numeric import NumericSignalStrategy


def _ctx(rsi, r=0.0009):
    return {
        "timestamp": "2026-06-18T10:00:00-04:00",
        "symbols": {
            "QQQ": {"close": 101.0, "vwap": 100.0, "rsi": rsi,
                    "returns": {"1_bar": r, "3_bar": r, "6_bar": r, "12_bar": r}},
        },
    }


def test_from_config_reads_genes():
    cfg = StrategyConfig(numeric_min_strength=0.005, numeric_rsi_bull_max=60.0)
    strat = NumericSignalStrategy.from_config(cfg)
    assert strat.min_strength == 0.005
    assert strat.rsi_bull_max == 60.0


def test_rsi_gene_changes_entry():
    # rsi 65: base (max 72) takes the long; a tighter gene (max 62) skips it.
    base = NumericSignalStrategy.from_config(StrategyConfig())
    tight = NumericSignalStrategy.from_config(StrategyConfig(numeric_rsi_bull_max=62.0))
    assert base.signal(_ctx(rsi=65.0))["action"] == "BUY_BULL"
    assert tight.signal(_ctx(rsi=65.0))["action"] == "NO_TRADE"


def test_strength_gene_changes_entry():
    base = NumericSignalStrategy.from_config(StrategyConfig())               # min 0.0008
    strict = NumericSignalStrategy.from_config(StrategyConfig(numeric_min_strength=0.0015))
    assert base.signal(_ctx(rsi=55.0, r=0.0010))["action"] == "BUY_BULL"
    assert strict.signal(_ctx(rsi=55.0, r=0.0010))["action"] == "NO_TRADE"


def test_entry_genes_are_guardrail_bounded_and_patchable(cfg):
    for key in ("strategy.numeric_min_vwap_dev", "strategy.numeric_min_strength",
                "strategy.numeric_rsi_bull_max", "strategy.numeric_rsi_bear_min"):
        assert key in gr.PARAM_BOUNDS
    # a patch within bounds applies; out of bounds is rejected by guardrails
    patched = apply_patch(cfg, {"strategy.numeric_rsi_bull_max": 70.0})
    assert patched.strategy.numeric_rsi_bull_max == 70.0
    assert gr.check_patch({"strategy.numeric_rsi_bull_max": 999.0}, cfg)  # violation list non-empty
