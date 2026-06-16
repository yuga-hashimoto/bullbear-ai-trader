"""Shadow evaluation: evaluate champion + challengers on the same data.

Challengers are evaluated virtually (no capital) with the same cost/slippage/
spread assumptions as the champion (the BacktestEngine applies them). Results
feed the bandit allocator and the promotion policy.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..config.settings import Config
from ..data.store import load_features
from .challenger import CANARY, SHADOW
from .evaluator import evaluate, robustness_check
from .experiment_store import EvolutionEventType, ExperimentStore
from .registry import EvolutionRegistry


def run_shadow(
    base_cfg: Config,
    registry: EvolutionRegistry,
    store: ExperimentStore,
    agent_type: str | None = None,
    signal_file: str | None = None,
    matrix: pd.DataFrame | None = None,
    with_robustness: bool = True,
) -> dict[str, Any]:
    if matrix is None:
        matrix = load_features(base_cfg)
    champion = registry.ensure_champion()
    champion_metrics = evaluate(base_cfg, champion.config_patch, agent_type, signal_file, matrix)

    results: dict[str, Any] = {}
    for chal in registry.list_challengers():
        if chal.status not in (SHADOW, CANARY):
            continue
        store.emit(EvolutionEventType.CHALLENGER_SHADOW_STARTED, {"challenger_id": chal.challenger_id})
        m = evaluate(base_cfg, chal.config_patch, agent_type, signal_file, matrix)
        if with_robustness:
            m["robustness"] = robustness_check(base_cfg, chal.config_patch, agent_type, signal_file, matrix)
        # accumulate "days" observed in shadow/canary (one cycle == one day here).
        m["shadow_days"] = chal.metrics.get("shadow_days", 0) + 1
        m["canary_days"] = chal.metrics.get("canary_days", 0) + (1 if chal.status == CANARY else 0)
        chal.metrics = m
        registry.update_challenger(chal)
        store.record_shadow({"challenger_id": chal.challenger_id,
                             "net_pnl_after_costs": m.get("net_pnl_after_costs"),
                             "profit_factor": m.get("profit_factor"),
                             "num_trades": m.get("num_trades")})
        results[chal.challenger_id] = m

    return {"champion": champion_metrics, "challengers": results}
