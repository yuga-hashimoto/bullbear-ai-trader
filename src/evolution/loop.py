"""Continuous evolution loop + promote/rollback mechanics.

One cycle: shadow-evaluate champion + challengers, update bandit allocations,
graduate strong shadows to canary, evaluate the promotion policy, optionally
auto-promote (after close, policy-gated), detect drift, and generate new
mutations. Champion switches happen only ``after_close``; rollback is allowed
any time. The Risk Engine is never bypassed — a promoted patch is re-validated
by the guardrails first.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from ..config.settings import Config
from ..data.store import load_features
from . import guardrails as gr
from .bandit import ArmStats, Bandit
from .challenger import CANARY, PROMOTED, Challenger
from .champion import Champion
from .drift_detector import detect_drift
from .evaluator import robustness_check
from .experiment_store import EvolutionEventType, ExperimentStore
from .mutation_generator import generate_mutations
from .promotion_policy import evaluate_promotion, load_promotion_policy
from .registry import EvolutionRegistry
from .rollback_policy import evaluate_rollback, load_rollback_policy
from .shadow_runner import run_shadow

REWARD_METRICS = {
    "net_pnl_after_costs": lambda m: m.get("net_pnl_after_costs", 0.0),
    "expectancy": lambda m: m.get("expectancy", 0.0),
    "profit_factor": lambda m: m.get("profit_factor", 0.0),
    "sharpe": lambda m: m.get("sharpe_ratio", 0.0),
    "sortino": lambda m: m.get("sortino_ratio", 0.0),
    "risk_adjusted_expectancy": lambda m: m.get("expectancy", 0.0) / (1.0 + abs(m.get("max_drawdown_pct", 0.0))),
}


def _reward(metric_name: str, m: dict[str, Any]) -> float:
    return REWARD_METRICS.get(metric_name, REWARD_METRICS["risk_adjusted_expectancy"])(m)


# --------------------------------------------------------------- promote
def promote_challenger(base_cfg: Config, registry: EvolutionRegistry, store: ExperimentStore,
                       challenger: Challenger, env: str, reasons: dict[str, Any]) -> bool:
    """Swap the challenger in as the new Champion (guardrail re-validated)."""
    violations = gr.check_patch(challenger.config_patch, base_cfg)
    if violations:
        store.emit(EvolutionEventType.PROMOTION_FAILED,
                   {"challenger_id": challenger.challenger_id, "reason": "guardrail", "violations": violations})
        return False
    current = registry.load_champion()
    registry.push_previous_champion(current)
    new_champ = Champion(
        champion_id=challenger.challenger_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_patch=challenger.config_patch,
        agent_prompt_version=challenger.agent_prompt_version,
        risk_policy_version=challenger.risk_policy_version,
        metrics=challenger.metrics,
        notes=f"auto-promoted from {challenger.challenger_id} ({env})",
    )
    registry.save_champion(new_champ)
    challenger.status = PROMOTED
    registry.update_challenger(challenger)
    registry.record_promotion({"challenger_id": challenger.challenger_id, "env": env, "reasons": reasons})
    store.emit(EvolutionEventType.CHAMPION_PROMOTED,
               {"challenger_id": challenger.challenger_id, "env": env})
    return True


# -------------------------------------------------------------- rollback
def rollback_to_fallback(registry: EvolutionRegistry, store: ExperimentStore,
                         reasons: list[str], on_day: date | None = None,
                         freeze_days: int = 3) -> bool:
    fallback = registry.fallback_champion()
    store.emit(EvolutionEventType.ROLLBACK_TRIGGERED, {"reasons": reasons})
    if fallback is None:
        store.emit(EvolutionEventType.CHAMPION_ROLLED_BACK, {"note": "no fallback champion"})
        return False
    registry.save_champion(Champion.from_dict(fallback))
    registry.record_rollback({"reasons": reasons, "restored": fallback.get("champion_id")})
    on_day = on_day or datetime.now(timezone.utc).date()
    registry.freeze_promotions(on_day + timedelta(days=freeze_days))
    store.emit(EvolutionEventType.CHAMPION_ROLLED_BACK, {"restored": fallback.get("champion_id")})
    return True


def maybe_rollback(registry: EvolutionRegistry, store: ExperimentStore, state: dict[str, Any],
                   policy: dict[str, Any] | None = None, on_day: date | None = None) -> bool:
    """Evaluate the rollback policy against a live/paper degradation state."""
    policy = policy or load_rollback_policy()
    result = evaluate_rollback(state, policy)
    if not result.should_rollback:
        return False
    return rollback_to_fallback(registry, store, result.reasons, on_day,
                                policy.get("freeze_promotions_after_rollback_days", 3))


# ----------------------------------------------------------------- cycle
def run_evolution_cycle(cfg: Config, env: str = "paper", agent_type: str | None = None,
                        signal_file: str | None = None, after_close: bool = True,
                        seed: int = 0) -> dict[str, Any]:
    reports_dir = cfg.path("reports_dir")
    store = ExperimentStore(reports_dir)
    registry = EvolutionRegistry(reports_dir)
    registry.ensure_champion()
    matrix = load_features(cfg)

    shadow = run_shadow(cfg, registry, store, agent_type, signal_file, matrix)
    champ_metrics = shadow["champion"]

    # --- bandit allocation -------------------------------------------------
    adaptive = cfg.raw.get("adaptive_allocation", {})
    reward_metric = adaptive.get("reward_metric", "risk_adjusted_expectancy")
    bandit = Bandit(mode=adaptive.get("mode", "epsilon_greedy"),
                    epsilon=adaptive.get("epsilon", 0.1),
                    max_challenger_allocation_pct=adaptive.get("max_challenger_allocation_pct", 30),
                    min_allocation_pct=adaptive.get("min_allocation_pct", 0),
                    min_trades=load_promotion_policy(cfg.raw.get("evolution", {}).get(
                        "promotion_policy_file", "config/promotion_policy.yaml")).get("min_trades", 30))
    arms = [ArmStats(cid, _reward(reward_metric, m), m.get("num_trades", 0), m.get("max_drawdown_pct", 0.0))
            for cid, m in shadow["challengers"].items()]
    allocations = bandit.allocate(arms)
    for chal in registry.list_challengers():
        if chal.challenger_id in allocations:
            chal.allocation_pct = allocations[chal.challenger_id]
            registry.update_challenger(chal)
    if allocations:
        registry.record_allocation({"allocations": allocations,
                                    "champion_pct": Bandit.champion_allocation(allocations)})
        store.emit(EvolutionEventType.ALLOCATION_UPDATED, {"allocations": allocations})

    # --- canary graduation (strong shadows) --------------------------------
    from .canary_runner import promote_to_canary
    from .challenger import SHADOW
    for chal in registry.list_challengers():
        if chal.status == SHADOW and chal.metrics.get("profit_factor", 0) >= 1.0 \
                and chal.metrics.get("num_trades", 0) > 0:
            promote_to_canary(registry, store, chal.challenger_id,
                              allocations.get(chal.challenger_id, 10.0))

    # --- promotion evaluation ----------------------------------------------
    policy = load_promotion_policy(cfg.raw.get("evolution", {}).get(
        "promotion_policy_file", "config/promotion_policy.yaml"))
    today = datetime.now(timezone.utc).date()
    passing: list[tuple[Challenger, Any]] = []
    for chal in registry.list_challengers():
        if chal.status != CANARY:
            continue
        rob = chal.metrics.get("robustness") or robustness_check(cfg, chal.config_patch, agent_type, signal_file, matrix)
        result = evaluate_promotion(champ_metrics, chal.metrics, env=env,
                                    days_shadow=chal.metrics.get("shadow_days", 0),
                                    days_canary=chal.metrics.get("canary_days", 0),
                                    policy=policy, robustness=rob)
        store.emit(EvolutionEventType.PROMOTION_EVALUATED,
                   {"challenger_id": chal.challenger_id, "passed": result.passed, "reasons": result.reasons})
        if result.passed:
            store.emit(EvolutionEventType.PROMOTION_PASSED, {"challenger_id": chal.challenger_id})
            passing.append((chal, result))
        else:
            store.emit(EvolutionEventType.PROMOTION_FAILED, {"challenger_id": chal.challenger_id})

    # --- auto-promotion (after close, policy-gated) ------------------------
    promoted = None
    can_promote = (after_close and policy.get("environment_allowed", {}).get(env, False)
                   and policy.get("allow_auto_promote_to_champion", False)
                   and not registry.is_frozen(today))
    if can_promote and passing:
        best, result = max(passing, key=lambda pr: pr[0].metrics.get("expectancy", 0.0))
        if promote_challenger(cfg, registry, store, best, env, result.reasons):
            promoted = best.challenger_id

    # --- drift detection ---------------------------------------------------
    drift = detect_drift(registry.load_champion().metrics or champ_metrics, champ_metrics)
    for alert in drift:
        store.record_drift(alert)
        store.emit(EvolutionEventType.DRIFT_DETECTED, alert)

    # --- mutation generation (after close) ---------------------------------
    new_candidates: list[str] = []
    if after_close:
        n = cfg.raw.get("evolution", {}).get("mutations_per_day", 3)
        for cand in generate_mutations(cfg, n=n, seed=seed):
            store.emit(EvolutionEventType.MUTATION_GENERATED, {"candidate_id": cand["candidate_id"]})
            chal = registry.create_challenger(cand["config_patch"], source="mutation",
                                              notes=f"from {cand['candidate_id']}")
            store.emit(EvolutionEventType.CHALLENGER_CREATED, {"challenger_id": chal.challenger_id})
            new_candidates.append(chal.challenger_id)

    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "env": env,
        "champion_id": registry.load_champion().champion_id,
        "num_challengers": len(registry.list_challengers()),
        "allocations": allocations,
        "promoted": promoted,
        "drift_alerts": len(drift),
        "new_candidates": new_candidates,
    }
    store.write_status(status)
    store.emit(EvolutionEventType.EVOLUTION_LOOP_COMPLETED, {"promoted": promoted,
               "challengers": len(registry.list_challengers())})
    return status
