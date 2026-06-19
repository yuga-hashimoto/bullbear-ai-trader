"""Live-track-record driven evolution cycle (the automatic "judge").

Unlike :func:`evolution.loop.run_evolution_cycle` (which re-backtests on the
historical matrix), this cycle judges challengers by their *live shadow* track
record — the forward performance accumulated bar-by-bar by ``LiveShadowBook``.
Run once per day after the US close. It:

1. records every challenger's DNA (config patch) + track record to the history DB,
2. updates bandit allocations from live reward,
3. graduates strong shadows to canary,
4. evaluates the (strict, safety-gated) promotion policy,
5. retires the weakest losing DNA, and
6. spawns fresh mutations so the pool keeps exploring — "keep the good DNA".

The Champion's live trading is never touched here; promotion is policy-gated and
re-validated by guardrails (reusing :func:`loop.promote_challenger`).
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from ..config.settings import Config
from ..utils.logging import get_logger
from .bandit import ArmStats, Bandit
from .challenger import CANARY, LIVE_ELIGIBLE, SHADOW, Challenger
from .experiment_store import EvolutionEventType, ExperimentStore
from .history_db import EvolutionHistoryDB
from .loop import promote_challenger
from .mutation_generator import generate_mutations
from .promotion_policy import evaluate_promotion, load_promotion_policy
from .registry import EvolutionRegistry

log = get_logger(__name__)

_ACTIVE = (SHADOW, CANARY)


def _reward(m: dict[str, Any]) -> float:
    return float(m.get("net_pnl_after_costs", 0.0))


def _db_path(cfg: Config):
    return cfg.path("reports_dir") / "evolution" / "history.db"


def run_live_evolution_cycle(cfg: Config, env: str = "paper",
                             after_close: bool = True, seed: int | None = None) -> dict[str, Any]:
    reports_dir = cfg.path("reports_dir")
    registry = EvolutionRegistry(reports_dir)
    store = ExperimentStore(reports_dir)
    registry.ensure_champion()
    champion = registry.load_champion()
    evo = cfg.raw.get("evolution", {})
    ls = evo.get("live_shadow", {})
    num_challengers = int(ls.get("num_challengers", 5))
    canary_min_trades = int(evo.get("canary_min_trades", 5))
    retire_min_trades = int(evo.get("retire_min_trades", 5))
    max_retire = int(evo.get("max_retire_per_cycle", evo.get("mutations_per_day", 3)))
    ts = datetime.now(timezone.utc).isoformat()

    db = EvolutionHistoryDB(_db_path(cfg))
    try:
        db.record_champion(champion.champion_id, champion.config_patch,
                           source="current", metrics=champion.metrics, promoted_at=ts) \
            if not db.champion_history() else None
        active = [c for c in registry.list_challengers() if c.status in _ACTIVE]

        # 1) record DNA + track record -----------------------------------
        for c in active:
            db.upsert_dna(c.challenger_id, c.config_patch, source=c.source,
                          status=c.status, created_at=c.created_at, notes=c.notes)
            db.record_track(c.challenger_id, c.metrics, ts=ts)

        # 2) bandit allocations from live reward --------------------------
        adaptive = cfg.raw.get("adaptive_allocation", {})
        bandit = Bandit(mode=adaptive.get("mode", "epsilon_greedy"),
                        epsilon=adaptive.get("epsilon", 0.1),
                        max_challenger_allocation_pct=adaptive.get("max_challenger_allocation_pct", 30),
                        min_allocation_pct=adaptive.get("min_allocation_pct", 0),
                        min_trades=canary_min_trades)
        arms = [ArmStats(c.challenger_id, _reward(c.metrics), c.metrics.get("num_trades", 0),
                         c.metrics.get("max_drawdown_pct", 0.0)) for c in active]
        allocations = bandit.allocate(arms) if arms else {}
        for c in active:
            if c.challenger_id in allocations:
                c.allocation_pct = allocations[c.challenger_id]
                registry.update_challenger(c)
        if allocations:
            registry.record_allocation({"allocations": allocations,
                                        "champion_pct": Bandit.champion_allocation(allocations)})

        # 3) canary graduation (strong live shadows) ----------------------
        for c in active:
            if (c.status == SHADOW and _reward(c.metrics) > 0
                    and c.metrics.get("num_trades", 0) >= canary_min_trades):
                c.status = CANARY
                registry.update_challenger(c)
                db.upsert_dna(c.challenger_id, c.config_patch, status=CANARY)
                db.record_event("CANARY_GRADUATED", {"challenger_id": c.challenger_id})
                store.emit(EvolutionEventType.CHALLENGER_CANARY_STARTED,
                           {"challenger_id": c.challenger_id})

        # 4) promotion evaluation (strict, safety-gated) ------------------
        policy = load_promotion_policy(evo.get("promotion_policy_file",
                                               "config/promotion_policy.yaml"))
        promoted = None
        passing: list[Challenger] = []
        for c in [x for x in registry.list_challengers() if x.status == CANARY]:
            res = evaluate_promotion(champion.metrics, c.metrics, env=env,
                                     days_shadow=c.metrics.get("shadow_days", 0),
                                     days_canary=c.metrics.get("canary_days", 0),
                                     policy=policy, robustness=c.metrics.get("robustness"))
            db.record_event("PROMOTION_EVALUATED",
                            {"challenger_id": c.challenger_id, "passed": res.passed,
                             "reasons": res.reasons})
            if res.passed:
                c.status = LIVE_ELIGIBLE
                registry.update_challenger(c)
                passing.append(c)
        can_promote = (after_close and policy.get("environment_allowed", {}).get(env, False)
                       and policy.get("allow_auto_promote_to_champion", False)
                       and not registry.is_frozen(datetime.now(timezone.utc).date()))
        if can_promote and passing:
            best = max(passing, key=lambda x: x.metrics.get("total_return_pct", 0.0))
            if promote_challenger(cfg, registry, store, best, env, {}):
                promoted = best.challenger_id
                db.retire_dna(best.challenger_id, status="CHAMPION", reason="promoted")
                db.record_champion(best.challenger_id, best.config_patch,
                                   source="promotion", metrics=best.metrics)
                db.record_event("CHAMPION_PROMOTED", {"challenger_id": best.challenger_id})
                champion = registry.load_champion()

        # 5) retire the weakest losing DNA --------------------------------
        survivors = [c for c in registry.list_challengers() if c.status in _ACTIVE]
        judged = [c for c in survivors
                  if c.metrics.get("num_trades", 0) >= retire_min_trades and _reward(c.metrics) < 0]
        judged.sort(key=lambda c: _reward(c.metrics))  # worst first
        retired: list[str] = []
        for c in judged[:max_retire]:
            survivors = [s for s in survivors if s.challenger_id != c.challenger_id]
            db.retire_dna(c.challenger_id, status="REJECTED",
                          reason=f"net_pnl {_reward(c.metrics):.2f} over "
                                 f"{c.metrics.get('num_trades', 0)} trades")
            db.record_event("CHALLENGER_RETIRED",
                            {"challenger_id": c.challenger_id, "metrics": c.metrics})
            retired.append(c.challenger_id)

        # 6) spawn fresh mutations to refill the pool, derived from the BEST
        #    performer (champion vs best survivor), and never duplicating an
        #    existing challenger's DNA -----------------------------------------
        best_survivor = max(survivors, key=lambda c: _reward(c.metrics), default=None)
        if best_survivor is not None and _reward(best_survivor.metrics) > _reward(champion.metrics):
            base_patch, parent_id = best_survivor.config_patch, best_survivor.challenger_id
        else:
            base_patch, parent_id = champion.config_patch, champion.champion_id
        avoid = [s.config_patch for s in survivors]
        n_new = max(0, num_challengers - len(survivors))
        rng_seed = seed if seed is not None else random.Random().randint(0, 10_000_000)
        spawned: list[str] = []
        for cand in generate_mutations(cfg, n=n_new, seed=rng_seed,
                                       base_patch=base_patch, avoid=avoid):
            chal = Challenger.create(cand["config_patch"], source="mutation",
                                     notes=f"spawned from {parent_id}")
            chal.status = SHADOW
            survivors.append(chal)
            db.upsert_dna(chal.challenger_id, chal.config_patch, parent_id=parent_id,
                          source="mutation", status=SHADOW, created_at=chal.created_at,
                          notes=chal.notes)
            db.record_event("CHALLENGER_SPAWNED",
                            {"challenger_id": chal.challenger_id, "parent_id": parent_id})
            store.emit(EvolutionEventType.CHALLENGER_CREATED, {"challenger_id": chal.challenger_id})
            spawned.append(chal.challenger_id)

        registry._save_challengers(survivors)  # pool == survivors + freshly spawned

        status = {"updated_at": ts, "env": env, "champion_id": champion.champion_id,
                  "active": len(survivors), "promoted": promoted,
                  "retired": retired, "spawned": spawned,
                  "allocations": allocations}
        store.write_status(status)
        store.emit(EvolutionEventType.EVOLUTION_LOOP_COMPLETED,
                   {"promoted": promoted, "retired": len(retired), "spawned": len(spawned)})
        db.record_event("LIVE_EVOLUTION_CYCLE",
                        {"retired": retired, "spawned": spawned, "promoted": promoted})
        return status
    finally:
        db.close()
