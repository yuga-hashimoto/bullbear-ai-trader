"""Mutation generator: produce challenger candidates by perturbing safe knobs.

Only parameters in :data:`guardrails.PARAM_BOUNDS` are mutated, always within
bounds, and every candidate is re-checked by the guardrails before being kept.
This makes it structurally impossible to generate a candidate that loosens a
safety limit, weakens the live gate, or games the cost model.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from ..config.settings import Config
from . import guardrails as gr

# Knobs eligible for mutation (subset of PARAM_BOUNDS), with integer flags.
_INT_PARAMS = {"risk.max_holding_minutes", "risk.no_trade_first_minutes",
               "risk.no_new_entry_last_minutes", "risk.max_trades_per_day",
               "risk.max_consecutive_losses", "strategy.max_concurrent_positions"}

# High-impact "genes": these bind on (almost) every trade, so mutating them
# produces challengers that genuinely diverge from the champion. We deliberately
# exclude rarely-binding knobs (max_trades_per_day, max_consecutive_losses,
# no_trade_first/last_minutes) and strategy.expected_return_weight, which the
# live numeric/fusion decision path does not consume (a "dead gene" that would
# leave a challenger identical to the base). All remain guardrail-bounded.
_HIGH_IMPACT = [
    # exit / risk genes (trend-riding: take_profit is intentionally NOT a gene —
    # it stays "off" so challengers can't regress into scalpers)
    "risk.confidence_threshold",    # changes WHICH signals become trades (entries)
    "risk.trailing_stop_pct",       # the trend-break exit — the key knob
    "risk.max_loss_per_trade_pct",  # stop-loss — affects losing trades + sizing
    "risk.max_holding_minutes",     # how long a trend is allowed to ride
    # V9 entry genes — challengers explore the validated entry neighbourhood
    "strategy.numeric_min_vwap_dev",
    "strategy.numeric_min_strength",
    "strategy.numeric_rsi_bull_max",
    "strategy.numeric_rsi_bear_min",
]
_MUTABLE = [k for k in _HIGH_IMPACT if k in gr.PARAM_BOUNDS]


def _sample_value(key: str, rng: random.Random) -> Any:
    lo, hi = gr.PARAM_BOUNDS[key]
    if key in _INT_PARAMS:
        return rng.randint(int(lo), int(hi))
    return round(rng.uniform(lo, hi), 5)


def _patch_key(patch: dict[str, Any]) -> tuple:
    """Canonical, hashable key so duplicate DNA can be detected/avoided."""
    return tuple(sorted((k, round(float(v), 5)) for k, v in patch.items()))


def propose_patch(base_cfg: Config, rng: random.Random, k: int = 2,
                  base_patch: dict[str, Any] | None = None) -> dict[str, Any]:
    """Propose a guardrail-valid patch mutating up to ``k`` parameters.

    Derives from ``base_patch`` when given (the best performer's DNA), otherwise
    from the current Champion's patch — so the gene pool always builds on the
    best-known-good configuration rather than the raw base.
    """
    if base_patch is not None:
        current_patch = dict(base_patch)
    else:
        from .registry import EvolutionRegistry
        registry = EvolutionRegistry(base_cfg.path("reports_dir"))
        current_patch = {}
        try:
            champ = registry.load_champion()
            if champ and champ.config_patch:
                current_patch = dict(champ.config_patch)
        except Exception:
            pass  # If no champion exists yet, fallback to base config

    # Helper to get parameter value from Champion or Base Config
    def get_base_value(key: str) -> Any:
        if key in current_patch:
            return current_patch[key]
        parts = key.split(".", 1)
        if len(parts) == 2:
            section, name = parts[0], parts[1]
            if section == "risk":
                return getattr(base_cfg.risk, name, None)
            elif section == "strategy":
                return getattr(base_cfg.strategy, name, None)
        return None

    for _ in range(20):  # retry until a safe patch is produced
        keys = rng.sample(_MUTABLE, k=min(k, len(_MUTABLE)))
        patch = {}
        for key in keys:
            base_val = get_base_value(key)
            if base_val is not None:
                lo, hi = gr.PARAM_BOUNDS[key]
                span = hi - lo
                # Mutate slightly (+/- 15% of total span) around the current champion's value
                noise = rng.uniform(-0.15 * span, 0.15 * span)
                mutated_val = base_val + noise
                mutated_val = max(lo, min(hi, mutated_val))
                
                if key in _INT_PARAMS:
                    patch[key] = int(round(mutated_val))
                else:
                    patch[key] = round(mutated_val, 5)
            else:
                patch[key] = _sample_value(key, rng)
                
        if gr.is_safe(patch, base_cfg):
            # Check if merged with current champion remains safe
            merged_patch = {**current_patch, **patch}
            if gr.is_safe(merged_patch, base_cfg):
                return patch
    return {}  # could not produce a safe patch


def generate_mutations(
    base_cfg: Config,
    n: int = 3,
    seed: int = 0,
    candidates_dir: str | Path | None = None,
    base_patch: dict[str, Any] | None = None,
    avoid: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate ``n`` DISTINCT candidate patches and persist them.

    ``base_patch`` seeds the mutations from the best performer's DNA. ``avoid``
    lists existing patches that the new candidates must NOT duplicate — combined
    with intra-batch dedup this guarantees no two live challengers are identical.
    """
    rng = random.Random(seed)
    out_dir = Path(candidates_dir or (base_cfg.path("reports_dir") / "candidates"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple] = {_patch_key(p) for p in (avoid or []) if p}
    candidates: list[dict[str, Any]] = []
    attempts = 0
    while len(candidates) < n and attempts < n * 40:
        attempts += 1
        patch = propose_patch(base_cfg, rng, base_patch=base_patch)
        if not patch:
            continue
        key = _patch_key(patch)
        if key in seen:  # identical to an existing or already-spawned challenger
            continue
        seen.add(key)
        cand_id = f"cand_{seed}_{len(candidates)}"
        cdir = out_dir / cand_id
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "candidate.yaml").write_text(yaml.safe_dump({"candidate_id": cand_id,
                                                             "config_patch": patch}, sort_keys=False))
        (cdir / "mutation.json").write_text(__import__("json").dumps(patch, indent=2))
        (cdir / "expected_effect.md").write_text(_describe(patch))
        candidates.append({"candidate_id": cand_id, "config_patch": patch})
    return candidates


def _describe(patch: dict[str, Any]) -> str:
    lines = ["# Expected effect\n", "Mutated parameters (within safe bounds):\n"]
    for k, v in patch.items():
        lines.append(f"- `{k}` -> `{v}`")
    lines.append("\nAll changes are guardrail-validated: no safety limit is loosened, "
                 "no cost/slippage/spread is altered, the live gate is untouched, and "
                 "the Risk Engine is not bypassed.")
    return "\n".join(lines)
