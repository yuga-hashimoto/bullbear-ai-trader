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
_MUTABLE = [k for k in gr.PARAM_BOUNDS if k != "strategy.max_concurrent_positions"]


def _sample_value(key: str, rng: random.Random) -> Any:
    lo, hi = gr.PARAM_BOUNDS[key]
    if key in _INT_PARAMS:
        return rng.randint(int(lo), int(hi))
    return round(rng.uniform(lo, hi), 3)


def propose_patch(base_cfg: Config, rng: random.Random, k: int = 2) -> dict[str, Any]:
    """Propose a guardrail-valid patch mutating up to ``k`` parameters."""
    for _ in range(20):  # retry until a safe patch is produced
        keys = rng.sample(_MUTABLE, k=min(k, len(_MUTABLE)))
        patch = {key: _sample_value(key, rng) for key in keys}
        if gr.is_safe(patch, base_cfg):
            return patch
    return {}  # could not produce a safe patch (shouldn't happen with these bounds)


def generate_mutations(
    base_cfg: Config,
    n: int = 3,
    seed: int = 0,
    candidates_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Generate ``n`` candidate patches and persist them under candidates_dir."""
    rng = random.Random(seed)
    out_dir = Path(candidates_dir or (base_cfg.path("reports_dir") / "candidates"))
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for i in range(n):
        patch = propose_patch(base_cfg, rng)
        if not patch:
            continue
        cand_id = f"cand_{seed}_{i}"
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
