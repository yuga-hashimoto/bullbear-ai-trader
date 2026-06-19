"""Challengers must be distinct and derive from the best performer's DNA."""
from __future__ import annotations

import dataclasses
import random

from src.evolution.mutation_generator import (
    _patch_key,
    generate_mutations,
    propose_patch,
)


def _cfg(cfg, tmp_path):
    return dataclasses.replace(cfg, paths={**cfg.paths, "reports_dir": str(tmp_path / "r")})


def test_generated_mutations_are_all_distinct(cfg, tmp_path):
    cands = generate_mutations(_cfg(cfg, tmp_path), n=5, seed=3)
    keys = [_patch_key(c["config_patch"]) for c in cands]
    assert len(keys) == 5
    assert len(set(keys)) == len(keys)            # no two identical
    assert all(c["config_patch"] for c in cands)  # none empty (== identical to champion)


def test_avoid_excludes_existing_dna(cfg, tmp_path):
    existing = {"risk.take_profit_pct": 1.5}
    cands = generate_mutations(_cfg(cfg, tmp_path), n=5, seed=7, avoid=[existing])
    keys = {_patch_key(c["config_patch"]) for c in cands}
    assert _patch_key(existing) not in keys


def test_base_patch_seeds_mutations_from_best(cfg, tmp_path):
    # Derive from a champion that already runs take_profit 2.5 — mutations of
    # that gene must cluster near 2.5, not the raw base config (1.2).
    c = _cfg(cfg, tmp_path)
    rng = random.Random(0)
    vals = []
    for _ in range(80):
        p = propose_patch(c, rng, base_patch={"risk.take_profit_pct": 2.5})
        if "risk.take_profit_pct" in p:
            vals.append(p["risk.take_profit_pct"])
    assert vals                                   # the gene was sometimes chosen
    assert all(2.0 <= v <= 3.0 for v in vals)     # clustered around the base 2.5
