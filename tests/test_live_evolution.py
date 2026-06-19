"""Live-track-record driven evolution: retire losers, keep winners, spawn new."""
from __future__ import annotations

import dataclasses

from src.evolution.challenger import SHADOW
from src.evolution.history_db import EvolutionHistoryDB
from src.evolution.live_evolution import run_live_evolution_cycle
from src.evolution.registry import EvolutionRegistry


def _cfg(cfg, tmp_path, num=3):
    paths = {**cfg.paths, "reports_dir": str(tmp_path / "reports")}
    raw = {**cfg.raw, "evolution": {**cfg.raw.get("evolution", {}),
                                    "live_shadow": {"num_challengers": num},
                                    "retire_min_trades": 2, "canary_min_trades": 2,
                                    "max_retire_per_cycle": 2}}
    return dataclasses.replace(cfg, paths=paths, raw=raw)


def _seed_challenger(reg: EvolutionRegistry, patch, metrics):
    c = reg.create_challenger(patch, source="mutation")
    c.status = SHADOW
    c.metrics = metrics
    reg.update_challenger(c)
    return c


def test_loser_retired_winner_kept_pool_refilled(cfg, tmp_path):
    c = _cfg(cfg, tmp_path, num=3)
    reg = EvolutionRegistry(c.path("reports_dir"))
    reg.ensure_champion()
    loser = _seed_challenger(reg, {"risk.take_profit_pct": 1.1},
                             {"num_trades": 5, "net_pnl_after_costs": -500.0,
                              "total_return_pct": -0.5, "win_rate_pct": 20.0, "equity": 99500.0})
    winner = _seed_challenger(reg, {"risk.take_profit_pct": 1.2},
                              {"num_trades": 5, "net_pnl_after_costs": 800.0,
                               "total_return_pct": 0.8, "win_rate_pct": 80.0, "equity": 100800.0})

    status = run_live_evolution_cycle(c, seed=1)

    assert loser.challenger_id in status["retired"]
    assert winner.challenger_id not in status["retired"]
    # pool refilled back to the configured size
    active = [x for x in reg.list_challengers() if x.status == SHADOW]
    assert len(active) >= 2
    assert winner.challenger_id in {x.challenger_id for x in reg.list_challengers()}

    db = EvolutionHistoryDB(c.path("reports_dir") / "evolution" / "history.db")
    assert db.get_dna(loser.challenger_id)["status"] == "REJECTED"
    assert db.track_history(winner.challenger_id)  # track record recorded
    assert status["spawned"]                       # at least one new DNA spawned
    # spawned DNA carries lineage back to the best survivor
    sp = db.get_dna(status["spawned"][0])
    assert sp["parent_id"] is not None
    db.close()


def test_no_retirement_without_enough_trades(cfg, tmp_path):
    c = _cfg(cfg, tmp_path, num=3)
    reg = EvolutionRegistry(c.path("reports_dir"))
    reg.ensure_champion()
    _seed_challenger(reg, {"risk.take_profit_pct": 1.1},
                     {"num_trades": 1, "net_pnl_after_costs": -500.0})

    status = run_live_evolution_cycle(c, seed=2)
    assert status["retired"] == []  # too few trades to judge — protected
