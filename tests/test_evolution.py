"""Champion/Challenger, shadow, promotion, rollback, bandit, mutation, safety."""
from __future__ import annotations

import dataclasses

import pytest

from src.evolution import guardrails as gr
from src.evolution.bandit import ArmStats, Bandit
from src.evolution.canary_runner import promote_to_canary
from src.evolution.challenger import CANARY, DRAFT, LIVE_ELIGIBLE, PROMOTED, SHADOW
from src.evolution.champion import PatchError, apply_patch
from src.evolution.experiment_store import ExperimentStore
from src.evolution.loop import promote_challenger, rollback_to_fallback
from src.evolution.mutation_generator import generate_mutations
from src.evolution.promotion_policy import evaluate_promotion, load_promotion_policy
from src.evolution.registry import EvolutionRegistry
from src.evolution.rollback_policy import evaluate_rollback, load_rollback_policy
from src.evolution.shadow_runner import run_shadow

POLICY = load_promotion_policy()


def _cfg_window(cfg):
    """Align test window to the in-memory labeled_matrix fixture (Jan 2-12)."""
    return dataclasses.replace(cfg, test_start="2024-01-02", test_end="2024-01-12")


# ---------------------------------------------------------- registry
def test_champion_yaml_created(cfg, tmp_path):
    reg = EvolutionRegistry(tmp_path)
    reg.ensure_champion()
    assert (tmp_path / "registry" / "champion.yaml").exists()


def test_create_challenger_is_draft(cfg, tmp_path):
    reg = EvolutionRegistry(tmp_path)
    reg.ensure_champion()
    chal = reg.create_challenger({"risk.confidence_threshold": 0.7}, source="manual")
    assert chal.status == DRAFT
    assert reg.list_challengers()[0].challenger_id == chal.challenger_id


def test_shadow_result_saved(cfg, labeled_matrix, tmp_path):
    cfg2 = _cfg_window(cfg)
    reg = EvolutionRegistry(tmp_path)
    reg.ensure_champion()
    chal = reg.create_challenger({"risk.confidence_threshold": 0.55}, source="manual")
    chal.status = SHADOW
    reg.update_challenger(chal)
    store = ExperimentStore(tmp_path)
    res = run_shadow(cfg2, reg, store, agent_type="mock", matrix=labeled_matrix, with_robustness=False)
    assert res["challengers"], "expected challenger shadow metrics"
    assert (tmp_path / "evolution" / "shadow_pnl.jsonl").exists()
    assert "num_trades" in reg.list_challengers()[0].metrics


def test_canary_promotion(cfg, tmp_path):
    reg = EvolutionRegistry(tmp_path)
    reg.ensure_champion()
    chal = reg.create_challenger({"risk.confidence_threshold": 0.6}, source="manual")
    chal.status = SHADOW
    reg.update_challenger(chal)
    store = ExperimentStore(tmp_path)
    assert promote_to_canary(reg, store, chal.challenger_id, allocation_pct=10)
    assert reg.get_challenger(chal.challenger_id).status == CANARY


# ---------------------------------------------------------- promotion
def _champ():
    return {"win_rate_pct": 40.0, "expectancy": 0.0, "max_drawdown_pct": -10.0,
            "profit_factor": 1.0, "net_pnl_after_costs": 0.0, "worst_day_pct": -1.0}


def _good_challenger():
    return {"win_rate_pct": 45.0, "expectancy": 0.01, "max_drawdown_pct": -8.0,
            "profit_factor": 1.5, "num_trades": 120, "net_pnl_after_costs": 500.0,
            "worst_day_pct": -1.0, "sharpe_ratio": 1.2, "paper_days": 190,
            "recent_3m_net_pnl": 100.0, "sealed_oos_pass": True,
            "forward_shadow_pass": True}


def _good_rob():
    return {"overfitting_risk": "LOW", "out_of_sample_pass": True}


def _eval(champ, chal, env="paper", ds=10, dc=190, rob=None):
    return evaluate_promotion(champ, chal, env=env, days_shadow=ds, days_canary=dc,
                              policy=POLICY, robustness=rob or _good_rob())


def test_promotion_unmet_when_weak():
    weak = {**_good_challenger(), "profit_factor": 0.8, "expectancy": -0.01}
    assert not _eval(_champ(), weak).passed


def test_high_winrate_bad_drawdown_not_promoted():
    chal = {**_good_challenger(), "win_rate_pct": 70.0, "max_drawdown_pct": -25.0}
    r = _eval(_champ(), chal)
    assert not r.passed and r.reasons["drawdown_not_worse"] is False


def test_low_profit_factor_not_promoted():
    chal = {**_good_challenger(), "profit_factor": 1.0}
    assert not _eval(_champ(), chal).passed


def test_insufficient_trades_not_promoted():
    chal = {**_good_challenger(), "num_trades": 10}
    r = _eval(_champ(), chal)
    assert not r.passed and r.reasons["min_trades"] is False


def test_robustness_fail_not_promoted():
    r = _eval(_champ(), _good_challenger(), rob={"overfitting_risk": "HIGH", "out_of_sample_pass": False})
    assert not r.passed and r.reasons["robustness_ok"] is False


def test_conditions_met_auto_promote_paper(cfg, tmp_path):
    r = _eval(_champ(), _good_challenger(), env="paper")
    assert r.passed
    # mechanics: promote swaps champion and records previous.
    reg = EvolutionRegistry(tmp_path)
    reg.ensure_champion()
    chal = reg.create_challenger({"risk.confidence_threshold": 0.7}, source="manual")
    chal.metrics = _good_challenger()
    chal.status = LIVE_ELIGIBLE
    reg.update_challenger(chal)
    store = ExperimentStore(tmp_path)
    assert promote_challenger(cfg, reg, store, chal, "paper", r.reasons)
    assert reg.load_champion().champion_id == chal.challenger_id
    assert reg.previous_champions()           # fallback recorded
    assert reg.get_challenger(chal.challenger_id).status == PROMOTED


def test_live_env_not_auto_promoted_by_default():
    r = _eval(_champ(), _good_challenger(), env="live")
    assert not r.passed and r.reasons["environment_allowed"] is False


# ---------------------------------------------------------- rollback
ROLL = load_rollback_policy()


def test_rollback_on_max_daily_loss():
    assert "max_daily_loss" in evaluate_rollback({"daily_loss_pct": 3.0}, ROLL).reasons


def test_rollback_on_consecutive_losses():
    assert "max_consecutive_losses" in evaluate_rollback({"consecutive_losses": 5}, ROLL).reasons


def test_rollback_on_underperformance():
    assert "underperformance_vs_fallback" in evaluate_rollback(
        {"underperformance_vs_fallback_pct": 2.0}, ROLL).reasons


def test_rollback_records_and_freezes(cfg, tmp_path):
    from datetime import date
    reg = EvolutionRegistry(tmp_path)
    reg.ensure_champion()
    # need a fallback to roll back to.
    from src.evolution.champion import Champion
    reg.push_previous_champion(Champion.initial())
    store = ExperimentStore(tmp_path)
    assert rollback_to_fallback(reg, store, ["max_daily_loss"], on_day=date(2024, 1, 10), freeze_days=3)
    assert reg.rollbacks()                       # event recorded
    assert reg.is_frozen(date(2024, 1, 11))      # promotions frozen


# ---------------------------------------------------------- bandit
def test_better_challenger_gets_more_allocation():
    b = Bandit(min_trades=10, max_challenger_allocation_pct=30)
    alloc = b.allocate([ArmStats("A", reward=2.0, trades=50, drawdown_pct=-5),
                        ArmStats("B", reward=1.0, trades=50, drawdown_pct=-5)])
    assert alloc["A"] > alloc["B"]


def test_high_drawdown_gets_less_allocation():
    b = Bandit(min_trades=10, max_challenger_allocation_pct=30)
    alloc = b.allocate([ArmStats("A", reward=1.0, trades=50, drawdown_pct=-1),
                        ArmStats("B", reward=1.0, trades=50, drawdown_pct=-15)])
    assert alloc["A"] > alloc["B"]


def test_allocation_not_exceed_max():
    b = Bandit(min_trades=10, max_challenger_allocation_pct=30)
    alloc = b.allocate([ArmStats("A", 5.0, 50, -2), ArmStats("B", 3.0, 50, -2)])
    assert sum(alloc.values()) <= 30 + 1e-6


def test_below_min_trades_not_allocated():
    b = Bandit(min_trades=30, max_challenger_allocation_pct=30)
    alloc = b.allocate([ArmStats("A", 99.0, 5, -1)])  # great reward but too few trades
    assert alloc["A"] == 0.0


# ---------------------------------------------------------- mutation / guardrails
def test_mutation_candidate_generated(cfg, tmp_path):
    cands = generate_mutations(cfg, n=2, seed=1, candidates_dir=tmp_path / "cands")
    assert cands
    cid = cands[0]["candidate_id"]
    assert (tmp_path / "cands" / cid / "candidate.yaml").exists()
    assert gr.is_safe(cands[0]["config_patch"], cfg)


def test_guardrail_rejects_risk_weakening(cfg):
    assert gr.check_patch({"risk.max_daily_loss_pct": 10.0}, cfg)


def test_guardrail_rejects_cost_lowering(cfg):
    assert gr.check_patch({"costs.slippage_pct": 0.0}, cfg)


def test_guardrail_rejects_live_gate_weakening(cfg):
    assert gr.check_patch({"trading.live_trading_enabled": True}, cfg)


# ---------------------------------------------------------- safety
def test_apply_patch_cannot_touch_costs_or_trading(cfg):
    with pytest.raises(PatchError):
        apply_patch(cfg, {"costs.slippage_pct": 0.0})
    with pytest.raises(PatchError):
        apply_patch(cfg, {"trading.live_trading_enabled": True})


def test_promotion_refuses_unsafe_patch(cfg, tmp_path):
    reg = EvolutionRegistry(tmp_path)
    reg.ensure_champion()
    before = reg.load_champion().champion_id
    chal = reg.create_challenger({"risk.max_daily_loss_pct": 99.0}, source="manual")
    chal.status = LIVE_ELIGIBLE
    chal.metrics = _good_challenger()
    reg.update_challenger(chal)
    store = ExperimentStore(tmp_path)
    assert not promote_challenger(cfg, reg, store, chal, "paper", {})
    assert reg.load_champion().champion_id == before   # champion unchanged


def test_load_evolution_empty_does_not_crash(tmp_path):
    from src.reports.loader import load_evolution
    evo = load_evolution(tmp_path)
    assert evo["champion"] == {}
    assert evo["challengers"] == []
    assert evo["promotions"].empty
