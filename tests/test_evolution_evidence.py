from __future__ import annotations

from src.evolution.challenger import DRAFT, LIVE_ELIGIBLE, Challenger
from src.evolution.promotion_policy import evaluate_promotion, load_promotion_policy


def _champion():
    return {
        "win_rate_pct": 40.0,
        "expectancy": 0.0,
        "max_drawdown_pct": -10.0,
        "profit_factor": 1.0,
        "net_pnl_after_costs": 0.0,
        "worst_day_pct": -1.0,
    }


def _eligible():
    return {
        "win_rate_pct": 45.0,
        "expectancy": 10.0,
        "max_drawdown_pct": -8.0,
        "profit_factor": 1.5,
        "num_trades": 120,
        "net_pnl_after_costs": 500.0,
        "worst_day_pct": -1.0,
        "sharpe_ratio": 1.2,
        "paper_days": 190,
        "recent_3m_net_pnl": 100.0,
        "sealed_oos_pass": True,
        "forward_shadow_pass": True,
    }


def test_new_challenger_starts_as_draft():
    challenger = Challenger.create({}, source="manual")

    assert challenger.status == DRAFT
    assert challenger.trial_count == 0
    assert challenger.evidence == {}


def test_six_month_paper_evidence_is_required():
    metrics = {**_eligible(), "paper_days": 182}
    result = evaluate_promotion(
        _champion(),
        metrics,
        env="paper",
        days_shadow=30,
        days_canary=182,
        policy=load_promotion_policy(),
        robustness={"overfitting_risk": "LOW", "out_of_sample_pass": True},
    )

    assert not result.passed
    assert result.reasons["min_paper_days"] is False


def test_live_eligibility_requires_sharpe_recent_profit_and_forward_evidence():
    metrics = {
        **_eligible(),
        "sharpe_ratio": 0.9,
        "recent_3m_net_pnl": -1.0,
        "forward_shadow_pass": False,
    }
    result = evaluate_promotion(
        _champion(),
        metrics,
        env="paper",
        days_shadow=30,
        days_canary=190,
        policy=load_promotion_policy(),
        robustness={"overfitting_risk": "LOW", "out_of_sample_pass": True},
    )

    assert not result.passed
    assert result.reasons["min_sharpe"] is False
    assert result.reasons["recent_3m_positive"] is False
    assert result.reasons["forward_shadow_pass"] is False


def test_complete_evidence_can_be_marked_live_eligible_without_enabling_live():
    challenger = Challenger.create({}, source="manual")
    challenger.metrics = _eligible()
    challenger.status = LIVE_ELIGIBLE

    assert challenger.status == LIVE_ELIGIBLE
