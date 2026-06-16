"""Adaptive allocation across challengers (multi-armed bandit).

Default: a deterministic epsilon-greedy blend (exploration + exploitation) so
allocations are reproducible. Allocation rises with reward and falls with
drawdown; challengers below ``min_trades`` are not over-allocated; the total
challenger allocation never exceeds ``max_challenger_allocation_pct`` (the rest
stays with the Champion). On live, reallocation is disabled by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

_DRAWDOWN_CAP = 20.0  # |drawdown%| at which the allocation multiplier hits ~0


@dataclass(frozen=True)
class ArmStats:
    arm_id: str
    reward: float          # higher is better (e.g. risk-adjusted expectancy)
    trades: int
    drawdown_pct: float    # negative; more negative is worse


class Bandit:
    def __init__(self, mode: str = "epsilon_greedy", epsilon: float = 0.1,
                 max_challenger_allocation_pct: float = 30.0,
                 min_allocation_pct: float = 0.0, min_trades: int = 30) -> None:
        self.mode = mode
        self.epsilon = epsilon
        self.max_total = max_challenger_allocation_pct
        self.min_alloc = min_allocation_pct
        self.min_trades = min_trades

    def _score(self, arm: ArmStats) -> float:
        base = max(arm.reward, 0.0)
        dd_mult = max(0.0, 1.0 - abs(arm.drawdown_pct) / _DRAWDOWN_CAP)
        return base * dd_mult

    def allocate(self, arms: list[ArmStats]) -> dict[str, float]:
        """Return {challenger_id: allocation_pct}; champion takes the remainder."""
        if not arms:
            return {}
        eligible = [a for a in arms if a.trades >= self.min_trades]
        alloc = {a.arm_id: self.min_alloc for a in arms}
        if not eligible:
            return alloc

        scores = {a.arm_id: self._score(a) for a in eligible}
        total_score = sum(scores.values())
        n = len(eligible)
        for a in eligible:
            exploit = (scores[a.arm_id] / total_score) if total_score > 0 else (1.0 / n)
            explore = 1.0 / n
            share = (1.0 - self.epsilon) * exploit + self.epsilon * explore
            alloc[a.arm_id] = round(self.max_total * share, 4)
        return alloc

    @staticmethod
    def champion_allocation(challenger_alloc: dict[str, float]) -> float:
        return round(max(0.0, 100.0 - sum(challenger_alloc.values())), 4)
