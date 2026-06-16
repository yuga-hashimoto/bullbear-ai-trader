"""Canary promotion: graduate a strong shadow challenger to a small allocation.

Canary == small (virtual on paper) allocation under near-real conditions, still
fully gated by the Risk Engine. This module only manages the SHADOW -> CANARY
transition and allocation bookkeeping; capital impact is handled by the
Allocator (and, on paper, remains virtual).
"""
from __future__ import annotations

from .challenger import CANARY, SHADOW
from .experiment_store import EvolutionEventType, ExperimentStore
from .registry import EvolutionRegistry


def promote_to_canary(
    registry: EvolutionRegistry,
    store: ExperimentStore,
    challenger_id: str,
    allocation_pct: float = 10.0,
) -> bool:
    chal = registry.get_challenger(challenger_id)
    if chal is None or chal.status != SHADOW:
        return False
    chal.status = CANARY
    chal.allocation_pct = allocation_pct
    registry.update_challenger(chal)
    store.emit(EvolutionEventType.CHALLENGER_CANARY_STARTED,
               {"challenger_id": challenger_id, "allocation_pct": allocation_pct})
    return True
