"""Capital / policy allocator.

Combines the Champion signal with Challenger signals into a single set of final
order intents (so a single live/paper account never receives conflicting
orders). Only the Champion drives the real (paper) account; challengers are
evaluated in shadow. Same-symbol opposite-direction challenger signals are
recorded as rejected conflicts. When uncertain, the result is NO_TRADE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as(sig: Any, attr: str):
    return sig.get(attr) if isinstance(sig, dict) else getattr(sig, attr, None)


@dataclass
class AllocatorResult:
    final_order_intents: list[dict[str, Any]] = field(default_factory=list)
    shadow_order_intents: list[dict[str, Any]] = field(default_factory=list)
    rejected_conflicts: list[dict[str, Any]] = field(default_factory=list)


def _actionable(sig: Any) -> bool:
    return _as(sig, "action") in ("BUY_BULL", "BUY_BEAR", "EXIT")


class Allocator:
    def combine(self, champion_signal: Any, challenger_signals: dict[str, Any],
                allocations: dict[str, float] | None = None) -> AllocatorResult:
        res = AllocatorResult()
        champ_sym = _as(champion_signal, "symbol")
        champ_dir = _as(champion_signal, "direction")

        if _actionable(champion_signal):
            res.final_order_intents.append({
                "source": "champion", "symbol": champ_sym,
                "action": _as(champion_signal, "action"), "direction": champ_dir,
            })

        for cid, sig in (challenger_signals or {}).items():
            if not _actionable(sig):
                continue
            entry = {"challenger_id": cid, "symbol": _as(sig, "symbol"),
                     "action": _as(sig, "action"), "direction": _as(sig, "direction"),
                     "allocation_pct": (allocations or {}).get(cid, 0.0)}
            # Conflict: same symbol as champion's live order but opposite direction.
            if champ_sym and entry["symbol"] == champ_sym and entry["direction"] != champ_dir:
                res.rejected_conflicts.append({**entry, "reason": "conflicts_with_champion"})
            res.shadow_order_intents.append(entry)
        return res
