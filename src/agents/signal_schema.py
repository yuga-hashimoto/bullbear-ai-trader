"""Standard trade-signal schema + validation.

The agent (OpenClaw / HermesAgent / Mock / Replay / LocalModel) communicates
ONLY through this normalized JSON. Natural-language agent responses are never
used to drive trades — only a validated :class:`Signal` is. Validation is the
first gate; the Risk Engine is the authoritative second gate.

Uses dataclass + explicit validation (no extra runtime dependency).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --- allowed enums ----------------------------------------------------------
TARGET_FAMILIES = {"NASDAQ", "SEMICONDUCTOR", "MARKET"}
DIRECTIONS = {"UP", "DOWN", "FLAT"}
ACTIONS = {"BUY_BULL", "BUY_BEAR", "NO_TRADE", "EXIT"}

BULL_SYMBOLS = {"TQQQ", "SOXL"}
BEAR_SYMBOLS = {"SQQQ", "SOXS"}
TRADABLE_SYMBOLS = BULL_SYMBOLS | BEAR_SYMBOLS

# Which tradable symbol a (family, side) pair maps to.
FAMILY_BULL = {"NASDAQ": "TQQQ", "SEMICONDUCTOR": "SOXL"}
FAMILY_BEAR = {"NASDAQ": "SQQQ", "SEMICONDUCTOR": "SOXS"}


class SignalValidationError(ValueError):
    """Raised when a signal violates the schema or business rules."""


@dataclass(frozen=True)
class Signal:
    timestamp: str
    agent_name: str
    target_family: str
    direction: str
    action: str
    symbol: str | None = None
    confidence: float = 0.0
    agent_version: str = ""
    expected_holding_minutes: int = 0
    reason: str = ""
    risk_notes: list[str] = field(default_factory=list)
    features_used: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Signal":
        """Build a Signal from a raw dict, tolerating missing optional keys.

        Required keys: timestamp, agent_name, target_family, direction, action.
        Raises :class:`SignalValidationError` if a required key is absent or the
        payload is not a mapping.
        """
        if not isinstance(data, dict):
            raise SignalValidationError(f"signal must be a mapping, got {type(data)!r}")
        required = ("timestamp", "agent_name", "target_family", "direction", "action")
        missing = [k for k in required if k not in data]
        if missing:
            raise SignalValidationError(f"missing required fields: {missing}")
        sym = data.get("symbol")
        if sym is not None:
            sym = str(sym).upper()
        try:
            return cls(
                timestamp=str(data["timestamp"]),
                agent_name=str(data["agent_name"]),
                target_family=str(data["target_family"]).upper(),
                direction=str(data["direction"]).upper(),
                action=str(data["action"]).upper(),
                symbol=sym,
                confidence=float(data.get("confidence", 0.0)),
                agent_version=str(data.get("agent_version", "")),
                expected_holding_minutes=int(data.get("expected_holding_minutes", 0)),
                reason=str(data.get("reason", "")),
                risk_notes=list(data.get("risk_notes", []) or []),
                features_used=dict(data.get("features_used", {}) or {}),
                raw_response=dict(data.get("raw_response", {}) or {}),
            )
        except (TypeError, ValueError) as exc:
            raise SignalValidationError(f"malformed signal field: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- validation ----------------------------------------------------------
    def validate(self) -> "Signal":
        """Structural + consistency validation. Returns self or raises.

        This guards the *shape* of the signal. Business gates (confidence
        threshold, allowed-symbol policy) live in the Risk Engine.
        """
        if self.target_family not in TARGET_FAMILIES:
            raise SignalValidationError(f"invalid target_family: {self.target_family!r}")
        if self.direction not in DIRECTIONS:
            raise SignalValidationError(f"invalid direction: {self.direction!r}")
        if self.action not in ACTIONS:
            raise SignalValidationError(f"invalid action: {self.action!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise SignalValidationError(f"confidence out of [0,1]: {self.confidence}")
        if self.expected_holding_minutes < 0:
            raise SignalValidationError("expected_holding_minutes must be >= 0")
        if not isinstance(self.risk_notes, list):
            raise SignalValidationError("risk_notes must be a list")

        if self.action == "BUY_BULL":
            self._require_buy_symbol(FAMILY_BULL, BULL_SYMBOLS, "bull")
        elif self.action == "BUY_BEAR":
            self._require_buy_symbol(FAMILY_BEAR, BEAR_SYMBOLS, "bear")
        elif self.action == "NO_TRADE":
            if self.symbol is not None:
                raise SignalValidationError("NO_TRADE must not carry a symbol")
        elif self.action == "EXIT":
            if self.symbol is not None and self.symbol not in TRADABLE_SYMBOLS:
                raise SignalValidationError(f"EXIT symbol not tradable: {self.symbol!r}")
        return self

    def _require_buy_symbol(self, family_map: dict, side_set: set, side: str) -> None:
        if self.target_family not in family_map:
            raise SignalValidationError(
                f"{self.action} requires NASDAQ or SEMICONDUCTOR family, "
                f"got {self.target_family!r}"
            )
        expected = family_map[self.target_family]
        if self.symbol is None:
            raise SignalValidationError(f"{self.action} requires a {side} symbol")
        if self.symbol not in side_set:
            raise SignalValidationError(
                f"{self.action} symbol must be one of {sorted(side_set)}, got {self.symbol!r}"
            )
        if self.symbol != expected:
            raise SignalValidationError(
                f"{self.target_family} {self.action} must use {expected}, got {self.symbol!r}"
            )


def no_trade_signal(timestamp: str, agent_name: str, reason: str = "") -> Signal:
    """Convenience constructor for the safe default (NO_TRADE, FLAT)."""
    return Signal(
        timestamp=timestamp,
        agent_name=agent_name,
        target_family="MARKET",
        direction="FLAT",
        action="NO_TRADE",
        symbol=None,
        confidence=0.0,
        reason=reason,
    )
