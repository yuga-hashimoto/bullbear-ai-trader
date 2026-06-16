"""Champion model + config-patch application.

The Champion is the currently adopted configuration. PaperRunner (and a future
LiveRunner) use the Champion by default. A challenger's ``config_patch`` is
applied on top of the base config to derive the config used for evaluation.

``apply_patch`` only touches whitelisted, safety-neutral sections. The
guardrails module is the authority on what a patch may contain; apply_patch
additionally refuses unknown sections so a patch can never reach, say, the live
gate or the cost model.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config.settings import Config

# Sections a challenger patch may modify (safety-neutral knobs only).
PATCHABLE_SECTIONS = {"risk", "strategy", "agent"}


@dataclass
class Champion:
    champion_id: str
    created_at: str
    config_patch: dict[str, Any] = field(default_factory=dict)
    agent_prompt_version: str = ""
    risk_policy_version: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = "initial champion"

    @classmethod
    def initial(cls) -> "Champion":
        return cls(
            champion_id="champion_initial",
            created_at=datetime.now(timezone.utc).isoformat(),
            config_patch={},
            notes="initial champion (base config)",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Champion":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


class PatchError(ValueError):
    """Raised when a patch targets a forbidden/unknown config section."""


def apply_patch(cfg: Config, patch: dict[str, Any]) -> Config:
    """Return a new Config with ``patch`` (dot-keyed) applied.

    Only ``risk.*``, ``strategy.*`` and ``agent.*`` may be patched. Any other
    section (costs, trading, market, runner, ...) is rejected — a challenger can
    never weaken costs or the live gate through a patch.
    """
    if not patch:
        return cfg
    grouped: dict[str, dict[str, Any]] = {}
    for dotted, value in patch.items():
        if "." not in dotted:
            raise PatchError(f"patch key must be 'section.field': {dotted!r}")
        section, field_name = dotted.split(".", 1)
        if section not in PATCHABLE_SECTIONS:
            raise PatchError(f"section not patchable: {section!r}")
        grouped.setdefault(section, {})[field_name] = value

    replacements: dict[str, Any] = {}
    for section, fields in grouped.items():
        current = getattr(cfg, section)
        valid = set(current.__dataclass_fields__)
        bad = set(fields) - valid
        if bad:
            raise PatchError(f"unknown {section} fields: {sorted(bad)}")
        replacements[section] = dataclasses.replace(current, **fields)
    return dataclasses.replace(cfg, **replacements)
