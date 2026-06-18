"""Champion / Challenger registry persistence.

Files under ``reports/registry/``:
    champion.yaml             current adopted config (Champion)
    previous_champions.jsonl  history of demoted/replaced champions (fallbacks)
    challengers.json          active challengers
    promotions.jsonl          promotion records
    rollbacks.jsonl           rollback records
    allocations.jsonl         allocation-update records
    registry_state.json       freeze-until and misc state
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .challenger import Challenger
from .champion import Champion


class EvolutionRegistry:
    def __init__(self, reports_dir: str | Path) -> None:
        self.dir = Path(reports_dir) / "registry"
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- champion ------------------------------------------------------------
    def champion_path(self) -> Path:
        return self.dir / "champion.yaml"

    def ensure_champion(self) -> Champion:
        if not self.champion_path().exists():
            champ = Champion.initial()
            self.save_champion(champ)
            return champ
        return self.load_champion()

    def load_champion(self) -> Champion:
        return Champion.from_dict(yaml.safe_load(self.champion_path().read_text()))

    def save_champion(self, champion: Champion) -> None:
        self.champion_path().write_text(yaml.safe_dump(champion.to_dict(), sort_keys=False))

    def push_previous_champion(self, champion: Champion) -> None:
        self._append("previous_champions.jsonl", champion.to_dict())

    def previous_champions(self) -> list[dict[str, Any]]:
        return self._read_jsonl("previous_champions.jsonl")

    def fallback_champion(self) -> dict[str, Any] | None:
        prev = self.previous_champions()
        return prev[-1] if prev else None

    # -- challengers ---------------------------------------------------------
    def list_challengers(self) -> list[Challenger]:
        path = self.dir / "challengers.json"
        if not path.exists():
            return []
        return [Challenger.from_dict(d) for d in json.loads(path.read_text())]

    def _save_challengers(self, challengers: list[Challenger]) -> None:
        path = self.dir / "challengers.json"
        path.write_text(json.dumps([c.to_dict() for c in challengers], indent=2, default=str))

    def create_challenger(self, config_patch: dict[str, Any], source: str = "mutation",
                          **kw) -> Challenger:
        chal = Challenger.create(config_patch, source=source, **kw)
        challengers = self.list_challengers()
        
        # チャレンジャーの最大数を5つに制限する
        MAX_CHALLENGERS = 5
        if len(challengers) >= MAX_CHALLENGERS:
            def sort_key(c: Challenger) -> tuple[int, float, str]:
                metrics = c.metrics or {}
                num_trades = metrics.get("num_trades", 0)
                total_return = metrics.get("total_return_pct", 0.0)
                # 取引実績があるものを優先削除対象とする (0), 未実績は保護 (1)
                trade_status = 0 if num_trades > 0 else 1
                # total_return（利益率）が低いほど先に削除されるようにする
                # 作成日時が古いほど先に削除されるようにする
                created_at = c.created_at or ""
                return (trade_status, total_return, created_at)
            
            challengers.sort(key=sort_key)
            # 5個枠に収まるように、最も不要なものを削除する
            num_to_delete = len(challengers) - MAX_CHALLENGERS + 1
            challengers = challengers[num_to_delete:]
            
        challengers.append(chal)
        self._save_challengers(challengers)
        return chal

    def get_challenger(self, challenger_id: str) -> Challenger | None:
        for c in self.list_challengers():
            if c.challenger_id == challenger_id:
                return c
        return None

    def update_challenger(self, challenger: Challenger) -> None:
        challengers = self.list_challengers()
        for i, c in enumerate(challengers):
            if c.challenger_id == challenger.challenger_id:
                challengers[i] = challenger
                break
        else:
            challengers.append(challenger)
        self._save_challengers(challengers)

    # -- records -------------------------------------------------------------
    def record_promotion(self, rec: dict[str, Any]) -> None:
        self._append("promotions.jsonl", {"timestamp": _now(), **rec})

    def record_rollback(self, rec: dict[str, Any]) -> None:
        self._append("rollbacks.jsonl", {"timestamp": _now(), **rec})

    def record_allocation(self, rec: dict[str, Any]) -> None:
        self._append("allocations.jsonl", {"timestamp": _now(), **rec})

    def promotions(self) -> list[dict[str, Any]]:
        return self._read_jsonl("promotions.jsonl")

    def rollbacks(self) -> list[dict[str, Any]]:
        return self._read_jsonl("rollbacks.jsonl")

    def allocations(self) -> list[dict[str, Any]]:
        return self._read_jsonl("allocations.jsonl")

    # -- promotion freeze ----------------------------------------------------
    def freeze_promotions(self, until: date) -> None:
        state = self._state()
        state["freeze_promotions_until"] = until.isoformat()
        self._write_json("registry_state.json", state)

    def is_frozen(self, on: date) -> bool:
        state = self._state()
        until = state.get("freeze_promotions_until")
        if not until:
            return False
        return on <= date.fromisoformat(until)

    def _state(self) -> dict[str, Any]:
        return self._read_json("registry_state.json") or {}

    # -- io ------------------------------------------------------------------
    def _append(self, name: str, obj: dict[str, Any]) -> None:
        with open(self.dir / name, "a") as fh:
            fh.write(json.dumps(obj, default=str) + "\n")

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.dir / name
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def _write_json(self, name: str, obj: Any) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=str))

    def _read_json(self, name: str) -> dict[str, Any] | None:
        path = self.dir / name
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
