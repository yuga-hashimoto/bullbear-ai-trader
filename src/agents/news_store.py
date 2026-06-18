"""Persistent deduplication and latest-analysis state for news processing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class NewsStateStore:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "news_state.json"
        self._state = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"seen_news_ids": [], "latest_analysis": None}
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"seen_news_ids": [], "latest_analysis": None}
        return {
            "seen_news_ids": list(data.get("seen_news_ids", [])),
            "latest_analysis": data.get("latest_analysis"),
        }

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def is_seen(self, news_id: str) -> bool:
        return news_id in set(self._state["seen_news_ids"])

    def mark_seen(self, news_ids: Iterable[str]) -> None:
        seen = set(self._state["seen_news_ids"])
        seen.update(str(x) for x in news_ids)
        self._state["seen_news_ids"] = sorted(seen)
        self._save()

    def save_analysis(self, analysis: dict) -> None:
        self._state["latest_analysis"] = analysis
        self._save()

    @property
    def latest_analysis(self) -> dict | None:
        return self._state.get("latest_analysis")
