"""SQLite history of challenger DNA (config patches) and track records.

Everything the evolution loop decides is recorded here so the lineage of
"surviving DNA" and each candidate's trading track record are queryable and
durable (survives restarts, independent of the JSONL event logs):

- ``dna``             one row per challenger: its config patch (the DNA), parent
                      lineage, source, lifecycle status, and retirement time.
- ``track_record``    time series of each challenger's live shadow performance.
- ``champion_history``every champion adopted, with its DNA and metrics.
- ``events``          promotions, retirements, spawns (audit trail).

Stdlib ``sqlite3`` only — no external dependency.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dna (
    challenger_id TEXT PRIMARY KEY,
    parent_id     TEXT,
    source        TEXT,
    config_patch  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    status        TEXT NOT NULL,
    retired_at    TEXT,
    notes         TEXT
);
CREATE TABLE IF NOT EXISTS track_record (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    challenger_id    TEXT NOT NULL,
    ts               TEXT NOT NULL,
    num_trades       INTEGER,
    net_pnl          REAL,
    total_return_pct REAL,
    win_rate_pct     REAL,
    equity           REAL,
    metrics          TEXT
);
CREATE INDEX IF NOT EXISTS idx_track_challenger ON track_record(challenger_id);
CREATE TABLE IF NOT EXISTS champion_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    champion_id  TEXT NOT NULL,
    config_patch TEXT NOT NULL,
    promoted_at  TEXT NOT NULL,
    source       TEXT,
    metrics      TEXT,
    notes        TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     TEXT NOT NULL,
    event  TEXT NOT NULL,
    detail TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvolutionHistoryDB:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EvolutionHistoryDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ DNA
    def upsert_dna(self, challenger_id: str, config_patch: dict[str, Any], *,
                   parent_id: str | None = None, source: str = "mutation",
                   status: str = "SHADOW", created_at: str | None = None,
                   notes: str = "") -> None:
        """Insert a new DNA row, or update mutable fields if it already exists."""
        existing = self._conn.execute(
            "SELECT challenger_id FROM dna WHERE challenger_id = ?", (challenger_id,)
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO dna (challenger_id, parent_id, source, config_patch, "
                "created_at, status, retired_at, notes) VALUES (?,?,?,?,?,?,?,?)",
                (challenger_id, parent_id, source, json.dumps(config_patch),
                 created_at or _now(), status, None, notes),
            )
        else:
            self._conn.execute(
                "UPDATE dna SET status = ?, notes = ? WHERE challenger_id = ?",
                (status, notes, challenger_id),
            )
        self._conn.commit()

    def retire_dna(self, challenger_id: str, *, status: str = "REJECTED",
                   reason: str = "") -> None:
        self._conn.execute(
            "UPDATE dna SET status = ?, retired_at = ?, notes = ? WHERE challenger_id = ?",
            (status, _now(), reason, challenger_id),
        )
        self._conn.commit()

    def get_dna(self, challenger_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM dna WHERE challenger_id = ?", (challenger_id,)
        ).fetchone()
        return self._dna_row(row) if row else None

    def list_dna(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM dna ORDER BY created_at").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM dna WHERE status = ? ORDER BY created_at", (status,)
            ).fetchall()
        return [self._dna_row(r) for r in rows]

    @staticmethod
    def _dna_row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["config_patch"] = json.loads(d["config_patch"]) if d["config_patch"] else {}
        return d

    # --------------------------------------------------------- track record
    def record_track(self, challenger_id: str, metrics: dict[str, Any],
                     ts: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO track_record (challenger_id, ts, num_trades, net_pnl, "
            "total_return_pct, win_rate_pct, equity, metrics) VALUES (?,?,?,?,?,?,?,?)",
            (challenger_id, ts or _now(), int(metrics.get("num_trades", 0)),
             float(metrics.get("net_pnl_after_costs", 0.0)),
             float(metrics.get("total_return_pct", 0.0)),
             float(metrics.get("win_rate_pct", 0.0)),
             float(metrics.get("equity", 0.0)), json.dumps(metrics)),
        )
        self._conn.commit()

    def track_history(self, challenger_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM track_record WHERE challenger_id = ? ORDER BY ts",
            (challenger_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------- champions
    def record_champion(self, champion_id: str, config_patch: dict[str, Any], *,
                        source: str = "", metrics: dict[str, Any] | None = None,
                        notes: str = "", promoted_at: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO champion_history (champion_id, config_patch, promoted_at, "
            "source, metrics, notes) VALUES (?,?,?,?,?,?)",
            (champion_id, json.dumps(config_patch), promoted_at or _now(), source,
             json.dumps(metrics or {}), notes),
        )
        self._conn.commit()

    def champion_history(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM champion_history ORDER BY promoted_at"
        ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- events
    def record_event(self, event: str, detail: dict[str, Any] | None = None) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, event, detail) VALUES (?,?,?)",
            (_now(), event, json.dumps(detail or {})),
        )
        self._conn.commit()

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
