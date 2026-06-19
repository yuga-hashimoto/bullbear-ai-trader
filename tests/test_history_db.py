"""SQLite history of DNA + track records."""
from __future__ import annotations

from src.evolution.history_db import EvolutionHistoryDB


def test_dna_upsert_and_retire(tmp_path):
    db = EvolutionHistoryDB(tmp_path / "h.db")
    db.upsert_dna("chal_a", {"risk.take_profit_pct": 1.2}, parent_id="champ",
                  source="mutation", status="SHADOW")
    assert db.get_dna("chal_a")["status"] == "SHADOW"
    assert db.get_dna("chal_a")["config_patch"] == {"risk.take_profit_pct": 1.2}
    assert db.get_dna("chal_a")["parent_id"] == "champ"

    db.retire_dna("chal_a", status="REJECTED", reason="losing")
    assert db.get_dna("chal_a")["status"] == "REJECTED"
    assert db.get_dna("chal_a")["retired_at"] is not None
    assert len(db.list_dna(status="REJECTED")) == 1
    db.close()


def test_track_record_history(tmp_path):
    db = EvolutionHistoryDB(tmp_path / "h.db")
    db.record_track("chal_a", {"num_trades": 3, "net_pnl_after_costs": 12.5,
                               "total_return_pct": 0.5, "win_rate_pct": 66.0,
                               "equity": 100012.5}, ts="2026-06-19T00:00:00Z")
    db.record_track("chal_a", {"num_trades": 4, "net_pnl_after_costs": -3.0,
                               "total_return_pct": -0.1, "win_rate_pct": 50.0,
                               "equity": 99997.0}, ts="2026-06-19T05:00:00Z")
    hist = db.track_history("chal_a")
    assert len(hist) == 2
    assert hist[0]["net_pnl"] == 12.5
    assert hist[1]["num_trades"] == 4
    db.close()


def test_champion_and_events(tmp_path):
    db = EvolutionHistoryDB(tmp_path / "h.db")
    db.record_champion("champ_1", {}, source="initial", metrics={"x": 1})
    db.record_event("CHALLENGER_RETIRED", {"challenger_id": "chal_a"})
    assert len(db.champion_history()) == 1
    assert db.events()[0]["event"] == "CHALLENGER_RETIRED"
    db.close()


def test_db_survives_reopen(tmp_path):
    path = tmp_path / "h.db"
    db = EvolutionHistoryDB(path)
    db.upsert_dna("chal_a", {"risk.take_profit_pct": 1.0}, status="SHADOW")
    db.close()
    again = EvolutionHistoryDB(path)
    assert again.get_dna("chal_a") is not None
    again.close()
