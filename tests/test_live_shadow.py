"""Live shadow challengers decide on the same bar as the champion."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.evolution.live_shadow import LiveShadowBook, ShadowChallenger
from src.features.builder import build_feature_matrix, prepare_feature_matrix

ET = ZoneInfo("America/New_York")


class _Session:
    """Minimal session double matching the attributes ShadowChallenger reads."""

    def __init__(self, close="16:00", since_open=120, to_close=120):
        self.close_dt = pd.Timestamp("2024-01-10 16:00", tz=ET)
        self.open_dt = pd.Timestamp("2024-01-10 09:30", tz=ET)
        self.is_early_close = False
        self._since = since_open
        self._to = to_close

    def minutes_since_open(self, now):
        return self._since

    def minutes_to_close(self, now):
        return self._to


def _cfg(cfg, tmp_path):
    paths = {**cfg.paths, "reports_dir": str(tmp_path / "reports")}
    return dataclasses.replace(cfg, paths=paths)


def _bull_context(ts: str) -> dict:
    # Fully trend-aligned bull bar so the V9 entry fires (QQQ -> TQQQ).
    return {
        "timestamp": ts,
        "symbols": {
            "QQQ": {"close": 101.0, "vwap": 100.0, "rsi": 58.0,
                    "returns": {"1_bar": 0.004, "3_bar": 0.01, "6_bar": 0.008, "12_bar": 0.006}},
            "SMH": {"close": None, "vwap": None, "rsi": None, "returns": {"3_bar": None}},
        },
    }


def _row(cfg, frames):
    matrix, _ = prepare_feature_matrix(build_feature_matrix(frames, cfg))
    return matrix.iloc[-1]


def test_book_creates_requested_number_of_challengers(cfg, frames, tmp_path):
    c = _cfg(cfg, tmp_path)
    book = LiveShadowBook(c, num_challengers=4, capital=100000.0)
    assert len(book.challengers) == 4
    # registry persisted them so the dashboard loader can read them
    reg = json.loads((tmp_path / "reports" / "registry" / "challengers.json").read_text())
    assert len([x for x in reg if x["status"] == "SHADOW"]) == 4


def test_challenger_trades_on_bullish_bar(cfg, frames, tmp_path):
    c = _cfg(cfg, tmp_path)
    book = LiveShadowBook(c, num_challengers=3, capital=100000.0)
    now = datetime(2024, 1, 10, 12, 0, tzinfo=ET)
    row = _row(c, frames)
    ctx = _bull_context(now.isoformat())

    book.on_bar(now, row, ctx, None, _Session())

    # At least one challenger took a virtual position on the bullish bar.
    assert any(sc.position is not None for sc in book.challengers)
    # Metrics + state were persisted for the dashboard / restart.
    challengers = json.loads(
        (tmp_path / "reports" / "registry" / "challengers.json").read_text())
    assert all("total_return_pct" in x["metrics"] for x in challengers)
    state = json.loads((tmp_path / "reports" / "registry" / "shadow_state.json").read_text())
    assert set(state) == {sc.challenger_id for sc in book.challengers}


def test_shadow_state_survives_restart(cfg, frames, tmp_path):
    c = _cfg(cfg, tmp_path)
    book = LiveShadowBook(c, num_challengers=2, capital=100000.0)
    now = datetime(2024, 1, 10, 12, 0, tzinfo=ET)
    row = _row(c, frames)
    book.on_bar(now, row, _bull_context(now.isoformat()), None, _Session())
    seqs = {sc.challenger_id: sc.trade_seq for sc in book.challengers}

    reborn = LiveShadowBook(c, num_challengers=2, capital=100000.0)
    for sc in reborn.challengers:
        assert sc.trade_seq == seqs[sc.challenger_id]


def test_one_failing_challenger_does_not_break_the_book(cfg, frames, tmp_path):
    c = _cfg(cfg, tmp_path)
    book = LiveShadowBook(c, num_challengers=2, capital=100000.0)
    now = datetime(2024, 1, 10, 12, 0, tzinfo=ET)
    row = _row(c, frames)

    class _Boom(ShadowChallenger):
        def on_bar(self, *a, **k):
            raise RuntimeError("boom")

    book.challengers[0].__class__ = _Boom  # force the first to explode
    # Must not raise — the book isolates per-challenger failures.
    book.on_bar(now, row, _bull_context(now.isoformat()), None, _Session())
