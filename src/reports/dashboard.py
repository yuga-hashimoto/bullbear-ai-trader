"""Streamlit dashboard for backtest results (READ-ONLY).

Launch:
    streamlit run src/reports/dashboard.py
Optionally point at a reports dir:
    BULLBEAR_REPORTS_DIR=reports streamlit run src/reports/dashboard.py

This UI is for VIEWING backtest results only. It deliberately contains NO order
buttons, NO live-trading toggle and NO broker controls. It cannot place trades.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `streamlit run src/reports/dashboard.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.reports.loader import (  # noqa: E402
    DEFAULT_REPORTS_DIR,
    RunData,
    list_run_ids,
    load_evolution,
    load_run,
    load_runtime,
)

st.set_page_config(page_title="bullbear backtest dashboard", layout="wide")

SAFETY = (
    "🔒 Read-only backtest viewer. No live trading, no orders. "
    "Not investment advice."
)


def _reports_dir() -> str:
    return os.environ.get("BULLBEAR_REPORTS_DIR", DEFAULT_REPORTS_DIR)


def _confidence_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if "confidence" not in df.columns or df.empty:
        return df
    lo, hi = st.slider("confidence range", 0.0, 1.0, (0.0, 1.0), 0.01, key=key)
    return df[(df["confidence"].fillna(0) >= lo) & (df["confidence"].fillna(0) <= hi)]


def _multiselect_filter(df: pd.DataFrame, col: str, key: str) -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return df
    options = sorted(x for x in df[col].dropna().unique())
    if not options:
        return df
    chosen = st.multiselect(col, options, default=options, key=key)
    return df[df[col].isin(chosen)] if chosen else df


def overview_tab(run: RunData) -> None:
    s, m, c = run.summary, run.metrics, run.counters
    st.subheader(f"Run: {run.run_id}")
    meta_cols = st.columns(4)
    meta_cols[0].write(f"**Created:** {s.get('created_at', '?')}")
    meta_cols[1].write(f"**Symbols:** {', '.join(s.get('symbols', []))}")
    period = s.get("period", {})
    meta_cols[2].write(f"**Period:** {period.get('start','?')} → {period.get('end','?')}")
    meta_cols[3].write(f"**Interval / Agent:** {s.get('interval','?')} / {s.get('agent_type','?')}")

    grid = st.columns(4)
    grid[0].metric("Total return %", m.get("total_return_pct", "—"))
    grid[1].metric("Max drawdown %", m.get("max_drawdown_pct", "—"))
    grid[2].metric("Win rate %", m.get("win_rate_pct", "—"))
    grid[3].metric("Profit factor", m.get("profit_factor", "—"))
    grid2 = st.columns(4)
    grid2[0].metric("# trades", m.get("num_trades", "—"))
    grid2[1].metric("No-trade ratio", m.get("no_trade_ratio", "—"))
    grid2[2].metric("Rejected signals", m.get("rejected_signals", "—"))
    grid2[3].metric("Forced exits", m.get("forced_exits", "—"))

    with st.expander("All metrics"):
        st.json(m)
    with st.expander("Benchmark comparison"):
        st.json(run.benchmark)


def charts_tab(run: RunData) -> None:
    eq = run.equity
    if not eq.empty and "equity" in eq.columns:
        eq = eq.copy()
        # utc=True avoids "mixed timezones" when the series spans a DST change.
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
        eq = eq.set_index("timestamp")
        st.markdown("**Equity curve**")
        st.line_chart(eq["equity"])
        running_max = eq["equity"].cummax()
        drawdown = (eq["equity"] - running_max) / running_max * 100.0
        st.markdown("**Drawdown curve (%)**")
        st.area_chart(drawdown)
    else:
        st.info("No equity curve (no bars).")

    dp = run.daily_pnl
    if not dp.empty and "pnl" in dp.columns:
        dp = dp.copy()
        st.markdown("**Daily PnL**")
        st.bar_chart(dp.set_index("date")["pnl"])
        st.markdown("**Cumulative PnL**")
        st.line_chart(dp["pnl"].cumsum())

    trades = run.trades
    if not trades.empty and "symbol" in trades.columns:
        st.markdown("**PnL by symbol**")
        st.bar_chart(trades.groupby("symbol")["net_pnl"].sum())

    action_dist = run.counters.get("action_distribution", {})
    if action_dist:
        st.markdown("**Agent action counts**")
        st.bar_chart(pd.Series(action_dist))

    sig = run.agent_signals
    if not sig.empty and "confidence" in sig.columns:
        st.markdown("**Agent confidence distribution**")
        conf = pd.to_numeric(sig["confidence"], errors="coerce").dropna()
        if not conf.empty:
            binned = pd.cut(conf, bins=[i / 10 for i in range(11)])
            counts = binned.value_counts().sort_index()
            counts.index = counts.index.astype(str)  # Interval -> str for charting
            st.bar_chart(counts)

    rej = run.counters.get("risk_rejection_reasons", {})
    if rej:
        st.markdown("**Risk rejection reasons**")
        st.bar_chart(pd.Series(rej))

    if run.benchmark:
        st.markdown("**Benchmark total return % (buy & hold / cash)**")
        st.bar_chart(pd.Series(run.benchmark))


def trades_tab(run: RunData) -> None:
    trades = run.trades
    if trades.empty:
        st.info("No trades in this run.")
        return
    df = trades.copy()
    f1, f2 = st.columns(2)
    with f1:
        df = _multiselect_filter(df, "symbol", "tr_sym")
        df = _multiselect_filter(df, "direction", "tr_dir")
    with f2:
        df = _multiselect_filter(df, "exit_reason", "tr_exit")
    st.dataframe(df, width="stretch")

    if "trade_id" in trades.columns and len(trades):
        tid = st.selectbox("Inspect trade_id", sorted(trades["trade_id"].unique()))
        row = trades[trades["trade_id"] == tid].iloc[0].to_dict()
        st.markdown("**Trade detail**")
        st.json(row)
        st.markdown("**Related agent signal(s)**")
        sig = run.agent_signals
        if not sig.empty and "trade_id" in sig.columns:
            st.dataframe(sig[sig["trade_id"] == tid], width="stretch")
        st.markdown("**Related risk decision(s)**")
        rd = run.risk_decisions
        if not rd.empty and "trade_id" in rd.columns:
            st.dataframe(rd[rd["trade_id"] == tid], width="stretch")


def signals_tab(run: RunData) -> None:
    sig = run.agent_signals
    if sig.empty:
        st.info("No agent signals recorded.")
        return
    df = sig.copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        df = _multiselect_filter(df, "action", "sig_action")
        df = _multiselect_filter(df, "direction", "sig_dir")
    with c2:
        df = _multiselect_filter(df, "symbol", "sig_sym")
        df = _multiselect_filter(df, "accepted", "sig_acc")
    with c3:
        df = _multiselect_filter(df, "rejection_reason", "sig_rej")
        df = _confidence_filter(df, "sig_conf")
    cols = [c for c in ["timestamp", "agent_name", "target_family", "direction",
                        "action", "symbol", "confidence", "reason", "risk_notes",
                        "accepted", "rejection_reason", "trade_id"] if c in df.columns]
    st.dataframe(df[cols] if cols else df, width="stretch")


def risk_tab(run: RunData) -> None:
    rd = run.risk_decisions
    if rd.empty:
        st.info("No risk decisions recorded.")
        return
    df = rd.copy()
    c1, c2 = st.columns(2)
    with c1:
        df = _multiselect_filter(df, "decision", "rd_dec")
        df = _multiselect_filter(df, "symbol", "rd_sym")
    with c2:
        df = _multiselect_filter(df, "rejection_reason", "rd_rej")
    st.dataframe(df, width="stretch")


def compare_tab(reports_dir: str, all_ids: list[str]) -> None:
    chosen = st.multiselect("Select runs to compare", all_ids, default=all_ids[-2:] if len(all_ids) >= 2 else all_ids)
    if len(chosen) < 1:
        st.info("Select at least one run.")
        return
    runs = [load_run(reports_dir, rid) for rid in chosen]

    metric_rows = {r.run_id: r.metrics for r in runs}
    st.markdown("**Metrics comparison**")
    st.dataframe(pd.DataFrame(metric_rows), width="stretch")

    st.markdown("**Equity curves (overlaid)**")
    combined = {}
    for r in runs:
        if not r.equity.empty and "equity" in r.equity.columns:
            e = r.equity.copy()
            e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
            combined[r.run_id] = e.set_index("timestamp")["equity"]
    if combined:
        st.line_chart(pd.DataFrame(combined))
    else:
        st.info("No equity curves to overlay.")

    if len(runs) >= 2:
        st.markdown("**Config diff (run A vs run B)**")
        a, b = runs[0].summary, runs[1].summary
        keys = sorted(set(a) | set(b))
        diff = [{"field": k, runs[0].run_id: a.get(k), runs[1].run_id: b.get(k)}
                for k in keys if a.get(k) != b.get(k)]
        st.dataframe(pd.DataFrame(diff) if diff else pd.DataFrame([{"info": "no differences"}]),
                     width="stretch")


def runtime_tab(reports_dir: str) -> None:
    rt = load_runtime(reports_dir)
    if not rt["exists"]:
        st.info("No runtime data yet. Start the paper runner: "
                "`python -m src.cli run-paper --config config/default.yaml --agent mock`")
        return
    hb = rt["heartbeat"]
    if not hb:
        st.info("Runner has not written a heartbeat yet.")
    else:
        g = st.columns(4)
        g[0].metric("Runner status", hb.get("status", "—"))
        g[1].metric("Market state", hb.get("market_state", "—"))
        g[2].metric("Daily PnL", hb.get("daily_pnl", "—"))
        g[3].metric("Trades today", hb.get("trades_today", "—"))
        g2 = st.columns(4)
        g2[0].metric("Consecutive losses", hb.get("consecutive_losses", "—"))
        g2[1].metric("DAILY_STOP", str(hb.get("daily_stop", False)))
        g2[2].write(f"**Session:** {hb.get('current_session_open','?')} → {hb.get('current_session_close','?')}")
        g2[3].write(f"**Next open/close:** {hb.get('next_market_open','?')} / {hb.get('next_market_close','?')}")
        c = st.columns(3)
        c[0].write(f"**Last bar:** {hb.get('last_bar_time','—')}")
        c[1].write(f"**Last processed bar:** {hb.get('last_processed_bar_time','—')}")
        c[2].write(f"**Last order:** {hb.get('last_order_time','—')}")
        if hb.get("daily_stop"):
            st.error("DAILY_STOP active — no new entries for the rest of the session.")

    st.markdown("**Current paper positions**")
    pos = rt["current_positions"]
    st.dataframe(pd.DataFrame(pos) if pos else pd.DataFrame([{"info": "flat"}]), width="stretch")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Latest signal**")
        st.json(rt["latest_signal"] or {"info": "none"})
    with cols[1]:
        st.markdown("**Latest risk decision**")
        st.json(rt["latest_risk_decision"] or {"info": "none"})

    st.markdown("**Recent runtime events**")
    st.dataframe(rt["events"] if not rt["events"].empty else pd.DataFrame([{"info": "none"}]),
                 width="stretch")
    st.markdown("**Recent errors**")
    st.dataframe(rt["errors"] if not rt["errors"].empty else pd.DataFrame([{"info": "none"}]),
                 width="stretch")


def evolution_tab(reports_dir: str) -> None:
    evo = load_evolution(reports_dir)
    champ = evo["champion"]
    if not champ:
        st.info("No evolution registry yet. Create one with "
                "`python -m src.cli champion` / `create-challenger` / `run-evolution`.")
        return

    st.markdown("**Current Champion**")
    cc = st.columns(3)
    cc[0].write(f"**ID:** {champ.get('champion_id','?')}")
    cc[1].write(f"**Created:** {champ.get('created_at','?')}")
    cc[2].write(f"**Patch:** {champ.get('config_patch') or 'base config'}")
    with st.expander("Champion metrics"):
        st.json(champ.get("metrics", {}))

    prev = evo["previous_champions"]
    st.markdown("**Previous Champions (fallbacks)**")
    st.dataframe(prev if not prev.empty else pd.DataFrame([{"info": "none"}]), width="stretch")

    st.markdown("**Active Challengers / Shadow & Canary performance / Allocations**")
    ch = evo["challengers"]
    if ch:
        rows = []
        for c in ch:
            m = c.get("metrics", {}) or {}
            rows.append({"challenger_id": c.get("challenger_id"), "status": c.get("status"),
                         "allocation_pct": c.get("allocation_pct"), "source": c.get("source"),
                         "num_trades": m.get("num_trades"), "profit_factor": m.get("profit_factor"),
                         "expectancy": m.get("expectancy"), "max_drawdown_pct": m.get("max_drawdown_pct"),
                         "net_pnl_after_costs": m.get("net_pnl_after_costs"),
                         "patch": c.get("config_patch")})
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No challengers yet.")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Promotion candidates / policy pass-fail**")
        st.dataframe(_events_of(evo["events"], ("PROMOTION_EVALUATED", "PROMOTION_PASSED", "PROMOTION_FAILED")),
                     width="stretch")
        st.markdown("**Auto-promotion history**")
        st.dataframe(evo["promotions"] if not evo["promotions"].empty else pd.DataFrame([{"info": "none"}]),
                     width="stretch")
    with cols[1]:
        st.markdown("**Rollback status / history**")
        st.dataframe(evo["rollbacks"] if not evo["rollbacks"].empty else pd.DataFrame([{"info": "none"}]),
                     width="stretch")
        st.markdown("**Drift alerts**")
        st.dataframe(evo["drift"] if not evo["drift"].empty else pd.DataFrame([{"info": "none"}]),
                     width="stretch")

    st.markdown("**Bandit allocation history**")
    st.dataframe(evo["allocations"] if not evo["allocations"].empty else pd.DataFrame([{"info": "none"}]),
                 width="stretch")
    st.markdown("**Mutation history**")
    st.dataframe(evo["mutations"] if not evo["mutations"].empty else pd.DataFrame([{"info": "none"}]),
                 width="stretch")
    st.markdown("**Recent evolution events**")
    st.dataframe(evo["events"].tail(40) if not evo["events"].empty else pd.DataFrame([{"info": "none"}]),
                 width="stretch")
    st.caption("Read-only. No live orders, no live trading, no broker controls.")


def _events_of(df, types):
    if df.empty or "event" not in df.columns:
        return pd.DataFrame([{"info": "none"}])
    sub = df[df["event"].isin(types)]
    return sub if not sub.empty else pd.DataFrame([{"info": "none"}])


def main() -> None:
    st.title("bullbear-ai-trader — backtest dashboard")
    st.caption(SAFETY)

    reports_dir = st.sidebar.text_input("Reports dir", _reports_dir())
    all_ids = list_run_ids(reports_dir)
    if not all_ids:
        st.warning(f"No runs found under `{reports_dir}/runs/`. Run a backtest first.")
        return

    run_id = st.sidebar.selectbox("Run", ["latest", *all_ids], index=0)
    run = load_run(reports_dir, run_id)

    tabs = st.tabs(["Overview", "Charts", "Trades", "Agent Signals", "Risk Decisions",
                    "Runtime (Paper)", "Evolution", "Compare"])
    with tabs[0]:
        overview_tab(run)
    with tabs[1]:
        charts_tab(run)
    with tabs[2]:
        trades_tab(run)
    with tabs[3]:
        signals_tab(run)
    with tabs[4]:
        risk_tab(run)
    with tabs[5]:
        runtime_tab(reports_dir)
    with tabs[6]:
        evolution_tab(reports_dir)
    with tabs[7]:
        compare_tab(reports_dir, all_ids)


main()
