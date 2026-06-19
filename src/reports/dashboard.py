"""Streamlit dashboard for BullBear AI — Dual Momentum (READ-ONLY).

Shows the live monthly dual-momentum strategy: what to hold this month, the paper
equity curve since inception, the asset-momentum ranking, and recent holdings.
Reads ``reports/runtime/dual_momentum.json`` (written by the monthly runner). No
order buttons — viewing only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="BullBear AI — Dual Momentum", layout="wide",
                   initial_sidebar_state="collapsed")

DEFAULT_REPORTS_DIR = os.environ.get("BULLBEAR_REPORTS_DIR", "reports")
SAFETY_INFO = "閲覧専用 — 実資金は動きません。月次のデュアルモメンタム戦略の状況を表示します。"

_ASSET_LABEL = {
    "SPY": "S&P500 (米国大型株)", "QQQ": "NASDAQ100 (米ハイテク)",
    "EFA": "先進国株 (米国除く)", "EEM": "新興国株", "GLD": "金 (ゴールド)",
    "TLT": "米国長期債 (リスク退避)", "CASH": "現金 (リスク退避)",
}


def _load_state() -> dict | None:
    path = Path(DEFAULT_REPORTS_DIR) / "runtime" / "dual_momentum.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        :root{--bg:#0b0e14;--surface:#141925;--surface-2:#1b2231;--line:#232b3a;
          --txt:#e7ecf3;--muted:#8a97ab;--faint:#5d6982;--pos:#1fce8f;--neg:#f6465d;--accent:#6c8cff;}
        .stApp{background:radial-gradient(1200px 600px at 80% -10%,#161c2b 0%,var(--bg) 55%);}
        header[data-testid="stHeader"]{background:transparent;} #MainMenu,footer{visibility:hidden;}
        .block-container{padding-top:2.2rem;max-width:1180px;color:var(--txt);
          font-family:'Inter',-apple-system,sans-serif;}
        .num{font-variant-numeric:tabular-nums;} .pos{color:var(--pos)!important;} .neg{color:var(--neg)!important;}
        .hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:.3rem;}
        .brand{font-size:1.5rem;font-weight:700;letter-spacing:-.02em;} .brand b{color:var(--accent);}
        .brand-sub{font-size:.78rem;color:var(--faint);margin-left:.6rem;}
        .safe{font-size:.74rem;color:var(--faint);margin-bottom:1.3rem;}
        .pill{display:inline-flex;align-items:center;gap:.5rem;padding:.4rem .85rem;border-radius:999px;
          font-size:.8rem;font-weight:600;border:1px solid var(--line);background:var(--surface);}
        .dot{width:8px;height:8px;border-radius:50%;display:inline-block;}
        .on{color:var(--pos);} .on .dot{background:var(--pos);} .off{color:var(--muted);} .off .dot{background:var(--muted);}
        .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:.4rem 0 1.4rem;}
        .kpi{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px;}
        .kpi-l{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;}
        .kpi-v{font-size:1.6rem;font-weight:700;margin-top:.35rem;letter-spacing:-.02em;}
        .kpi-s{font-size:.75rem;color:var(--faint);margin-top:.2rem;}
        .sec{font-size:1.0rem;font-weight:700;margin:1.5rem 0 .7rem;}
        .panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 20px;}
        .rk{display:flex;align-items:center;gap:12px;padding:9px 2px;border-bottom:1px solid var(--line);}
        .rk:last-child{border-bottom:none;}
        .rk .nm{width:230px;font-size:.9rem;font-weight:600;}
        .rk .bar{flex:1;height:9px;border-radius:999px;background:var(--surface-2);overflow:hidden;}
        .rk .bar > i{display:block;height:100%;border-radius:999px;}
        .rk .v{width:74px;text-align:right;font-size:.86rem;font-variant-numeric:tabular-nums;font-weight:600;}
        table.h{width:100%;border-collapse:collapse;} table.h th{font-size:.7rem;color:var(--muted);
          text-transform:uppercase;text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);}
        table.h td{padding:11px 12px;border-bottom:1px solid var(--line);font-size:.9rem;font-variant-numeric:tabular-nums;}
        table.h tr:last-child td{border-bottom:none;}
        </style>
        """, unsafe_allow_html=True)


def _label(sym: str) -> str:
    return f"{sym} — {_ASSET_LABEL.get(sym, sym)}"


def main() -> None:
    _inject_css()
    state = _load_state()
    if not state:
        st.markdown('<div class="hd"><span class="brand">Bull<b>Bear</b> AI</span></div>',
                    unsafe_allow_html=True)
        st.info("デュアルモメンタムの状態がまだありません。月次ランナーの初回実行をお待ちください"
                "（`python -m src.cli run-dual-momentum`）。")
        return

    rec = state["recommendation"]
    paper = state["paper"]
    on = rec["is_risk_on"]
    pill = ("リスクオン（株を推奨）", "on") if on else ("リスク退避（債券/現金を推奨）", "off")
    st.markdown(
        f'<div class="hd"><div><span class="brand">Bull<b>Bear</b> AI</span>'
        f'<span class="brand-sub">Dual Momentum · {state["as_of_month"]}時点</span></div>'
        f'<span class="pill {pill[1]}"><span class="dot"></span>{pill[0]}</span></div>'
        f'<div class="safe">{SAFETY_INFO}</div>', unsafe_allow_html=True)

    bt = state.get("backtest_reference", {})
    ret_tone = "pos" if paper["total_return_pct"] >= 0 else "neg"
    months = paper.get("months", 0)
    started = state.get("inception_month", "—")
    cards = [
        ("今月の推奨（シグナル）", f"{rec['asset']} ×{rec['leverage']}", _ASSET_LABEL.get(rec["asset"], ""), ""),
        ("ペーパー資産", f"{paper['equity']:,.0f}",
         f"元本 {paper['capital']:,.0f} / {started}運用開始", ""),
        ("運用リターン（実績）", f"{paper['total_return_pct']:+,.1f}%",
         (f"運用{months}ヶ月" if months else "今月スタート（実績はこれから）"), ret_tone),
        ("過去検証（仮想・参考）", f"年率{bt.get('cagr_pct','—')}%",
         f"{bt.get('since','')}〜 / 最大DD {bt.get('max_drawdown_pct','—')}% / Sharpe {bt.get('sharpe','—')}", ""),
    ]
    grid = '<div class="grid">' + "".join(
        f'<div class="kpi"><div class="kpi-l">{l}</div><div class="kpi-v num {t}">{v}</div>'
        f'<div class="kpi-s">{s}</div></div>' for l, v, s, t in cards) + "</div>"
    st.markdown(grid, unsafe_allow_html=True)

    st.markdown('<div class="panel" style="color:var(--muted);font-size:.88rem">'
                '⚠️ これは戦略の<b>推奨（シグナル）</b>です。実際にEEM等を保有しているわけではなく、'
                '自動売買もしません。<b>買うかどうかはあなたが判断します。</b>「ペーパー資産」は'
                'この推奨どおりに動いた場合の仮想成績で、' +
                ('<b>今月開始したばかりなので実績はこれから</b>積み上がります。' if months == 0
                 else f'運用開始から{months}ヶ月の仮想実績です。') +
                '右上「過去検証」は2005年からの仮想シミュレーション（実際の保有ではありません）。'
                '</div>', unsafe_allow_html=True)

    # forward paper equity curve
    curve = state.get("equity_curve") or []
    if len(curve) > 1:
        st.markdown('<div class="sec">ペーパー資産の推移（運用開始から）</div>', unsafe_allow_html=True)
        df = pd.DataFrame(curve)
        df["month"] = pd.to_datetime(df["month"])
        st.line_chart(df.set_index("month")["equity"], height=240, color="#6c8cff")

    # momentum ranking
    st.markdown('<div class="sec">資産モメンタム・ランキング（強い順に推奨）</div>', unsafe_allow_html=True)
    ranking = rec.get("ranking") or []
    mx = max((abs(r["momentum_pct"]) for r in ranking), default=1.0) or 1.0
    rows = ""
    for r in ranking:
        m = r["momentum_pct"]
        held = " ◀ 今月の推奨" if r["symbol"] == rec["asset"] else ""
        color = "var(--pos)" if m >= 0 else "var(--neg)"
        w = min(100, abs(m) / mx * 100)
        rows += (f'<div class="rk"><div class="nm">{_label(r["symbol"])}{held}</div>'
                 f'<div class="bar"><i style="width:{w}%;background:{color}"></i></div>'
                 f'<div class="v {"pos" if m>=0 else "neg"}">{m:+.1f}%</div></div>')
    st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)

    # holding history
    st.markdown('<div class="sec">直近の推奨・仮想保有の履歴</div>', unsafe_allow_html=True)
    hist = state.get("history") or []
    hrows = "".join(
        f'<tr><td>{h["month"]}</td><td>{_label(h["held"]) if h.get("held") else "—"}</td>'
        f'<td class="{"pos" if (h.get("return_pct") or 0)>=0 else "neg"}">{h.get("return_pct",0):+.1f}%</td></tr>'
        for h in reversed(hist))
    st.markdown(f'<div class="panel"><table class="h"><thead><tr><th>月</th><th>保有資産</th>'
                f'<th>月次リターン</th></tr></thead><tbody>{hrows}</tbody></table></div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
