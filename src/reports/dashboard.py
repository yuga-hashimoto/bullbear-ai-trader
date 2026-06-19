"""Streamlit dashboard for BullBear AI Trader (READ-ONLY).

A clean, professional monitoring console: live status, decision log, and the
champion/challenger evolution leaderboard. No order buttons, no trade switches —
viewing only. Styling is a dark fintech theme with minimal ornamentation.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path

import yaml

# Allow importing from project root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.reports.loader import (
    DEFAULT_REPORTS_DIR,
    load_diary_events,
    load_evolution,
    load_runtime,
    load_runtime_performance,
)
from src.reports.diary import translate_reason

st.set_page_config(page_title="BullBear AI", layout="wide",
                   initial_sidebar_state="collapsed")

SAFETY_INFO = "閲覧専用コンソール — 実資金は動きません。注文・取引操作はできません。"
LIVE_REFRESH_INTERVAL = "10s"


# ───────────────────────────────────────────────────────── theme / css
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        :root{
          --bg:#0b0e14; --surface:#141925; --surface-2:#1b2231; --line:#232b3a;
          --txt:#e7ecf3; --muted:#8a97ab; --faint:#5d6982;
          --pos:#1fce8f; --neg:#f6465d; --accent:#6c8cff; --amber:#f0b429;
        }
        .stApp{background:radial-gradient(1200px 600px at 80% -10%,#161c2b 0%,var(--bg) 55%);}
        header[data-testid="stHeader"]{background:transparent;}
        #MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}
        .block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1180px;
          font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--txt);}
        .stApp, .block-container, p, span, div{font-family:'Inter',-apple-system,sans-serif;}
        .num{font-variant-numeric:tabular-nums;}
        .pos{color:var(--pos)!important;} .neg{color:var(--neg)!important;} .flat{color:var(--muted)!important;}

        /* header */
        .hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:.35rem;}
        .hd-l{display:flex;align-items:baseline;gap:.7rem;}
        .brand{font-size:1.55rem;font-weight:700;letter-spacing:-.02em;}
        .brand b{color:var(--accent);}
        .brand-sub{font-size:.78rem;color:var(--faint);font-weight:500;}
        .safe{font-size:.74rem;color:var(--faint);margin-bottom:1.4rem;}

        /* status pill */
        .pill{display:inline-flex;align-items:center;gap:.5rem;padding:.4rem .85rem;border-radius:999px;
          font-size:.8rem;font-weight:600;border:1px solid var(--line);background:var(--surface);}
        .dot{width:8px;height:8px;border-radius:50%;display:inline-block;}
        .pill-live{color:var(--pos);} .pill-live .dot{background:var(--pos);box-shadow:0 0 0 0 rgba(31,206,143,.6);animation:pulse 1.8s infinite;}
        .pill-sleep{color:var(--muted);} .pill-sleep .dot{background:var(--muted);}
        .pill-stop{color:var(--neg);} .pill-stop .dot{background:var(--neg);}
        @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(31,206,143,.5);}70%{box-shadow:0 0 0 7px rgba(31,206,143,0);}100%{box-shadow:0 0 0 0 rgba(31,206,143,0);}}

        /* kpi grid */
        .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:.4rem 0 1.4rem;}
        .kpi{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px;
          transition:border-color .2s,transform .2s;}
        .kpi:hover{border-color:#33405a;transform:translateY(-2px);}
        .kpi-l{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;}
        .kpi-v{font-size:1.7rem;font-weight:700;margin-top:.35rem;letter-spacing:-.02em;}
        .kpi-s{font-size:.75rem;color:var(--faint);margin-top:.2rem;}

        /* sections */
        .sec{display:flex;align-items:baseline;gap:.6rem;margin:1.6rem 0 .8rem;}
        .sec-t{font-size:1.02rem;font-weight:700;letter-spacing:-.01em;}
        .sec-s{font-size:.76rem;color:var(--faint);}

        /* panel / cards */
        .panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px 20px;}
        .pos-row{display:flex;flex-wrap:wrap;gap:1.6rem;align-items:center;background:var(--surface);
          border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:12px;padding:14px 18px;margin-bottom:10px;}
        .pos-row .k{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;}
        .pos-row .v{font-size:1.05rem;font-weight:600;margin-top:1px;}
        .sym{font-size:1.15rem;font-weight:700;}
        .empty{color:var(--muted);font-size:.9rem;padding:14px 2px;}

        /* decision */
        .dec{display:grid;grid-template-columns:230px 1fr;gap:18px;}
        .badge{display:inline-block;padding:.3rem .7rem;border-radius:8px;font-weight:600;font-size:.92rem;
          border:1px solid var(--line);}
        .badge.pos{background:rgba(31,206,143,.12);border-color:rgba(31,206,143,.35);}
        .badge.neg{background:rgba(246,70,93,.12);border-color:rgba(246,70,93,.35);}
        .badge.flat{background:var(--surface-2);}
        .meter{height:7px;border-radius:999px;background:var(--surface-2);overflow:hidden;margin-top:.5rem;}
        .meter > i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#9db4ff);}
        .reason{background:var(--surface-2);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
          font-size:.95rem;line-height:1.6;}

        /* timeline */
        .tl{display:flex;flex-direction:column;}
        .tl-row{display:grid;grid-template-columns:128px 64px 1fr;gap:14px;align-items:center;
          padding:11px 2px;border-bottom:1px solid var(--line);}
        .tl-row:last-child{border-bottom:none;}
        .tl-time{font-size:.78rem;color:var(--faint);font-variant-numeric:tabular-nums;}
        .tl-tag{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.03em;text-align:center;
          padding:.18rem 0;border-radius:6px;border:1px solid var(--line);color:var(--muted);}
        .tag-ind{color:#9db4ff;border-color:rgba(108,140,255,.3);} .tag-blue{color:#56b6ff;border-color:rgba(86,182,255,.3);}
        .tag-green{color:var(--pos);border-color:rgba(31,206,143,.3);} .tag-amber{color:var(--amber);border-color:rgba(240,180,41,.3);}
        .tag-red{color:var(--neg);border-color:rgba(246,70,93,.3);}
        .tl-msg{font-size:.9rem;color:var(--txt);line-height:1.45;}

        /* leaderboard */
        table.lb{width:100%;border-collapse:collapse;}
        table.lb th{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;
          text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);font-weight:600;}
        table.lb td{padding:13px 12px;border-bottom:1px solid var(--line);font-size:.92rem;}
        table.lb tr:last-child td{border-bottom:none;}
        table.lb td.num{font-variant-numeric:tabular-nums;text-align:right;font-weight:600;}
        table.lb .rk{width:34px;height:24px;display:inline-flex;align-items:center;justify-content:center;
          border-radius:6px;background:var(--surface-2);font-size:.78rem;font-weight:700;color:var(--muted);}
        table.lb .nm{font-weight:600;}
        .st-tag{font-size:.72rem;padding:.2rem .55rem;border-radius:999px;border:1px solid var(--line);color:var(--muted);}

        .stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
        .stTabs [data-baseweb="tab"]{height:42px;padding:0 18px;font-weight:600;font-size:.9rem;color:var(--muted);}
        .stTabs [aria-selected="true"]{color:var(--txt);}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────── helpers
def _reports_dir() -> str:
    return os.environ.get("BULLBEAR_REPORTS_DIR", DEFAULT_REPORTS_DIR)


def _tone(v: float) -> str:
    return "pos" if v > 0 else "neg" if v < 0 else "flat"


def _signed_usd(v: float) -> str:
    return f"{'+' if v >= 0 else '−'}${abs(v):,.2f}"


def position_direction_label(direction: str) -> str:
    if direction == "UP":
        return "値上がり狙い（ETFを買い）"
    if direction == "DOWN":
        return "値下がり局面狙い（逆ETFを買い）"
    return "方向不明"


_ACTION = {
    "BUY_BULL": ("買い — 上昇を期待", "pos"),
    "BUY_BEAR": ("買い — 下落局面（逆ETF）", "neg"),
    "NO_TRADE": ("様子見", "flat"),
    "EXIT": ("手仕舞い", "flat"),
}

_DIARY_TAG = {
    "🧠": ("判断", "tag-ind"), "🛡️": ("審査", "tag-ind"), "🛒": ("約定", "tag-blue"),
    "💰": ("決済", "tag-green"), "🔔": ("市場", "tag-blue"), "▶️": ("起動", ""),
    "⏹️": ("停止", ""), "⚠️": ("警告", "tag-amber"), "🛑": ("停止", "tag-red"),
}


def get_nickname(challenger_id: str) -> str:
    if not challenger_id or challenger_id == "Base_Model":
        return "オグリキャップ"
    prefixes = ["ディープ", "キング", "トウカイ", "シンボリ", "メジロ", "オルフェ", "ゴールド",
                "サイレンス", "オグリ", "サクラ", "ウオッカ", "ジェンティル", "ナリタ", "スペシャル",
                "コントレイル", "アーモンド", "マヤノ", "ダイワ", "ライス", "ハル", "タマモ",
                "セイウン", "マチカネ", "エア", "アグネス", "グラス", "ビワ", "ツインターボ", "ミホノ"]
    suffixes = ["インパクト", "カメハメハ", "テイオー", "ルドルフ", "マックイーン", "エーヴル", "シップ",
                "スズカ", "キャップ", "ウララ", "バクシンオー", "ダービー", "ドンナ", "ブライアン",
                "ウィーク", "レイル", "アイ", "トップガン", "スカーレット", "シャワー", "クロス",
                "スカイ", "フクキタル", "グルーヴ", "タキオン", "ワンダー", "ハヤヒデ", "ブルボン", "オー"]
    import hashlib
    h = int(hashlib.md5(challenger_id.encode("utf-8")).hexdigest(), 16)
    return f"{prefixes[h % len(prefixes)]}{suffixes[(h // len(prefixes)) % len(suffixes)]}"


def _load_initial_cash(config_path: str) -> float:
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
            return float(cfg_data.get("backtest", {}).get("initial_cash", 100000.0))
        except Exception:
            pass
    return 100000.0


def update_initial_cash_in_yaml(file_path: str, new_value: float) -> bool:
    try:
        path = Path(file_path)
        if not path.exists():
            return False
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        in_backtest = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("backtest:"):
                in_backtest = True
                continue
            if in_backtest and line.strip() and not line.startswith(" ") and not line.startswith("#"):
                in_backtest = False
            if in_backtest and stripped.startswith("initial_cash:"):
                prefix, suffix = line.split(":", 1)
                sub = suffix.split("#", 1)
                comment = f"  # {sub[1].strip()}" if len(sub) > 1 else ""
                lines[idx] = f"{prefix}: {new_value}{comment}\n"
                path.write_text("".join(lines), encoding="utf-8")
                return True
    except Exception as e:  # noqa: BLE001
        st.error(f"元本の書き換えに失敗しました: {e}")
    return False


def restart_runner(config_path: str) -> None:
    subprocess.run([".venv/bin/python", "-m", "src.cli", "stop-runner",
                    "--config", config_path], check=False)
    time.sleep(2)
    log_dir = Path("reports/runtime")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "paper_runner.log", "a") as lf:
        subprocess.Popen([".venv/bin/python", "-m", "src.cli", "run-paper",
                          "--config", config_path, "--agent", "external"],
                         stdout=lf, stderr=lf, start_new_session=True)


def _section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="sec"><span class="sec-t">{title}</span>'
                f'<span class="sec-s">{sub}</span></div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────── tab: live
@st.fragment(run_every=LIVE_REFRESH_INTERVAL)
def render_status_tab(reports_dir: str, config_path: str) -> None:
    rt = load_runtime(reports_dir)
    if not rt["exists"]:
        st.markdown('<div class="panel empty">AIシステム（PaperRunner）は現在オフラインです。'
                    '通常は米国市場の時間帯に自動稼働します。</div>', unsafe_allow_html=True)
        return

    hb = rt["heartbeat"] or {}
    status = hb.get("status", "stopped")
    initial_cash = _load_initial_cash(config_path)
    perf = load_runtime_performance(reports_dir, initial_cash)
    daily_pnl = float(hb.get("daily_pnl", 0.0))
    trades_today = int(hb.get("trades_today", 0))
    total_ret = perf["total_return_pct"]

    # KPI grid
    cards = [
        ("ペーパー資産", f"${perf['current_equity']:,.2f}", "現在の評価額", "flat"),
        ("本日損益", _signed_usd(daily_pnl), "当日の確定+評価損益", _tone(daily_pnl)),
        ("累計リターン", f"{total_ret:+.2f}%", f"元本 ${initial_cash:,.0f} 比", _tone(total_ret)),
        ("本日約定", f"{trades_today}", "回", "flat"),
    ]
    grid = '<div class="grid">' + "".join(
        f'<div class="kpi"><div class="kpi-l">{l}</div>'
        f'<div class="kpi-v num {tone}">{v}</div><div class="kpi-s">{s}</div></div>'
        for l, v, s, tone in cards) + "</div>"
    st.markdown(grid, unsafe_allow_html=True)

    _section("保有ポジション")
    pos = rt["current_positions"]
    if not pos:
        st.markdown('<div class="panel empty">現在ポジションはありません（現金で待機中）。</div>',
                    unsafe_allow_html=True)
    else:
        for p in pos:
            entry = p.get("entry_price")
            entry_s = f"${entry:,.2f}" if entry is not None else "—"
            up = p.get("unrealized_pct")
            up_s = f"{up:+.2f}%" if up is not None else "—"
            tone = _tone(up if up is not None else 0)
            st.markdown(
                f'<div class="pos-row"><div><div class="k">銘柄</div>'
                f'<div class="v sym">{p.get("symbol","—")}</div></div>'
                f'<div><div class="k">方向</div><div class="v">{position_direction_label(str(p.get("direction","")))}</div></div>'
                f'<div><div class="k">取得単価</div><div class="v num">{entry_s}</div></div>'
                f'<div><div class="k">含み損益</div><div class="v num {tone}">{up_s}</div></div></div>',
                unsafe_allow_html=True)

    _section("直近のAI判断")
    sig = rt["latest_signal"]
    if not sig:
        st.markdown('<div class="panel empty">判断履歴がまだありません。</div>', unsafe_allow_html=True)
        return
    label, tone = _ACTION.get(sig.get("action", ""), (sig.get("action", "—"), "flat"))
    conf = sig.get("confidence")
    try:
        conf_pct = int(float(conf) * 100) if conf is not None else 0
    except Exception:
        conf_pct = 0
    st.markdown(
        f'<div class="panel"><div class="dec">'
        f'<div><span class="badge {tone}">{label}</span>'
        f'<div style="margin-top:.9rem" class="k kpi-l">対象銘柄</div>'
        f'<div class="v num" style="font-size:1.05rem;font-weight:600">{sig.get("symbol") or "—"}</div>'
        f'<div style="margin-top:.7rem" class="kpi-l">自信度 {conf_pct}%</div>'
        f'<div class="meter"><i style="width:{max(0,min(conf_pct,100))}%"></i></div></div>'
        f'<div><div class="kpi-l" style="margin-bottom:.5rem">判断の根拠</div>'
        f'<div class="reason">{translate_reason(sig.get("reason"))}</div></div>'
        f'</div></div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────── tab: log
@st.fragment(run_every=LIVE_REFRESH_INTERVAL)
def render_diary_tab(reports_dir: str) -> None:
    _section("判断ログ", "日本時間 / 判断・見送り理由・売買結果")
    entries = load_diary_events(reports_dir, limit=40)
    if not entries:
        st.markdown('<div class="panel empty">まだ記録がありません。取引が始まると自動で記録されます。</div>',
                    unsafe_allow_html=True)
        return
    rows = ""
    for e in entries:
        tag, cls = _DIARY_TAG.get(e.get("icon", ""), ("ログ", ""))
        time_s = str(e["time"]).replace(" JST", "")
        rows += (f'<div class="tl-row"><div class="tl-time">{time_s}</div>'
                 f'<div class="tl-tag {cls}">{tag}</div>'
                 f'<div class="tl-msg">{e["message"]}</div></div>')
    st.markdown(f'<div class="panel"><div class="tl">{rows}</div></div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────── tab: evolution
@st.fragment(run_every=LIVE_REFRESH_INTERVAL)
def render_evolution_tab(reports_dir: str, initial_cash: float) -> None:
    evo = load_evolution(reports_dir)
    champ = evo["champion"]
    if not champ:
        st.markdown('<div class="panel empty">自動成長システムは準備中です。最初のデータ取得をお待ちください。</div>',
                    unsafe_allow_html=True)
        return

    perf = load_runtime_performance(reports_dir, initial_cash)
    total_ret = perf["total_return_pct"]
    win = perf["win_rate_pct"]
    win_s = "—" if win is None else f"{win:.1f}%"
    nickname = get_nickname(champ.get("champion_id", "Base_Model"))

    _section("チャンピオン", "現在の正式採用モデル")
    st.markdown(
        f'<div class="panel"><div class="grid" style="margin:0;grid-template-columns:1.4fr 1fr 1fr 1fr">'
        f'<div class="kpi" style="border-left:3px solid var(--accent)"><div class="kpi-l">採用モデル</div>'
        f'<div class="kpi-v" style="font-size:1.25rem">{nickname}</div>'
        f'<div class="kpi-s">{champ.get("champion_id","")}</div></div>'
        f'<div class="kpi"><div class="kpi-l">累計リターン</div><div class="kpi-v num {_tone(total_ret)}">{total_ret:+.2f}%</div>'
        f'<div class="kpi-s">{_signed_usd(perf["total_pnl"])}</div></div>'
        f'<div class="kpi"><div class="kpi-l">勝率</div><div class="kpi-v num">{win_s}</div>'
        f'<div class="kpi-s">決済 {perf["closed_trades"]} 件</div></div>'
        f'<div class="kpi"><div class="kpi-l">本日損益</div><div class="kpi-v num {_tone(perf["daily_pnl"])}">{_signed_usd(perf["daily_pnl"])}</div>'
        f'<div class="kpi-s">評価額 ${perf["current_equity"]:,.2f}</div></div></div></div>',
        unsafe_allow_html=True)

    _section("挑戦者リーダーボード", "本命と同じ5分足で仮想売買中（最大5体）")
    ch = evo["challengers"]
    if not ch:
        st.markdown('<div class="panel empty">稼働中の挑戦者はいません（本命のみ稼働）。</div>',
                    unsafe_allow_html=True)
        return
    ranked = sorted(ch, key=lambda c: float((c.get("metrics") or {}).get("total_return_pct", 0.0)),
                    reverse=True)
    rows = ""
    for i, c in enumerate(ranked, 1):
        m = c.get("metrics") or {}
        ret = float(m.get("total_return_pct", 0.0))
        cwin = float(m.get("win_rate_pct", 0.0))
        trades = int(m.get("num_trades", 0))
        active = str(c.get("status", "")).upper() in ("SHADOW", "BACKTEST_PASSED", "DRAFT")
        state = "シャドウ稼働" if active else "カナリア審査"
        rows += (f'<tr><td><span class="rk">{i}</span></td>'
                 f'<td class="nm">{get_nickname(c.get("challenger_id"))}</td>'
                 f'<td class="num {_tone(ret)}">{ret:+.2f}%</td>'
                 f'<td class="num">{cwin:.1f}%</td>'
                 f'<td class="num">{trades}</td>'
                 f'<td><span class="st-tag">{state}</span></td></tr>')
    st.markdown(
        '<div class="panel"><table class="lb"><thead><tr>'
        '<th>順位</th><th>モデル</th><th style="text-align:right">リターン</th>'
        '<th style="text-align:right">勝率</th><th style="text-align:right">取引</th><th>状態</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────── main
def main() -> None:
    _inject_css()

    # sidebar: capital control (kept, de-cluttered)
    st.sidebar.markdown("### 設定")
    config_path = st.sidebar.text_input("設定ファイル", "config/default.yaml")
    initial_cash = _load_initial_cash(config_path)
    new_cash = st.sidebar.number_input("運用元本 (USD)", min_value=1.0,
                                       value=float(initial_cash), step=1000.0)
    if new_cash != initial_cash and st.sidebar.button("元本を更新して再起動"):
        if update_initial_cash_in_yaml(config_path, new_cash):
            st.sidebar.success(f"元本を ${new_cash:,.0f} に更新。再起動します。")
            restart_runner(config_path)
            st.rerun()
    reports_dir = st.sidebar.text_input("レポート出力先", _reports_dir())

    status = "stopped"
    rt = load_runtime(reports_dir)
    if rt["exists"]:
        status = (rt["heartbeat"] or {}).get("status", "stopped")
    pill = {"running": ("ライブ稼働中", "live"),
            "sleeping": ("待機中 — 市場クローズ", "sleep")}.get(status, ("停止中", "stop"))
    st.markdown(
        f'<div class="hd"><div class="hd-l"><span class="brand">Bull<b>Bear</b> AI</span>'
        f'<span class="brand-sub">トレーディング・コンソール</span></div>'
        f'<span class="pill pill-{pill[1]}"><span class="dot"></span>{pill[0]}</span></div>'
        f'<div class="safe">{SAFETY_INFO}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["ライブ", "判断ログ", "進化 (A/Bテスト)"])
    with tabs[0]:
        render_status_tab(reports_dir, config_path)
    with tabs[1]:
        render_diary_tab(reports_dir)
    with tabs[2]:
        render_evolution_tab(reports_dir, initial_cash)


if __name__ == "__main__":
    main()
