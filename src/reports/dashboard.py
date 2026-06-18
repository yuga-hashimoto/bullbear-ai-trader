"""Streamlit dashboard for BullBear AI Trader (READ-ONLY / BEGINNER-FRIENDLY).

This UI is designed to be extremely simple and clear for stock trading beginners.
It avoids complex jargon and focuses on:
  1. What is the AI doing right now?
  2. How much money do we have?
  3. What are the results so far?
"""
from __future__ import annotations

import os
import sys
import yaml
import subprocess
import time
from pathlib import Path

# Allow importing from project root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from src.reports.loader import (
    DEFAULT_REPORTS_DIR,
    RunData,
    list_run_ids,
    load_evolution,
    load_run,
    load_runtime,
)

st.set_page_config(page_title="BullBear AI トレーダー", layout="wide")

# Safety description
SAFETY_INFO = "🔒 閲覧専用画面（実際のお金は動きません）。注文ボタンや取引スイッチはありません。"

def _reports_dir() -> str:
    return os.environ.get("BULLBEAR_REPORTS_DIR", DEFAULT_REPORTS_DIR)

def translate_action(action: str) -> str:
    mapping = {
        "BUY_BULL": "🟢 値上がりを期待して「買い」",
        "BUY_BEAR": "🔴 値下がりを期待して「買い」",
        "NO_TRADE": "⚪ 様子見（何もしない）",
        "EXIT": "🟡 手仕舞い（売却）",
    }
    return mapping.get(action, f"❓ {action}")

def translate_direction(direction: str) -> str:
    mapping = {
        "UP": "📈 上昇傾向",
        "DOWN": "📉 下落傾向",
        "FLAT": "➡️ 横ばい",
    }
    return mapping.get(direction, direction)

def get_nickname(challenger_id: str) -> str:
    if not challenger_id or challenger_id == "Base_Model":
        return "オグリキャップ (本命)"
    
    # 競走馬風・和風・カタカナ風の語彙
    prefixes = [
        "ディープ", "キング", "トウカイ", "シンボリ", "メジロ", "オルフェ", "ゴールド", "サイレンス", "オグリ", 
        "サクラ", "ウオッカ", "ジェンティル", "ナリタ", "スペシャル", "コントレイル", "アーモンド", "マヤノ", 
        "ダイワ", "ライス", "ハル", "タマモ", "セイウン", "マチカネ", "エア", "アグネス", "グラス", "ビワ",
        "ツインターボ", "ミホノ", "サイレンス"
    ]
    suffixes = [
        "インパクト", "カメハメハ", "テイオー", "ルドルフ", "マックイーン", "エーヴル", "シップ", "スズカ", "キャップ", 
        "ウララ", "バクシンオー", "ダービー", "ドンナ", "ブライアン", "ウィーク", "レイル", "アイ", "トップガン", 
        "スカーレット", "シャワー", "クロス", "スカイ", "フクキタル", "グルーヴ", "タキオン", "ワンダー", "ハヤヒデ",
        "ブルボン", "オー", "ターボ"
    ]
    
    import hashlib
    # 一意のハッシュ値からペアを選ぶ
    h = int(hashlib.md5(challenger_id.encode('utf-8')).hexdigest(), 16)
    p_idx = h % len(prefixes)
    s_idx = (h // len(prefixes)) % len(suffixes)
    
    return f"{prefixes[p_idx]}{suffixes[s_idx]}"

def update_initial_cash_in_yaml(file_path: str, new_value: float) -> bool:
    try:
        path = Path(file_path)
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        
        in_backtest = False
        replaced = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("backtest:"):
                in_backtest = True
                continue
            
            if in_backtest and line.strip() and not line.startswith(" ") and not line.startswith("#"):
                in_backtest = False
            
            if in_backtest and stripped.startswith("initial_cash:"):
                parts = line.split(":", 1)
                prefix = parts[0]
                suffix = parts[1]
                sub_parts = suffix.split("#", 1)
                comment = f"  # {sub_parts[1].strip()}" if len(sub_parts) > 1 else ""
                lines[idx] = f"{prefix}: {new_value}{comment}\n"
                replaced = True
                break
        
        if replaced:
            with path.open("w", encoding="utf-8") as f:
                f.writelines(lines)
            return True
    except Exception as e:
        st.error(f"元本書き換え中にエラーが発生しました: {e}")
    return False

def restart_runner(config_path: str) -> None:
    # Stop existing runner
    subprocess.run([".venv/bin/python", "-m", "src.cli", "stop-runner", "--config", config_path], check=False)
    time.sleep(2)
    
    # Start new runner in background
    log_dir = Path("reports/runtime")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "paper_runner.log"
    
    with open(log_file, "a") as lf:
        subprocess.Popen(
            [".venv/bin/python", "-m", "src.cli", "run-paper", "--config", config_path, "--agent", "external"],
            stdout=lf,
            stderr=lf,
            start_new_session=True
        )

# --- TAB 1: 🏠 現在の状況 (Live Status) ---
def render_status_tab(reports_dir: str, config_path: str) -> None:
    rt = load_runtime(reports_dir)
    
    # 1. Runner state banner
    if not rt["exists"]:
        st.warning("⚠️ AIシステム（PaperRunner）が動いていません。")
        st.info("※通常は夜間に自動で稼働します。")
        return
        
    hb = rt["heartbeat"] or {}
    status = hb.get("status", "stopped")
    
    if status == "running":
        st.success("🟢 **AIは現在元気に活動中！**（アメリカの取引時間です）")
    elif status == "sleeping":
        st.info("💤 **AIは待機中** （アメリカ市場が閉まっているため、次のオープンを待っています）")
    else:
        st.error("🔴 AIは現在停止しています。")

    # 2. Main wallet metrics
    st.markdown("### 💵 現在のお財布の状況")
    
    daily_pnl = float(hb.get("daily_pnl", 0.0))
    trades_today = int(hb.get("trades_today", 0))
    
    # Load current cash settings
    initial_cash = 100000.0
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
                initial_cash = cfg_data.get("backtest", {}).get("initial_cash", 100000.0)
        except Exception:
            pass
            
    current_cash = initial_cash + daily_pnl
    
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 現在のお金（運用元本＋本日の損益）", f"${current_cash:,.2f}")
    
    # PnL color logic
    pnl_str = f"+${daily_pnl:,.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):,.2f}"
    c2.metric("📅 本日の損益", pnl_str, delta=daily_pnl)
    c3.metric("🔄 本日の取引回数", f"{trades_today} 回")

    # 3. Positions
    st.markdown("### 💼 現在持っている株 (保有ポジション)")
    pos = rt["current_positions"]
    if not pos:
        st.info("現在は株を持っていません（現金のみで様子見中です）。")
    else:
        for p in pos:
            direction_label = "値上がり狙い（買い）" if p.get("direction") == "BULL" else "値下がり狙い（空売り）"
            st.info(
                f"**持ち株:** {p.get('symbol')} | "
                f"**取引の方向:** {direction_label} | "
                f"**購入価格:** ${p.get('entry_price'):,.2f} | "
                f"**現在の損益割合:** {p.get('unrealized_pct')}%"
            )

    # 4. Latest AI decision
    st.markdown("### 🧠 AIが直前に下した判断")
    latest_sig = rt["latest_signal"]
    
    if not latest_sig:
        st.info("AIの判断履歴がまだありません。")
    else:
        action_translated = translate_action(latest_sig.get("action", ""))
        direction_translated = translate_direction(latest_sig.get("direction", ""))
        
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.markdown(f"**【AIの提案】**\n### {action_translated}")
            st.markdown(f"**対象銘柄:** {latest_sig.get('symbol') or 'なし'}")
            st.markdown(f"**予測の自信度:** {int(float(latest_sig.get('confidence', 0)) * 100)}%")
        with sc2:
            st.markdown("**【AIがこの判断を下した理由】**")
            st.success(latest_sig.get("reason", "理由は特に記載されていません。"))

# --- TAB 2: 📈 これまでの成績 (Performance) ---
def render_performance_tab(run: RunData | None) -> None:
    if not run:
        st.info("📊 まだ取引データがありません。今日からの取引が行われると、ここにデータが蓄積されます。")
        return

    m = run.metrics
    
    st.markdown("### 📊 トータル運用成績のまとめ")
    
    c1, c2, c3, c4 = st.columns(4)
    # Total return
    total_ret = m.get("total_return_pct", 0.0)
    c1.metric("💹 トータルの利益率", f"{total_ret}%")
    
    # Win rate
    win_rate = m.get("win_rate_pct", 0.0)
    c2.metric("🎯 勝率", f"{win_rate}%")
    
    # Profit Factor (Recovery Factor)
    pf = m.get("profit_factor", 0.0)
    c3.metric(
        "🔄 回収率 (利益÷損失)", 
        f"{pf} 倍", 
        help="これが1.0倍を超えていれば、損よりも利益の方が大きい（黒字）という意味になります。"
    )
    
    # Max Drawdown
    dd = m.get("max_drawdown_pct", 0.0)
    c4.metric(
        "⚠️ 一番大きく資産が減った時の落ち込み幅", 
        f"{dd}%",
        help="過去の運用中、一時的にどれだけお財布が凹んだかを示すリスクの目安です。"
    )

    # Simple chart
    st.markdown("---")
    st.markdown("### 📈 お財布のお金の増え方 (資産推移)")
    eq = run.equity
    if not eq.empty and "equity" in eq.columns:
        eq = eq.copy()
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
        eq = eq.set_index("timestamp")
        st.line_chart(eq["equity"])
    else:
        st.info("資産推移を描画するためのデータがまだ十分にありません。")

# --- TAB 3: 🤖 AIの判断日記 (AI Diary) ---
def render_diary_tab(reports_dir: str) -> None:
    st.markdown("### 📝 AIの判断日記（タイムライン）")
    st.caption("AIが5分ごとに考えたことや、ニュースに対して下した判断の履歴です。")
    
    rt = load_runtime(reports_dir)
    events = rt["events"]
    
    if events.empty:
        st.info("まだ日記が書かれていません。取引が始まると自動的に記録されます。")
        return
        
    # Simplify events
    # We display events chronologically (recent first)
    df_disp = events.sort_index(ascending=False).head(30)
    
    for _, row in df_disp.iterrows():
        t_str = row.get("timestamp", "")
        # Format time if possible
        try:
            t_str = pd.Timestamp(t_str).strftime("%m/%d %H:%M:%S")
        except Exception:
            pass
            
        event_type = row.get("event", "")
        msg = row.get("message", "")
        
        # Friendly icons
        icon = "ℹ️"
        if "ORDER" in event_type or "TRADE" in event_type:
            icon = "🔄 [取引]"
        elif "SIGNAL" in event_type:
            icon = "🧠 [AI判断]"
        elif "ERROR" in event_type:
            icon = "⚠️ [エラー]"
            
        st.write(f"**{t_str}** | {icon} {msg}")

# --- TAB 4: 🧬 AIの自動成長 (ABテスト状況) ---
def render_evolution_tab(reports_dir: str) -> None:
    evo = load_evolution(reports_dir)
    champ = evo["champion"]
    
    if not champ:
        st.info("🧬 自動成長システムは現在準備中です。最初の数回のデータ取得をお待ちください。")
        return

    champ_id = champ.get('champion_id', 'Base_Model')
    champ_nickname = get_nickname(champ_id)
    metrics = champ.get('metrics', {}) or {}

    st.markdown("### 🏆 現在の本命AI（正式採用中の設定）")
    c1, c2, c3 = st.columns(3)
    c1.info(f"**本命AIの愛称:**\n### {champ_nickname}")
    
    # Show return delta
    total_ret = float(metrics.get("total_return_pct", 0.0))
    c2.metric("💹 トータルの利益率", f"{total_ret:+.2f}%")
    
    win_rate = float(metrics.get("win_rate_pct", 0.0))
    c3.metric("🎯 勝率", f"{win_rate:.1f}%")

    st.markdown("---")
    st.markdown("### ⚔️ 裏側で仮想テスト中の「挑戦者AI」たち (ABテスト)")
    st.caption("最大5つの挑戦者AI（競走馬たち）が、本命の座を奪うために裏で仮想取引の競い合いをしています。")
    
    ch = evo["challengers"]
    if not ch:
        st.info("現在、対戦中の挑戦者はいません（本命AIだけで稼働中）。")
    else:
        rows = []
        for idx, c in enumerate(ch, 1):
            m = c.get("metrics", {}) or {}
            c_id = c.get("challenger_id")
            c_nickname = get_nickname(c_id)
            
            c_ret = float(m.get("total_return_pct", 0.0))
            c_win = float(m.get("win_rate_pct", 0.0))
            
            rows.append({
                "順位": f"第 {idx} 候補",
                "競走馬名 (AI愛称)": c_nickname,
                "利益率 (利回り)": f"{c_ret:+.2f}%",
                "勝率": f"{c_win:.1f}%",
                "仮想取引回数": f"{m.get('num_trades', 0)} 回",
                "テスト状態": "仮想取引中（シャドウ）" if c.get("status") == "shadow" else "最終審査中（カナリア）",
            })
        st.table(pd.DataFrame(rows))

# --- MAIN RENDERER ---
def main() -> None:
    st.title("📈 BullBear AI トレーダー (かんたん初心者画面)")
    st.caption(SAFETY_INFO)

    # Sidebar
    st.sidebar.markdown("### ⚙️ お金の元本設定")
    config_path = st.sidebar.text_input("📁 設定ファイルパス", "config/default.yaml")
    
    # Read current initial cash
    initial_cash = 100000.0
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
                initial_cash = cfg_data.get("backtest", {}).get("initial_cash", 100000.0)
        except Exception:
            pass
            
    new_cash = st.sidebar.number_input(
        "💵 運用する元本 (米ドル $)", 
        min_value=1.0, 
        value=float(initial_cash), 
        step=1000.0,
        help="取引に使う元本を設定します。ドル表記ですので、1万ドルの場合は10000と入力してください。"
    )
    
    if new_cash != initial_cash:
        if st.sidebar.button("💾 この元本で再スタート（反映して再起動）"):
            if update_initial_cash_in_yaml(config_path, new_cash):
                st.sidebar.success(f"元本を ${new_cash:,.0f} ドルに更新しました！AIランナーを再起動します。")
                restart_runner(config_path)
                st.rerun()
                
    st.sidebar.markdown("---")
    reports_dir = st.sidebar.text_input("📊 レポート出力先ディレクトリ", _reports_dir())
    
    # Load past run if any
    all_ids = list_run_ids(reports_dir)
    run = None
    if all_ids:
        run = load_run(reports_dir, "latest")

    # Beginner tabs
    tabs = st.tabs([
        "🏠 現在の状況", 
        "📈 これまでの成績", 
        "🤖 AIの判断日記", 
        "🧬 AIの自動成長 (ABテスト)"
    ])
    
    with tabs[0]:
        render_status_tab(reports_dir, config_path)
    with tabs[1]:
        render_performance_tab(run)
    with tabs[2]:
        render_diary_tab(reports_dir)
    with tabs[3]:
        render_evolution_tab(reports_dir)

if __name__ == "__main__":
    main()
