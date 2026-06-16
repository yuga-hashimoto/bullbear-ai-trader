# bullbear-ai-trader

Day-trading **backtest** infrastructure for US leveraged ETFs
(**TQQQ / SQQQ / SOXL / SOXS**). The trade-decision intelligence lives **outside**
this repo: an external Agent (OpenClaw / HermesAgent / …) proposes trades; this
repo supplies data, features, market context, signal validation, the Risk
Engine, execution simulation, reporting and a read-only dashboard.

> ⚠️ **これは投資助言ではありません / This is NOT investment advice.** 研究・教育用です。
>
> 🔒 **バックテスト専用。実売買はデフォルトで完全に無効。** Live trading is disabled by
> default behind a triple safety gate (see [Live trading safety](#live-trading-safety)).
>
> 🧠 **売買判断は外部Agentに委譲。** The Agent only *proposes* via a standard Signal
> JSON. **It never places orders.** Every signal must pass the Risk Engine, which
> is authoritative — a rejected signal is never traded. NO_TRADE is the default.

---

## 設計思想 / Design

```
Data → Features → Agent Context → Agent Signal → Signal Validation
     → Risk Engine (authoritative) → Backtest Execution → Report / Dashboard
```

- **Agent はこのリポジトリの外**。OpenClaw / HermesAgent が複数モデル・外部ツールに接続して
  判断し、標準化された **Signal JSON** だけを返す。自然言語レスポンスは売買に使わない。
- **Risk Engine が最終権限**。confidence 閾値・許可銘柄・family整合性・損切り/利確/連敗停止/
  日次最大損失/大引け強制決済などを強制。Agentがどれだけ強く主張しても拒否すれば取引しない。
- 既存の ML モデル（LightGBM等）は**主経路から外し**、`LocalModelAgent` という *1つのAgent実装*
  として残してある（差し替え可能なベンチマーク用）。

## アーキテクチャ / Architecture

```
src/
  agents/      ★ Agent層: signal_schema, context, base, mock/replay/external/local_model, factory
  config/      設定 + ライブ取引セーフティゲート
  data/        DataSource interface + yfinance/moomoo/synthetic アダプタ, クリーニング, 保存
  features/    causal なテクニカル指標 + 特徴量行列（全銘柄の price+VWAP も保持）
  labeling/    将来リターンの UP/DOWN/FLAT（LocalModelAgent 学習用; 同日内のみ）
  models/      DirectionModel interface + LightGBM / sklearn（LocalModelAgent が利用）
  strategy/    Signal → 売買意図(ENTER/EXIT/NONE)の薄い写像（AIなし）
  risk/        Risk Engine（Signal検証 + 既存リスク制御; 権限は最上位）
  backtest/    実行モデル(コスト)・Agent駆動エンジン・指標・ウォークフォワード
  brokers/     BrokerBase + PaperBroker/BacktestBroker（動作）+ MoomooBroker（無効スケルトン）
  reports/     report(md/html), runs(run_id保存), loader, dashboard(Streamlit)
  utils/       ロギング, 米国市場時間ユーティリティ
tests/         schema / agent / context-leak / risk / engine / runs / dashboard
config/        default.yaml（実データ）, synthetic.yaml（オフラインデモ/テスト）
```

データ取得元（`data_source`）と Agent（`agent.type`）はどちらも interface/factory で差し替え可能。

## セットアップ / Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # streamlit はダッシュボード用（任意）
cp .env.example .env                    # 認証情報はコミットしない（.env は .gitignore 済み）
```

## クイックスタート（オフライン・ネットワーク不要）

合成データ + MockAgent で fetch → features → backtest → replay → validate → list を実行:

```bash
PYTHON=.venv/bin/python bash scripts/run_demo.sh
```

> MockAgent は**勝つためのAIではありません**。バックテスト基盤の動作確認用の単純ルールです。
> 合成データはランダムウォークで、コストの分だけ負けます（リーク・手数料漏れで偽の好成績が
> 出ていないことの健全性チェックを兼ねます）。

## CLI

```bash
# データ取得 / 特徴量
python -m src.cli fetch-data     --config config/default.yaml --symbols TQQQ SQQQ SOXL SOXS QQQ SMH SPY --interval 5m
python -m src.cli build-features --config config/default.yaml

# バックテスト（Agentを選択）
python -m src.cli backtest --config config/default.yaml --agent mock
python -m src.cli backtest --config config/default.yaml --agent replay --signals data/signals/sample.jsonl

# 外部Agent(OpenClaw/HermesAgent)の出力を JSONL に貯めて replay で検証する想定
python -m src.cli validate-signals --signals data/signals/sample.jsonl

# 一括（fetch→features→[train if local_model]→backtest）
python -m src.cli run-pipeline --config config/default.yaml --agent mock

# レポート / ダッシュボード / run管理
python -m src.cli report       --config config/default.yaml --run-id latest
python -m src.cli list-runs    --config config/default.yaml
python -m src.cli compare-runs --config config/default.yaml --run-ids <id1> <id2>
python -m src.cli serve-report --config config/default.yaml --run-id latest   # Streamlit起動

# LocalModelAgent を使う場合のみ ML を学習（主経路ではない補助機能）
python -m src.cli train --config config/default.yaml
python -m src.cli backtest --config config/default.yaml --agent local_model
```

## Signal JSON 仕様

Agent はこの形式のみを返す（自然言語は売買に使わない）。`pydantic`不要の dataclass+検証
（`src/agents/signal_schema.py`）。

```json
{
  "timestamp": "2026-01-01T14:35:00-05:00",
  "agent_name": "HermesAgent",
  "agent_version": "string",
  "target_family": "NASDAQ | SEMICONDUCTOR | MARKET",
  "direction": "UP | DOWN | FLAT",
  "action": "BUY_BULL | BUY_BEAR | NO_TRADE | EXIT",
  "symbol": "TQQQ | SQQQ | SOXL | SOXS | null",
  "confidence": 0.0,
  "expected_holding_minutes": 30,
  "reason": "string",
  "risk_notes": [],
  "features_used": {},
  "raw_response": {}
}
```

ルール: `BUY_BULL`→{TQQQ,SOXL}、`BUY_BEAR`→{SQQQ,SOXS}、family と symbol は整合必須
（NASDAQ→TQQQ/SQQQ, SEMICONDUCTOR→SOXL/SOXS）、`NO_TRADE`→symbol は null、`EXIT`→既存ポジ
を閉じる、`confidence`∈[0,1]。不正なSignalはすべて拒否（→NO_TRADE 扱い・カウント）。

## Risk Engine 仕様

Signal を受けて **ACCEPT / REJECT / FORCE_EXIT** を判定（`src/risk/engine.py`）:

- Signalスキーマ検証 / `confidence_threshold` / 許可銘柄チェック / 許可action / family-symbol整合
- 1トレード最大損失・利確・トレーリング・1日最大損失（連敗・日次でHALT）
- 最大取引回数/日・最大保有時間・寄付直後禁止・引け前新規禁止・大引け前強制決済
- 既存ポジション中の追加エントリー禁止・同一銘柄クールダウン
- すべての拒否理由をログ（`risk_decisions.jsonl` / レポート）に残す

## run 出力構成 / Report output

バックテスト実行ごとに `run_id`（`YYYY-MM-DD_HHMMSS`）でディレクトリを作成:

```
reports/
  latest.json            # {"run_id": "..."}（latest シンボリックリンクも作成）
  runs/<run_id>/
    config.yaml           # 設定スナップショット
    summary.json          # overview（期間・銘柄・headline metrics）
    metrics.json          # 全指標 + benchmark + counters
    trades.csv            # 全トレードログ（trade_id, 建値/決済値, pnl, 保有時間, entry/exit理由）
    daily_pnl.csv         # 日次損益
    equity_curve.csv      # 資産推移
    agent_signals.jsonl   # 全Signal（+ accepted / rejection_reason / trade_id）
    risk_decisions.jsonl  # 全リスク判定（+ risk_state, daily_pnl, ポジション）
    report.md / report.html
```

- **trades.csv**: 実際に約定したトレード。`trade_id` で signals / risk decisions と紐づく。
- **agent_signals.jsonl**: Agentが各バーで何を提案したか（NO_TRADE含む）と採否・理由。
- **risk_decisions.jsonl**: Risk Engineがなぜ許可/拒否/強制決済したか（その時点の risk_state 付き）。

## Web ダッシュボード（閲覧専用）/ Dashboard

```bash
streamlit run src/reports/dashboard.py
# レポート先を指定する場合:
BULLBEAR_REPORTS_DIR=reports streamlit run src/reports/dashboard.py
```

> 🔒 このUIは**バックテスト結果の閲覧専用**です。注文ボタン・live trading開始・Broker操作は
> 一切ありません。実売買はできません。

画面:
- **Overview**: run_id / 実行日時 / 銘柄 / 期間 / interval / Agent種別 / 初期資金 / 総リターン /
  最大DD / 勝率 / profit factor / 取引回数 / no-trade ratio / rejected / forced exits。
- **Charts**: equity curve / drawdown / daily PnL / 累積PnL / symbol別PnL / action別件数 /
  confidence分布 / risk rejection reasons / benchmark比較（QQQ・TQQQ・SOXL の buy&hold / cash）。
- **Trades**: 一覧（symbol/direction/exit_reason フィルタ）+ `trade_id` 選択で詳細 +
  関連 Agent Signal / Risk Decision を絞り込み表示。
- **Agent Signals**: timestamp/agent/family/direction/action/symbol/confidence/reason/risk_notes/
  accepted/rejection_reason をフィルタ（action/direction/symbol/accepted/reason/confidence範囲）。
- **Risk Decisions**: ACCEPT/REJECT/FORCE_EXIT・理由・risk_state をフィルタ表示。
- **Compare**: 複数 run を選択 → metrics 横並び比較・equity curve 重ね描き・設定差分。

## 複数 run の比較 / Comparing runs

CLI: `python -m src.cli compare-runs --run-ids <id1> <id2>`、または Dashboard の **Compare** タブ。
`list-runs` で run_id 一覧を確認できます。

## Paper Trading Runner（米国市場時間中の常時稼働）

`PaperRunner` は**米国市場の通常取引時間中だけ**バー単位で動き続け、時間外はスリープします。
**PaperBroker のみ**を使い、実注文は一切しません。Agentは提案、Risk Engineが最終判断、
拒否されたSignalは OrderIntent を生成しません。

```bash
python -m src.cli run-paper     --config config/default.yaml --agent mock
python -m src.cli run-paper     --config config/default.yaml --agent external   # 未設定時はNO_TRADE/fallback
python -m src.cli runner-status --config config/default.yaml      # heartbeat.json を表示
python -m src.cli stop-runner   --config config/default.yaml      # 次ループで安全停止
python -m src.cli run-live      --config config/default.yaml --enable-live-trading  # 必ず拒否される
```

- **取引時間判定は必ず America/New_York 基準**（09:30〜16:00, 月〜金）。日本時間は夏/冬で
  変わる（夏 ≈22:30〜翌5:00 / 冬 ≈23:30〜翌6:00）ため、実装は常にET基準。`pandas_market_calendars`
  (XNYS) で**祝日・短縮取引日**に対応（短縮日は早めにforce close）。pre-market/after-hours は
  初期では対象外（`runner.extended_hours_enabled` で将来拡張の余地のみ確保）。
- **バー境界に揃えてスリープ**し、同一barの二重処理を防止。stale data / Agent timeout は NO_TRADE、
  data/agent エラー多発・heartbeat書込失敗・想定外例外では position を保存して安全停止。
  引け前 force close、市場クローズ後にpositionが残れば警告。
- 安全停止: `max_daily_loss` / `max_consecutive_losses` 到達でその日は `DAILY_STOP`。

**Heartbeat の保存場所**: `reports/runtime/heartbeat.json`（`runner-status` で表示）。
そのほかのランタイム成果物: `reports/runtime/` 配下に `paper_events.jsonl`（全イベント）、
`errors.jsonl`、`current_positions.json`、`daily_state.json`、`latest_signal.json`、
`latest_risk_decision.json`。

**Runtime ダッシュボード**: `streamlit run src/reports/dashboard.py` の **Runtime (Paper)** タブで
runner状態 / market state / セッション時刻 / next open・close / 最終処理バー / 最新Signal /
最新Risk判定 / 現在のpaperポジション / Daily PnL / trades today / 連敗 / 直近イベント・エラー /
`DAILY_STOP` を確認できます（**閲覧専用。注文・live開始ボタンは存在しません**）。

> ⚠️ **PCがスリープすると runner も止まります。** 常時稼働させるなら Docker / systemd /
> macOS launchd / VPS 等での常駐を検討してください。**実売買は初期状態で無効**です。

## 自動進化 / Champion・Challenger・Shadow・Canary・Auto-Promotion

PaperRunner が常時稼働しながら、複数の Challenger（戦略/リスク/閾値の変種）を **同じ市場データ**で
並行評価し、**人間承認なし**で良いものを Champion に自動昇格できます。ただし**必ず**事前定義の
Promotion Policy / Guardrail / Rollback Rule を満たした場合のみ。

- **Champion**: 現在の正式採用設定（`reports/registry/champion.yaml`）。PaperRunner（将来のLiveRunnerも）
  が既定で使用。
- **Challenger**: 改善候補（`config_patch` で risk/strategy/agent のみ変更可能）。`SHADOW`→`CANARY`→
  `PROMOTED` と遷移。
- **Shadow evaluation**: Challenger を **ShadowBroker相当（仮想約定・資金不使用）**で評価。手数料・
  スプレッド・スリッページは Champion と同条件で反映（`reports/evolution/shadow_pnl.jsonl`）。
- **Canary**: Shadow優秀なものを小配分で実運用に近い条件で検証（Paperは仮想配分。LiveCanaryでも
  必ず Risk Engine を通す）。
- **Allocator**: 同一口座で矛盾注文が出ないよう最終注文を1つに統合。Champion が実口座を駆動し、
  Challenger は Shadow。逆方向の競合は `rejected_conflicts` に記録。
- **Adaptive allocation (bandit)**: epsilon-greedy で成績の良い Challenger に配分を寄せる。drawdown が
  大きいと配分↓、`min_trades` 未満は増やさない、`max_challenger_allocation_pct`(=30%) を超えない。
  **Liveでは初期無効**。

**Auto-Promotion（`config/promotion_policy.yaml`）— 勝率だけでは絶対に昇格しません。**
win rate / expectancy / profit factor / max drawdown / worst day / trade数 / robustness（過学習リスク）/
out-of-sample を**総合評価**し、全条件を満たした時のみ昇格。`environment_allowed.live: false`（Live自動昇格は初期無効）。
**Champion 切替は原則として市場クローズ後のみ。**

**Auto-Rollback（`config/rollback_policy.yaml`）— 市場中でも即時。** 日次損失・intraday drawdown・連敗・
Fallback比劣後・Agent/data エラー多発で旧 Champion（fallback）へ自動復帰し、一定期間 promotion を凍結。

**Mutation Generator** は安全な範囲でのみ候補を自動生成。**ガードレールが昇格・変異を統制**:
`max_daily_loss` 等の安全限界を緩める変更、fee/slippage/spread を都合よく下げる変更、live ゲートを
弱める変更、Risk Engine を迂回する変更は**すべて拒否**。`apply_patch` は risk/strategy/agent 以外の
セクション（costs/trading/market/runner…）を**構造的に変更不可**。

**Drift Detection**: win rate / expectancy 劣化、drawdown 加速、volatility regime 変化を検出し探索を増やす。

CLI:
```bash
python -m src.cli champion              --config config/default.yaml
python -m src.cli list-challengers      --config config/default.yaml
python -m src.cli create-challenger     --config config/default.yaml --from-run latest
python -m src.cli generate-mutations    --config config/default.yaml --run-id latest
python -m src.cli update-allocations    --config config/default.yaml
python -m src.cli evaluate-promotions   --config config/default.yaml
python -m src.cli auto-promote          --config config/default.yaml --env paper
python -m src.cli rollback-champion     --config config/default.yaml
python -m src.cli run-evolution         --config config/default.yaml --env paper
python -m src.cli evolution-status      --config config/default.yaml
```

**Evolution ダッシュボード**: `dashboard.py` の **Evolution** タブで Current/Previous Champion、
Active Challengers、Shadow/Canary成績、配分、Promotion候補と policy 合否、Rollback状況、
bandit配分履歴、Mutation履歴、Drift alerts、自動昇格/ロールバック履歴を確認（**閲覧専用**）。

> 🔒 **自動進化しても Risk Engine は迂回できません。** 自動昇格は Promotion Policy 充足時のみ、
> 安全限界を弱める自動変更は禁止、実売買は初期状態で無効、moomoo本番注文は呼びません。

## リスク / Risks

レバレッジETFは減価・高ボラ・ギャップで急変。バックテストと実運用は乖離。約定/スプレッド/
スリッページは悲観的近似。MockAgent/合成データの結果に投資的意味はありません。

## Live trading safety

実売買は **三重の独立スイッチが揃わない限り発動しない**（`assert_live_trading_allowed`）:
1. config `trading.live_trading_enabled: true`
2. 環境変数 `BULLBEAR_ALLOW_LIVE=1`
3. 明示フラグ（`explicit_flag=True` / `allow_live=True`）

`MoomooBroker` は構築時にこのゲートを通し、かつ全注文メソッドが現状 `NotImplementedError`。
backtest/paper は一切このゲートを呼ばない。CLI・Web UI に実発注機能は存在しない。

## 今後 moomoo OpenAPI と接続する方針

- 履歴データ: `src/data/moomoo_source.py`（OpenD経由・READ-ONLY）に `request_history_kline` を実装。
- 発注: `src/brokers/moomoo.py` に実装。ただし三重ゲート + paper検証 + 上限/サーキットブレーカ +
  二重発注防止 + レビューを経てからのみ。認証情報は `.env` のみ（`.env.example` を雛形に）。
- 外部Agent接続: `src/agents/external_agent.py` の transport（http/command/mcp/file/stdio/websocket）
  を実装。未設定時は接続せず、`fallback_to_no_trade` 時のみ Mock にフォールバック。

## テスト / Tests

```bash
.venv/bin/python -m pytest -q
```

カバー: Signal正常/異常系検証・不正symbol/action/confidence/family拒否・MockAgent・ReplayAgent
(JSONL/該当なしNO_TRADE)・Agent Contextに未来データが入らない・Risk拒否で注文が出ない・NO_TRADE
で注文なし・特徴量リーク無し・ラベル正当性・バックテスト再現性・live無効・run出力ファイル生成・
dashboardデータローダ（空/NO_TRADEのみ/rejectedのみで落ちない）・dashboard描画スモーク。

## 今後の課題 / Roadmap

- OpenClaw/HermesAgent との実 transport 実装（現状スケルトン）
- moomoo OpenD 履歴/発注の実装（現状無効スケルトン）
- ウォークフォワードのCLI配線（生成ロジックは実装済み）/ 最大2ポジション拡張
- ダッシュボードの高度化（Plotly化・trade単位の market context スナップショット保存）
```
