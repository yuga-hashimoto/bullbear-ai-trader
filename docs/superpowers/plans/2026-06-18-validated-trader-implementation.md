# Validated Self-Evolving Trader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current demo into a paper-operational system with risk-budget sizing, persistent OpenCode analysis state, honest validation gates, and no path to premature live trading.

**Architecture:** Introduce deterministic portfolio sizing and runtime state as script-owned services. OpenCode supplies expiring analysis only; numeric strategy and risk code remain authoritative. Historical evaluation, forward shadow, paper eligibility, and live eligibility become distinct states with minimum evidence requirements.

**Tech Stack:** Python 3.10+, pandas, dataclasses, YAML/JSON, pytest, yfinance/OpenCode-compatible HTTP API.

---

### Task 1: Make Configuration Match the Available Data and JPY Risk Contract

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/default.yaml`
- Modify: `config/synthetic.yaml`
- Test: `tests/test_settings.py`

- [ ] Add failing tests asserting a JPY base currency, 1,000,000 JPY capital, 10,000 JPY trade-risk limit, 50,000 JPY daily stop, 15% maximum drawdown, 3% portfolio-risk budget, and non-empty configured test slice.
- [ ] Run `pytest tests/test_settings.py -q` and confirm the new assertions fail.
- [ ] Add explicit account and risk-budget settings with validation.
- [ ] Change the default research dates to the available 2026 data window while preserving synthetic test configuration independence.
- [ ] Run `pytest tests/test_settings.py -q` and the full suite.

### Task 2: Add Deterministic Risk-Based Position Sizing

**Files:**
- Create: `src/risk/sizing.py`
- Modify: `src/risk/engine.py`
- Modify: `src/backtest/engine.py`
- Modify: `src/runners/paper_runner.py`
- Test: `tests/test_sizing.py`

- [ ] Write failing tests for stop-distance sizing, cash cap, portfolio-risk cap, FX conversion, missing stop rejection, and overnight gap-risk reduction.
- [ ] Confirm failures with `pytest tests/test_sizing.py -q`.
- [ ] Implement a pure `PositionSizer` returning quantity and rejection reason.
- [ ] Route both BacktestEngine and PaperRunner through the same sizing service.
- [ ] Verify targeted and full tests.

### Task 3: Correct Paper Accounting and Runtime Risk Stops

**Files:**
- Modify: `src/brokers/paper_broker.py`
- Modify: `src/runners/paper_runner.py`
- Modify: `src/backtest/portfolio.py`
- Test: `tests/test_paper_accounting.py`
- Test: `tests/test_paper_runner.py`

- [ ] Write failing tests proving entry and exit commissions are included, marked-to-market equity drives daily loss, bar counters advance, and unrealized loss can trigger emergency stop.
- [ ] Confirm failures.
- [ ] Centralize realized PnL calculation around broker fill metadata and persist peak equity/current drawdown.
- [ ] Make the runner close or reduce risk when the emergency daily stop or 15% drawdown circuit breaker fires.
- [ ] Verify targeted and full tests.

### Task 4: Make OpenCode Analysis Safe, Persistent, and Non-Ordering

**Files:**
- Create: `src/agents/analysis_schema.py`
- Create: `src/agents/news_store.py`
- Modify: `src/agents/external_agent.py`
- Modify: `src/agents/context.py`
- Test: `tests/test_opencode_analysis.py`
- Modify: `tests/test_opencode_agent.py`

- [ ] Write failing tests that no-news returns neutral analysis, seen-news IDs survive restart, expired analysis contributes nothing, malformed output fails closed, and OpenCode cannot directly issue BUY/SELL actions.
- [ ] Confirm failures.
- [ ] Implement `MarketAnalysis` with direction, confidence, validity interval, thesis, invalidation, risks, and source IDs.
- [ ] Persist news IDs and latest valid analysis under the runtime directory.
- [ ] Adapt ExternalAgentAdapter to translate analysis into a neutral agent contribution rather than direct orders.
- [ ] Verify targeted and full tests.

### Task 5: Separate Numeric Strategy from AI Fusion

**Files:**
- Create: `src/strategy/numeric.py`
- Create: `src/strategy/fusion.py`
- Modify: `src/strategy/strategy.py`
- Modify: `src/agents/factory.py`
- Test: `tests/test_signal_fusion.py`

- [ ] Write failing tests for numeric-only operation, AI-disabled equivalence, expired-AI equivalence, AI confirmation, AI conflict rejection, and bull/bear self-cancellation prevention.
- [ ] Confirm failures.
- [ ] Implement deterministic candidate scores from causal features.
- [ ] Implement fusion with configurable but bounded AI weight.
- [ ] Ensure AI alone cannot produce an order.
- [ ] Verify targeted and full tests.

### Task 6: Replace Misleading Evolution Promotion with Evidence States

**Files:**
- Modify: `src/evolution/challenger.py`
- Modify: `src/evolution/promotion_policy.py`
- Modify: `src/evolution/loop.py`
- Modify: `config/promotion_policy.yaml`
- Test: `tests/test_evolution_evidence.py`

- [ ] Write failing tests for `DRAFT → BACKTEST_PASSED → SEALED_OOS_PASSED → SHADOW → PAPER → LIVE_ELIGIBLE`, six-month paper minimum, 100 trades, PF 1.25, Sharpe 1.0, positive recent-three-month PnL, and no automatic live enablement.
- [ ] Confirm failures.
- [ ] Add evidence timestamps, periods, trial counts, and ablation metrics to challenger records.
- [ ] Block promotion when any evidence is absent.
- [ ] Keep live trading disabled even at `LIVE_ELIGIBLE`.
- [ ] Verify targeted and full tests.

### Task 7: Add Operational Doctor and Honest Status

**Files:**
- Create: `src/ops/doctor.py`
- Create: `src/ops/__init__.py`
- Modify: `src/cli.py`
- Modify: `README.md`
- Test: `tests/test_doctor.py`

- [ ] Write failing tests for data-window mismatch, missing OpenCode key, stale market data, missing FX data, unprofitable latest evaluation, and live-disabled status.
- [ ] Confirm failures.
- [ ] Add `bullbear doctor` and `bullbear readiness` JSON output.
- [ ] Correct README claims about no-news behavior, shadow meaning, paper duration, and live eligibility.
- [ ] Verify targeted and full tests.

### Task 8: Start and Verify Paper Operation

**Files:**
- Modify: `scripts/run_demo.sh`
- Create: `scripts/run_paper_once.sh`
- Modify: `README.md`

- [ ] Run the doctor against local data and record exact blockers.
- [ ] Run a deterministic one-step paper cycle using persisted runtime state.
- [ ] Run an all-data research backtest and save the resulting metrics.
- [ ] Run the full test suite and `git diff --check`.
- [ ] Confirm real-money live trading remains disabled.
- [ ] Commit the implementation in scoped commits.
