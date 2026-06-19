# Paper Runtime Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the known P0 failures that invalidate real-data backtests and cause the launchd-managed PaperRunner to report false health or restart continuously.

**Architecture:** Feature preparation will explicitly remove unusable all-NaN features and persist a health report. Paper data freshness will be measured from bar close plus configurable vendor delay, while service stop state will use a persistent disable flag. Readiness will derive running state from heartbeat semantics and recent runtime events rather than heartbeat age alone.

**Tech Stack:** Python 3.10+, pandas, pytest, YAML/JSON, macOS launchd.

---

### Task 1: Feature Health and Usable Feature Selection

**Files:**
- Modify: `src/features/builder.py`
- Modify: `src/backtest/engine.py`
- Modify: `src/runners/paper_runner.py`
- Modify: `src/pipeline.py`
- Test: `tests/test_backtest.py`
- Test: `tests/test_paper_runner.py`
- Create: `tests/test_feature_health.py`

- [ ] Add a failing test proving zero-volume context symbols do not create VWAP/volume features that are NaN for every row.
- [ ] Add a failing test proving an unrelated all-NaN feature column cannot empty a valid backtest matrix.
- [ ] Add a failing test proving `build-features` writes `feature_health_report.json` with dropped columns and NaN ratios.
- [ ] Run the targeted tests and confirm they fail for the missing behavior.
- [ ] Implement `prepare_feature_matrix()` to drop all-NaN feature columns before warmup row removal.
- [ ] Skip volume-dependent features when a symbol has no positive volume.
- [ ] Persist the feature health report beside the feature parquet.
- [ ] Route BacktestEngine and PaperRunner through the common preparation helper.
- [ ] Run targeted tests and the full suite.

### Task 2: Bar-Close and Vendor-Delay Freshness

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/default.yaml`
- Modify: `config/synthetic.yaml`
- Modify: `src/runners/paper_runner.py`
- Test: `tests/test_settings.py`
- Test: `tests/test_paper_runner.py`

- [ ] Add failing tests for `vendor_delay_seconds`, a 09:30 five-minute bar accepted at 09:39, and a genuinely old bar rejected.
- [ ] Run the targeted tests and confirm the 09:39 case fails under timestamp-start freshness.
- [ ] Compute freshness from `last_bar_time + interval`; apply vendor delay and stale threshold.
- [ ] Keep stale handling in NO_TRADE/wait state without calling `_safe_stop`.
- [ ] Write heartbeat status `waiting` with reason `stale_data`.
- [ ] Run targeted tests and the full suite.

### Task 3: Persistent Service Disable and Explicit Resume

**Files:**
- Modify: `src/runners/base.py`
- Modify: `src/runners/paper_runner.py`
- Modify: `src/cli.py`
- Modify: `deploy/run-paper-launchd.sh`
- Modify: `README.md`
- Test: `tests/test_paper_runner.py`
- Create: `tests/test_runner_control.py`

- [ ] Add failing tests proving a stop request survives runner startup and explicit resume clears it.
- [ ] Add a failing CLI parser test for `start-runner`.
- [ ] Run targeted tests and confirm failure.
- [ ] Replace transient startup-cleared stop behavior with `disable.flag`.
- [ ] Make `stop-runner` persist disable state and `start-runner` clear it.
- [ ] Make the launchd wrapper wait while disabled instead of repeatedly exec/restarting.
- [ ] Run targeted tests and the full suite.

### Task 4: Honest Readiness and Restart-Loop Detection

**Files:**
- Modify: `src/ops/doctor.py`
- Test: `tests/test_doctor.py`

- [ ] Add failing tests proving `status=error`, non-empty reason, recent errors, restart loops, and missing processed bars during an open market all make `paper.running` false.
- [ ] Run targeted tests and confirm failure.
- [ ] Parse recent runtime events and heartbeat fields into explicit checks.
- [ ] Require fresh heartbeat, `status == running`, empty reason, no recent errors, no restart loop, and a processed bar while open.
- [ ] Keep market-closed sleeping heartbeat valid without requiring a processed bar.
- [ ] Run targeted tests and the full suite.

### Task 5: Collision-Free Runs and Synthetic Isolation

**Files:**
- Modify: `src/reports/runs.py`
- Modify: `config/synthetic.yaml`
- Modify: `scripts/run_demo.sh`
- Test: `tests/test_runs.py`
- Test: `tests/test_settings.py`

- [ ] Add a failing test proving two IDs generated within the same second differ.
- [ ] Add failing config assertions proving all synthetic paths are isolated.
- [ ] Run targeted tests and confirm failure.
- [ ] Add microseconds to `new_run_id()`.
- [ ] Move synthetic raw/features/artifacts/reports/signals under synthetic-only directories.
- [ ] Update the demo to resolve and replay runs from configured synthetic paths.
- [ ] Run targeted tests and the full suite.

### Task 6: Runtime Verification

**Files:**
- No production changes expected.

- [ ] Run `.venv/bin/python -m pytest -q`.
- [ ] Run `git diff --check`.
- [ ] Run the default real-data `build-features` and verify the health report.
- [ ] Run a default backtest and verify it no longer fails from VIX all-NaN features.
- [ ] Install the updated launchd wrapper/plist, restart the service, and inspect heartbeat/events.
- [ ] Run `doctor` and report exact remaining readiness blockers.
- [ ] Confirm live trading remains disabled.
