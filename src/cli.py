"""Command-line interface (backtest / research only — never places orders).

    python -m src.cli fetch-data     --config config/default.yaml [--symbols ...] [--interval 5m]
    python -m src.cli build-features --config config/default.yaml
    python -m src.cli train          --config config/default.yaml      # LocalModelAgent helper
    python -m src.cli backtest       --config config/default.yaml --agent mock
    python -m src.cli backtest       --config config/default.yaml --agent replay --signals data/signals/sample.jsonl
    python -m src.cli run-pipeline   --config config/default.yaml --agent mock
    python -m src.cli validate-signals --signals data/signals/sample.jsonl
    python -m src.cli report         --config config/default.yaml --run-id latest
    python -m src.cli serve-report   --config config/default.yaml --run-id latest
    python -m src.cli list-runs      --config config/default.yaml
    python -m src.cli compare-runs   --config config/default.yaml --run-ids r1 r2
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .config.settings import load_config
from .pipeline import backtest, build_features, fetch_data, run_pipeline, train
from .utils.logging import get_logger, setup_logging

log = get_logger("cli")


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="config/default.yaml", help="path to YAML config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bullbear", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-data", help="download & store OHLCV")
    _add_config_arg(p_fetch)
    p_fetch.add_argument("--symbols", nargs="*", default=None)
    p_fetch.add_argument("--interval", default=None, choices=["1m", "5m", "15m"])

    for name in ("build-features", "train"):
        _add_config_arg(sub.add_parser(name))

    for name in ("backtest", "run-pipeline"):
        p = sub.add_parser(name)
        _add_config_arg(p)
        p.add_argument("--agent", default=None,
                       choices=["mock", "replay", "external", "local_model"])
        p.add_argument("--signals", default=None, help="JSONL signal file for replay agent")

    p_val = sub.add_parser("validate-signals", help="validate a signal JSONL file")
    p_val.add_argument("--signals", required=True)

    # Continuous runners.
    p_paper = sub.add_parser("run-paper", help="run the always-on paper trading runner")
    _add_config_arg(p_paper)
    p_paper.add_argument("--agent", default=None,
                         choices=["mock", "replay", "external", "local_model"])
    p_paper.add_argument("--signals", default=None)

    _add_config_arg(sub.add_parser("runner-status"))
    _add_config_arg(sub.add_parser("stop-runner"))
    _add_config_arg(sub.add_parser("start-runner"))
    _add_config_arg(sub.add_parser("doctor"))
    _add_config_arg(sub.add_parser("readiness"))

    p_live = sub.add_parser("run-live", help="(disabled) future live runner")
    _add_config_arg(p_live)
    p_live.add_argument("--enable-live-trading", action="store_true")

    # Evolution (Champion/Challenger/Shadow/Canary/Auto-promotion).
    for name in ("champion", "list-challengers", "evaluate-promotions",
                 "rollback-champion", "evolution-status", "update-allocations"):
        _add_config_arg(sub.add_parser(name))
    p_cc = sub.add_parser("create-challenger")
    _add_config_arg(p_cc)
    p_cc.add_argument("--from-run", default="latest")
    p_cc.add_argument("--seed", type=int, default=0)
    p_gm = sub.add_parser("generate-mutations")
    _add_config_arg(p_gm)
    p_gm.add_argument("--run-id", default="latest")
    p_gm.add_argument("--seed", type=int, default=0)
    for name in ("auto-promote", "run-evolution"):
        p = sub.add_parser(name)
        _add_config_arg(p)
        p.add_argument("--env", default="paper", choices=["paper", "live"])
        p.add_argument("--agent", default=None,
                       choices=["mock", "replay", "external", "local_model"])
        p.add_argument("--signals", default=None)

    p_le = sub.add_parser("run-live-evolution",
                          help="daily judge on live shadow track records (retire/keep/spawn)")
    _add_config_arg(p_le)
    p_le.add_argument("--env", default="paper", choices=["paper", "live"])

    p_dm = sub.add_parser("run-dual-momentum",
                          help="monthly dual-momentum recommendation + paper equity")
    _add_config_arg(p_dm)
    p_dm.add_argument("--leverage", type=float, default=1.5)
    p_dm.add_argument("--capital", type=float, default=1_000_000.0)

    for name in ("report", "serve-report"):
        p = sub.add_parser(name)
        _add_config_arg(p)
        p.add_argument("--run-id", default="latest")

    _add_config_arg(sub.add_parser("list-runs"))

    p_cmp = sub.add_parser("compare-runs")
    _add_config_arg(p_cmp)
    p_cmp.add_argument("--run-ids", nargs="+", required=True)
    return parser


def _handle_evolution(args, cfg) -> int:
    import random

    from .evolution.experiment_store import ExperimentStore
    from .evolution.registry import EvolutionRegistry

    reports_dir = cfg.path("reports_dir")
    registry = EvolutionRegistry(reports_dir)
    store = ExperimentStore(reports_dir)

    if args.command == "champion":
        print(json.dumps(registry.ensure_champion().to_dict(), indent=2, default=str))
        return 0

    if args.command == "list-challengers":
        print(json.dumps([c.to_dict() for c in registry.list_challengers()], indent=2, default=str))
        return 0

    if args.command == "create-challenger":
        from .evolution.mutation_generator import propose_patch

        registry.ensure_champion()
        patch = propose_patch(cfg, random.Random(args.seed))
        chal = registry.create_challenger(patch, source="manual",
                                          notes=f"from-run={args.from_run}")
        from .evolution.experiment_store import EvolutionEventType
        store.emit(EvolutionEventType.CHALLENGER_CREATED, {"challenger_id": chal.challenger_id})
        print(json.dumps(chal.to_dict(), indent=2, default=str))
        return 0

    if args.command == "generate-mutations":
        from .evolution.mutation_generator import generate_mutations

        cands = generate_mutations(cfg, n=cfg.raw.get("evolution", {}).get("mutations_per_day", 3),
                                   seed=args.seed)
        print(json.dumps({"candidates": cands}, indent=2, default=str))
        return 0

    if args.command == "update-allocations":
        from .evolution.bandit import ArmStats, Bandit

        ad = cfg.raw.get("adaptive_allocation", {})
        bandit = Bandit(mode=ad.get("mode", "epsilon_greedy"), epsilon=ad.get("epsilon", 0.1),
                        max_challenger_allocation_pct=ad.get("max_challenger_allocation_pct", 30),
                        min_trades=1)
        arms = [ArmStats(c.challenger_id, c.metrics.get("expectancy", 0.0),
                         c.metrics.get("num_trades", 0), c.metrics.get("max_drawdown_pct", 0.0))
                for c in registry.list_challengers()]
        alloc = bandit.allocate(arms)
        for c in registry.list_challengers():
            if c.challenger_id in alloc:
                c.allocation_pct = alloc[c.challenger_id]
                registry.update_challenger(c)
        registry.record_allocation({"allocations": alloc})
        print(json.dumps({"allocations": alloc,
                          "champion_pct": Bandit.champion_allocation(alloc)}, indent=2))
        return 0

    if args.command == "rollback-champion":
        from .evolution.loop import rollback_to_fallback

        ok = rollback_to_fallback(registry, store, ["manual"])
        print(json.dumps({"rolled_back": ok}, indent=2))
        return 0

    if args.command in ("evaluate-promotions", "auto-promote", "run-evolution"):
        from .evolution.loop import run_evolution_cycle

        env = getattr(args, "env", "paper")
        after_close = args.command in ("auto-promote", "run-evolution")
        status = run_evolution_cycle(cfg, env=env, agent_type=getattr(args, "agent", None),
                                     signal_file=getattr(args, "signals", None),
                                     after_close=after_close)
        print(json.dumps(status, indent=2, default=str))
        return 0

    if args.command == "evolution-status":
        out = {
            "champion": registry.ensure_champion().to_dict(),
            "challengers": [c.to_dict() for c in registry.list_challengers()],
            "status": store.read_status(),
            "recent_events": store.read_events(20),
            "promotions": registry.promotions()[-10:],
            "rollbacks": registry.rollbacks()[-10:],
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "validate-signals":
        from .agents.replay_agent import validate_signal_file

        print(json.dumps(validate_signal_file(args.signals), indent=2))
        return 0

    cfg = load_config(args.config)
    setup_logging(cfg.log_level)

    if args.command == "fetch-data":
        if args.interval:
            cfg = replace(cfg, interval=args.interval)
        print(json.dumps({"fetched": fetch_data(cfg, args.symbols)}, indent=2))
        return 0

    if args.command == "build-features":
        print(json.dumps({"features": str(build_features(cfg))}, indent=2))
        return 0

    if args.command == "train":
        artifacts = train(cfg)
        print(json.dumps({"models": {k: str(v) for k, v in artifacts.items()}}, indent=2))
        return 0

    if args.command == "backtest":
        print(json.dumps(backtest(cfg, args.agent, args.signals), indent=2, default=str))
        return 0

    if args.command == "run-pipeline":
        print(json.dumps(run_pipeline(cfg, args.agent, args.signals), indent=2, default=str))
        return 0

    if args.command == "list-runs":
        from .reports.runs import list_runs

        print(json.dumps({"runs": list_runs(cfg)}, indent=2))
        return 0

    if args.command == "report":
        from .reports.loader import load_run

        run = load_run(cfg.path("reports_dir"), args.run_id)
        print(json.dumps({"run_id": run.run_id, "summary": run.summary,
                          "metrics": run.metrics, "benchmark": run.benchmark}, indent=2, default=str))
        return 0

    if args.command == "compare-runs":
        from .reports.loader import load_run
        import pandas as pd

        runs = [load_run(cfg.path("reports_dir"), rid) for rid in args.run_ids]
        table = pd.DataFrame({r.run_id: r.metrics for r in runs})
        print(table.to_string())
        return 0

    if args.command == "run-paper":
        from .agents.factory import make_agent
        from .runners.paper_runner import PaperRunner

        agent = make_agent(cfg, args.agent, args.signals)
        log.info("starting PaperRunner (broker=paper, live disabled). Ctrl+C to stop.")
        PaperRunner(cfg, agent).run()
        return 0

    if args.command == "runner-status":
        from .runners.base import runtime_dir
        from .runners.heartbeat import RuntimeWriter

        hb = RuntimeWriter(runtime_dir(cfg)).read_heartbeat()
        if hb is None:
            print(json.dumps({"status": "no heartbeat found"}, indent=2))
            return 1
        print(json.dumps(hb, indent=2, default=str))
        return 0

    if args.command == "stop-runner":
        from .runners.base import request_stop

        flag = request_stop(cfg)
        print(json.dumps({"disable_flag": str(flag),
                          "note": "runner will exit and remain disabled"}, indent=2))
        return 0

    if args.command == "start-runner":
        from .runners.base import request_start

        flag = request_start(cfg)
        print(json.dumps({"disable_flag": str(flag),
                          "note": "runner is enabled; launchd will start it"}, indent=2))
        return 0

    if args.command in ("doctor", "readiness"):
        from .ops.doctor import build_readiness

        result = build_readiness(cfg)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["paper"]["ready"] else 1

    if args.command == "run-live":
        from .config.settings import LiveTradingDisabledError
        from .runners.live_runner import LiveRunner

        try:
            LiveRunner(cfg, enable_live_trading=args.enable_live_trading).run()
        except LiveTradingDisabledError as exc:
            print(json.dumps({"refused": str(exc),
                              "hint": "live trading is disabled by design in this build"}, indent=2))
            return 2
        except NotImplementedError as exc:
            print(json.dumps({"refused": str(exc)}, indent=2))
            return 2
        return 0

    if args.command in ("champion", "list-challengers", "evaluate-promotions",
                        "rollback-champion", "evolution-status", "update-allocations",
                        "create-challenger", "generate-mutations", "auto-promote",
                        "run-evolution"):
        return _handle_evolution(args, cfg)

    if args.command == "run-live-evolution":
        from .evolution.live_evolution import run_live_evolution_cycle

        status = run_live_evolution_cycle(cfg, env=args.env)
        print(json.dumps(status, indent=2, default=str))
        return 0

    if args.command == "run-dual-momentum":
        from .runners.dual_momentum_runner import DualMomentumRunner
        from .strategy.dual_momentum import DualMomentumConfig

        runner = DualMomentumRunner(
            reports_dir=cfg.path("reports_dir"),
            cfg=DualMomentumConfig(leverage=args.leverage),
            capital=args.capital,
        )
        print(json.dumps(runner.run(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "serve-report":
        from .reports.runs import resolve_run_id

        rid = resolve_run_id(cfg, args.run_id)
        cmd = "streamlit run src/reports/dashboard.py"
        print(f"Resolved run: {rid}")
        print(f"Launch the dashboard with:\n    BULLBEAR_REPORTS_DIR={cfg.paths['reports_dir']} {cmd}")
        try:
            import streamlit  # noqa: F401
            import subprocess

            print("Launching Streamlit...")
            env_cmd = ["streamlit", "run", "src/reports/dashboard.py"]
            subprocess.run(env_cmd, check=False)
        except ImportError:
            print("(streamlit not installed: `pip install streamlit` then run the command above.)")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
