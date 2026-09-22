"""steerdb command line: collect | oracle-gap | train | run | bench | online."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .arms import ARMS, get_arm, parse_arm_ids


def _queries(args):
    from .workload import load_workload

    qs = load_workload(args.workload)
    if getattr(args, "queries", None):
        wanted = set(args.queries.split(","))
        qs = [q for q in qs if q.name in wanted]
    return qs


def _store(args):
    from .store import ExperienceStore

    return ExperienceStore(args.store)


def cmd_collect(args) -> None:
    from .collect import collect
    from .db import connect

    arms = tuple(get_arm(a) for a in parse_arm_ids(args.arms))
    with connect(args.dsn) as conn:
        collect(conn, _store(args), _queries(args), arms, args.warmup, args.runs, args.timeout_ms)


def cmd_oracle_gap(args) -> None:
    from .evaluate import oracle_gap

    store = _store(args)
    gap = oracle_gap(store.bootstrap_table(), [q.name for q in _queries(args)])
    print(json.dumps(gap, indent=2))
    verdict = "GO" if gap["oracle_improvement"] >= 0.15 else "NO-GO (headroom < 15%)"
    print(f"oracle improvement {gap['oracle_improvement']:.1%} -> {verdict}")


def cmd_train(args) -> None:
    from .evaluate import complete_queries, metrics, model_policy, oracle_policy, stock_policy
    from .training import train_model
    from .workload import split_by_template

    store = _store(args)
    train, test = split_by_template(_queries(args))
    arm_ids = tuple(a.id for a in ARMS)
    table = store.bootstrap_table()
    train_names = set(complete_queries(table, [q.name for q in train], arm_ids))
    kwargs = {}
    if args.model == "treecnn":
        kwargs["device"] = args.device
        if args.epochs:
            kwargs["epochs"] = args.epochs
    model = train_model(args.model, store, train_names, seed=args.seed, **kwargs)
    out = Path(args.out or config.MODELS_DIR / args.model)
    model.save(out)
    on = f" on {model.device}" if hasattr(model, "device") else ""
    print(f"trained {args.model}{on} using {len(train_names)} queries -> {out}")

    test_names = complete_queries(table, [q.name for q in test], arm_ids)
    if test_names:
        stock, oracle = stock_policy(table, test_names), oracle_policy(table, test_names, arm_ids)
        m = metrics(model_policy(args.model, model, table, test_names, arm_ids), stock, oracle)
        print(
            f"held-out templates ({len(test_names)} queries): total {m['total_ms']:.0f} ms"
            f" = {m['total_vs_postgres']:.3f}x postgres (oracle"
            f" {sum(oracle.latency.values()) / sum(stock.latency.values()):.3f}x),"
            f" mean regret {m['mean_regret']:.1%}, regressions {m['n_regressions']}"
        )


def cmd_run(args) -> None:
    from .db import connect
    from .models import load_model
    from .router import Router, format_route
    from .selector import Selector

    sql = Path(args.file).read_text() if args.file else args.sql
    if not sql:
        sys.exit("give SQL as an argument or with --file")
    sql = sql.strip().rstrip(";")
    model_dir = Path(args.model or config.MODELS_DIR / "treecnn")
    model = load_model(model_dir) if (model_dir / "meta.json").exists() else None
    if model is None:
        print(f"no model at {model_dir}; falling back to stock Postgres (arm 0)")
    baseline = None
    if args.file:
        baseline = _store(args).baseline_ms(Path(args.file).stem)
    with connect(args.dsn) as conn:
        router = Router(conn, model, Selector(mode="greedy", min_gain=args.min_gain))
        if args.no_execute:
            route = router.choose(sql)
        else:
            route = router.run(sql, baseline_ms=baseline, warmup=0, runs=1)
    print(format_route(route))


def cmd_bench(args) -> None:
    from .report import full_report, save_report, to_markdown

    store = _store(args)
    queries = _queries(args)
    kwargs = {"treecnn": {"epochs": args.epochs}} if args.epochs else None
    overhead = None
    if args.overhead:
        from .db import connect
        from .training import train_model
        from .workload import split_by_template

        train, test = split_by_template(queries)
        model = train_model(args.model, store, {q.name for q in train})
        with connect(args.dsn) as conn:
            from .report import measure_overhead

            overhead = measure_overhead(conn, model, test)
    report = full_report(
        store,
        queries,
        main_model=args.model,
        ablations=not args.no_ablations,
        online_epochs=args.online_epochs,
        online_train_epochs=args.online_train_epochs,
        seed=args.seed,
        model_kwargs=kwargs,
        overhead=overhead,
    )
    out = Path(args.out)
    save_report(report, out / "results.json")
    figures = []
    try:
        from .plots import make_all

        figures = make_all(report, out / "figures", rel_to=Path(args.markdown).parent)
    except Exception as e:  # never lose the report over a plotting problem
        print(f"skipping plots: {type(e).__name__}: {e}")
    Path(args.markdown).write_text(to_markdown(report, figures), encoding="utf-8")
    print(f"wrote {out / 'results.json'} and {args.markdown}")


def cmd_online(args) -> None:
    from .online import online_loop
    from .workload import split_by_template

    store = _store(args)
    train, test = split_by_template(_queries(args))
    arm_ids = parse_arm_ids(args.arms)
    kwargs = {"epochs": args.epochs} if args.model == "treecnn" and args.epochs else {}
    common = dict(
        model_kind=args.model,
        epochs=args.online_epochs,
        retrain_every=args.retrain_every,
        mode=args.mode,
        arm_ids=arm_ids,
        seed=args.seed,
        model_kwargs=kwargs,
    )
    if args.live:
        from .db import connect

        with connect(args.dsn) as conn:
            online_loop(store.bootstrap_table(), train, test, conn=conn, store=store, **common)
    else:
        online_loop(store.bootstrap_table(), train, test, **common)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="steerdb", description=__doc__)
    p.add_argument("--dsn", default=config.DSN, help="Postgres DSN (env STEERDB_DSN)")
    p.add_argument("--store", default=str(config.STORE_PATH), help="experience store (SQLite)")
    p.add_argument("--workload", default=str(config.WORKLOAD_DIR), help="directory of JOB *.sql")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("collect", help="Phase 0: execute every query under every arm")
    s.add_argument("--queries", help="comma-separated query names (default: all)")
    s.add_argument("--arms", help="comma-separated arm ids (default: all)")
    s.add_argument("--warmup", type=int, default=config.WARMUP_RUNS)
    s.add_argument("--runs", type=int, default=config.MEASURED_RUNS)
    s.add_argument("--timeout-ms", type=int, default=config.STATEMENT_TIMEOUT_MS)
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("oracle-gap", help="go/no-go: best arm vs stock Postgres")
    s.add_argument("--queries")
    s.set_defaults(func=cmd_oracle_gap)

    s = sub.add_parser("train", help="train a value model on the train templates")
    s.add_argument("--model", choices=("lgbm", "treecnn"), default="treecnn")
    s.add_argument("--epochs", type=int, help="Tree-CNN epochs")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", help="output directory (default runs/models/<model>)")
    s.add_argument(
        "--device",
        default="auto",
        help="Tree-CNN device: auto | cpu | cuda (auto = GPU if present)",
    )
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("run", help="route one query: plan all arms, pick one, execute")
    s.add_argument("sql", nargs="?")
    s.add_argument("--file", help="read SQL from a file (its name is used to find arm-0 history)")
    s.add_argument("--model", help="model directory (default runs/models/treecnn)")
    s.add_argument("--min-gain", type=float, default=0.05)
    s.add_argument("--no-execute", action="store_true", help="only plan and choose")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("bench", help="full offline evaluation + ablations + plots")
    s.add_argument("--model", choices=("lgbm", "treecnn"), default="treecnn")
    s.add_argument("--epochs", type=int, help="Tree-CNN epochs")
    s.add_argument("--online-epochs", type=int, default=5)
    s.add_argument(
        "--online-train-epochs", type=int, default=25, help="Tree-CNN epochs per online retrain"
    )
    s.add_argument("--no-ablations", action="store_true")
    s.add_argument(
        "--overhead", action="store_true", help="measure planning/inference overhead (needs DB)"
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=str(config.RUNS_DIR / "eval"))
    s.add_argument("--markdown", default=str(config.REPO_ROOT / "docs" / "results.md"))
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("online", help="Phase 3: online learning loop")
    s.add_argument("--model", choices=("lgbm", "treecnn"), default="treecnn")
    s.add_argument("--epochs", type=int, help="Tree-CNN training epochs per retrain")
    s.add_argument("--online-epochs", type=int, default=5)
    s.add_argument("--retrain-every", type=int, default=25)
    s.add_argument("--mode", choices=("thompson", "greedy"), default="thompson")
    s.add_argument("--arms")
    s.add_argument("--live", action="store_true", help="execute against Postgres (default: replay)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_online)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
