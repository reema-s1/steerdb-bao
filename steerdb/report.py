"""Full evaluation: baselines, models, ablations -> results.json + docs/results.md.

Headline protocol: leave-templates-out cross-validation. Templates are dealt into k folds; each
fold is predicted by models trained only on the other folds' templates, so every query is
evaluated exactly once by a model that never saw its template (nor, for CEB, its join graph).
Compared with one fixed split, this uses every query for evaluation and keeps small and large
joins on both sides (JOB's fixed split 1-25 / 26-33 put 18 of its 23 test queries in the
12+-table category, versus 2 of 90 training queries).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import config
from .arms import ARM_SUBSETS, ARMS
from .evaluate import (
    PolicyResult,
    complete_queries,
    metrics,
    model_policy,
    oracle_gap,
    oracle_policy,
    random_policy,
    regressions,
    stock_policy,
)
from .log import log
from .models import CostModel
from .online import online_loop
from .selector import Selector
from .store import ExperienceStore
from .training import train_model
from .workload import Query, split_random, subsample_templates, template_folds

ALL_ARMS = tuple(a.id for a in ARMS)
Split = tuple[list[Query], list[Query]]


def _names(qs: list[Query]) -> list[str]:
    return [q.name for q in qs]


def workload_of(name: str) -> str:
    return "ceb" if name.startswith("ceb") else "job"


def cv_splits(queries: list[Query], k: int = 5, seed: int = 0) -> list[Split]:
    splits = []
    for fold in template_folds(queries, k, seed):
        held = set(fold)
        train = [q for q in queries if q.template not in held]
        test = [q for q in queries if q.template in held]
        splits.append((train, test))
    return splits


def _merge(parts: list[PolicyResult]) -> PolicyResult:
    out = PolicyResult(parts[0].name)
    for p in parts:
        out.chosen.update(p.chosen)
        out.latency.update(p.latency)
        out.fallbacks += p.fallbacks
    return out


def _restrict(res: PolicyResult, names: set[str]) -> PolicyResult:
    out = PolicyResult(res.name)
    out.chosen = {q: a for q, a in res.chosen.items() if q in names}
    out.latency = {q: v for q, v in res.latency.items() if q in names}
    return out


def evaluate_folds(
    store: ExperienceStore,
    splits: list[Split],
    kinds=("lgbm", "treecnn"),
    arm_ids=ALL_ARMS,
    seed: int = 0,
    model_kwargs: dict | None = None,
    train_fraction: float = 1.0,
    log=log,
) -> dict:
    """Train on each split's train side, predict its test side, merge all test predictions."""
    table = store.bootstrap_table()
    selector = Selector(mode="greedy")
    parts: dict[str, list[PolicyResult]] = {}
    train_seconds: dict[str, float] = {}
    n_train = []
    for i, (train, test) in enumerate(splits):
        if train_fraction < 1.0:
            train = subsample_templates(train, train_fraction, seed=seed + i)
        test_names = complete_queries(table, _names(test), arm_ids)
        train_names = set(complete_queries(table, _names(train), arm_ids))
        if not test_names or not train_names:
            continue
        n_train.append(len(train_names))
        fold = {
            "postgres": stock_policy(table, test_names),
            "oracle": oracle_policy(table, test_names, arm_ids),
            "random": random_policy(table, test_names, arm_ids, seed=seed + i),
            "cost-only": model_policy(
                "cost-only", CostModel(), table, test_names, arm_ids, selector
            ),
        }
        for kind in kinds:
            t0 = time.perf_counter()
            kw = (model_kwargs or {}).get(kind, {})
            model = train_model(kind, store, train_names, arm_ids, seed=seed, **kw)
            train_seconds[kind] = train_seconds.get(kind, 0.0) + time.perf_counter() - t0
            fold[kind] = model_policy(kind, model, table, test_names, arm_ids, selector)
        log(
            f"  fold {i + 1}/{len(splits)}: trained on {len(train_names)}, tested {len(test_names)}"
        )
        for k, r in fold.items():
            parts.setdefault(k, []).append(r)

    results = {k: _merge(v) for k, v in parts.items()}
    stock, oracle = results["postgres"], results["oracle"]
    evaluated = set(stock.latency)
    by_workload = {}
    for wl in sorted({workload_of(q) for q in evaluated}):
        names = {q for q in evaluated if workload_of(q) == wl}
        s, o = _restrict(stock, names), _restrict(oracle, names)
        by_workload[wl] = {}
        for k, r in results.items():
            m = metrics(_restrict(r, names), s, o)
            m.pop("speedups")
            by_workload[wl][k] = m
    return {
        "metrics": {k: metrics(r, stock, oracle) for k, r in results.items()},
        "by_workload": by_workload if len(by_workload) > 1 else {},
        "regressions": {
            k: regressions(r, stock, oracle)
            for k, r in results.items()
            if k not in ("postgres", "oracle")
        },
        "chosen": {k: r.chosen for k, r in results.items()},
        "train_seconds": train_seconds,
        "n_folds": len(n_train),
        "n_train": int(np.mean(n_train)) if n_train else 0,
        "n_test": len(evaluated),
    }


def _strip(res: dict) -> dict:
    """Drop per-query detail from ablation results."""
    return {
        k: {kk: vv for kk, vv in v.items() if kk != "speedups"} for k, v in res["metrics"].items()
    }


def full_report(
    store: ExperienceStore,
    queries: list[Query],
    main_model: str = "treecnn",
    ablations: bool = True,
    online_epochs: int = 5,
    online_train_epochs: int = 25,
    folds: int = 5,
    seed: int = 0,
    model_kwargs: dict | None = None,
    overhead: dict | None = None,
    log=log,
) -> dict:
    table = store.bootstrap_table()
    queries = [q for q in queries if q.name in table]
    splits = cv_splits(queries, folds, seed)
    all_names = _names(queries)
    workloads = sorted({workload_of(n) for n in all_names})
    gaps = {"all": oracle_gap(table, all_names)}
    for wl in workloads:
        gaps[wl] = oracle_gap(table, [n for n in all_names if workload_of(n) == wl])
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "session_settings": config.SESSION_SETTINGS,
        "protocol": {
            "folds": [sorted({q.template for q in test}) for _, test in splits],
            "n_queries": len(queries),
            "workloads": {wl: sum(workload_of(n) == wl for n in all_names) for wl in workloads},
        },
        "oracle_gap": gaps,
    }
    log(f"main evaluation: {len(splits)}-fold leave-templates-out cross-validation")
    main = evaluate_folds(store, splits, seed=seed, model_kwargs=model_kwargs, log=log)
    report["main"] = main

    if online_epochs:
        # The online loop retrains every 25 queries, so it uses a shorter per-retrain budget.
        online_kwargs = dict((model_kwargs or {}).get(main_model, {}))
        if main_model == "treecnn":
            online_kwargs["epochs"] = online_train_epochs
        train, test = splits[0]
        for mode in ("thompson", "greedy") if ablations else ("thompson",):
            log(f"online loop (replay, {mode}, fold 1)")
            res = online_loop(
                table,
                train,
                test,
                model_kind=main_model,
                epochs=online_epochs,
                mode=mode,
                seed=seed,
                model_kwargs=online_kwargs,
                log=log,
            )
            report.setdefault("online", {})[mode] = {
                "epochs": [asdict(e) for e in res.epochs],
                "postgres_train_ms": res.postgres_train_ms,
                "oracle_train_ms": res.oracle_train_ms,
            }

    if ablations:
        abl: dict = {}
        one = (main_model,)
        log("ablation: number of arms K")
        abl["arms"] = {}
        for k, subset in ARM_SUBSETS.items():
            r = evaluate_folds(
                store, splits, one, subset, seed=seed, model_kwargs=model_kwargs, log=log
            )
            abl["arms"][str(k)] = {"arms": list(subset), "model": main_model, **_strip(r)}
        log("ablation: amount of training data (fraction of each fold's training templates)")
        abl["train_fraction"] = {}
        for frac in (0.25, 0.5, 1.0):
            if frac == 1.0:
                r = main
            else:
                r = evaluate_folds(
                    store,
                    splits,
                    one,
                    seed=seed,
                    model_kwargs=model_kwargs,
                    train_fraction=frac,
                    log=log,
                )
            abl["train_fraction"][str(frac)] = {
                "n_train": r["n_train"],
                "model": main_model,
                **_strip(r),
            }
        log("ablation: seen (random split, leaky) vs unseen templates")
        leaky = [split_random(queries, test_frac=1 / len(splits), seed=seed)]
        r = evaluate_folds(store, leaky, one, seed=seed, model_kwargs=model_kwargs, log=log)
        abl["seen_vs_unseen"] = {
            "model": main_model,
            "random_split": _strip(r),
            "template_split": _strip(main),
        }
        report["ablations"] = abl

    if overhead:
        report["overhead"] = overhead
    return report


def measure_overhead(conn, model, queries: list[Query]) -> dict:
    from .router import Router

    router = Router(conn, model)
    planning, inference = [], []
    for q in queries:
        route = router.choose(q.sql)
        planning.append(route.planning_ms)
        inference.append(route.inference_ms)
    return {
        "n_queries": len(queries),
        "planning_ms_median": float(np.median(planning)),
        "planning_ms_p95": float(np.percentile(planning, 95)),
        "inference_ms_median": float(np.median(inference)),
        "inference_ms_p95": float(np.percentile(inference, 95)),
    }


# --------------------------------------------------------------------------- markdown

POLICY_ORDER = ("postgres", "oracle", "random", "cost-only", "lgbm", "treecnn")


def _fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.1f} ms"


def _metrics_table(ms: dict, order=POLICY_ORDER) -> list[str]:
    lines = [
        "| Policy | Total | vs. Postgres | p50 | p95 | p99 | Mean regret | >20% slower | >20% faster |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in [k for k in order if k in ms] + [k for k in ms if k not in order]:
        m = ms[k]
        lines.append(
            f"| {k} | {_fmt_ms(m['total_ms'])} | {m['total_vs_postgres']:.3f}x | {_fmt_ms(m['p50_ms'])}"
            f" | {_fmt_ms(m['p95_ms'])} | {_fmt_ms(m['p99_ms'])} | {m['mean_regret']:.1%}"
            f" | {m['n_regressions']} | {m['n_wins']} |"
        )
    return lines


def to_markdown(r: dict, figures: list[str] | None = None) -> str:
    main, proto = r["main"], r["protocol"]
    gaps = r["oracle_gap"]
    workloads = ", ".join(f"{wl.upper()} {n}" for wl, n in proto["workloads"].items())
    L = [
        "# steerdb results",
        "",
        f"_Generated {r['generated_at']} by `python experiments/run_eval.py`. Do not edit by hand._",
        "",
        "## Setup",
        "",
        f"- Machine: {r['machine']['platform']} ({r['machine']['processor'] or 'unknown CPU'}),"
        f" Python {r['machine']['python']}",
        f"- Session settings: {', '.join(f'`{k}={v}`' for k, v in r['session_settings'].items())}",
        f"- Workload: {proto['n_queries']} queries ({workloads})",
        f"- Protocol: {len(proto['folds'])}-fold leave-templates-out cross-validation. Every query"
        " is predicted by models trained only on other templates (~"
        f"{main['n_train']} training queries per fold).",
        "- Labels: warm cache, 1 warm-up + median of 3 runs; timeouts recorded as the timeout (censored)",
        "",
        "## Go/no-go: oracle gap",
        "",
        "Best-arm oracle vs. stock Postgres (total workload time):",
        "",
        "| Workload | Queries | Postgres | Oracle | Improvement | Best-arm counts |",
        "|---|---|---|---|---|---|",
    ]
    for wl, g in gaps.items():
        L.append(
            f"| {wl.upper() if wl != 'all' else 'all'} | {g['n_queries']} |"
            f" {_fmt_ms(g['postgres_total_ms'])} | {_fmt_ms(g['oracle_total_ms'])} |"
            f" **{g['oracle_improvement']:.1%}** | {g['oracle_best_arm_counts']} |"
        )
    L += [
        "",
        "## Headline: unseen templates (cross-validated)",
        "",
        *_metrics_table(main["metrics"]),
        "",
        "Effective latency includes the safety timeout: a non-default plan slower than"
        " 2x the Postgres latency is cancelled and arm 0 rerun (cost = timeout + arm 0).",
        "",
    ]
    for wl, ms in main.get("by_workload", {}).items():
        L += [f"### {wl.upper()} only", "", *_metrics_table(ms), ""]
    if figures:
        L += [f"![{Path(f).stem}]({f})" for f in figures] + [""]

    L += ["## Regressions (>20% slower than Postgres)", ""]
    for kind in ("treecnn", "lgbm"):
        rows = main["regressions"].get(kind, [])
        L.append(f"**{kind}:** {len(rows)} regression(s)")
        if rows:
            L += [
                "",
                "| Query | Chosen arm | Chosen | Postgres | Oracle arm | Oracle | Slowdown |",
                "|---|---|---|---|---|---|---|",
            ]
            L += [
                f"| {x['query']} | {x['chosen_arm']} | {_fmt_ms(x['chosen_ms'])} |"
                f" {_fmt_ms(x['postgres_ms'])} | {x['oracle_arm']} | {_fmt_ms(x['oracle_ms'])} |"
                f" {x['slowdown']:.2f}x |"
                for x in rows
            ]
        L.append("")

    if "online" in r:
        L += [
            "## Online learning (replay of bootstrap latencies)",
            "",
            "| Mode | Epoch | Train workload | Explored | Fallbacks | Test vs. Postgres |",
            "|---|---|---|---|---|---|",
        ]
        for mode, o in r["online"].items():
            for e in o["epochs"]:
                t = f"{e['test']['total_vs_postgres']:.3f}x" if e.get("test") else "-"
                L.append(
                    f"| {mode} | {e['epoch']} | {_fmt_ms(e['train_workload_ms'])} |"
                    f" {e['explored']} | {e['fallbacks']} | {t} |"
                )
        o = next(iter(r["online"].values()))
        L += [
            "",
            f"Reference train workload: Postgres {_fmt_ms(o['postgres_train_ms'])},"
            f" oracle {_fmt_ms(o['oracle_train_ms'])}.",
            "",
        ]

    if "ablations" in r:
        a = r["ablations"]
        L += [
            "## Ablations",
            "",
            "### Number of arms K",
            "",
            "| K | Arms | Postgres | Oracle | Model | Model vs. Postgres | Regressions |",
            "|---|---|---|---|---|---|---|",
        ]
        for k, v in a["arms"].items():
            m = v[v["model"]]
            L.append(
                f"| {k} | {v['arms']} | {_fmt_ms(v['postgres']['total_ms'])} |"
                f" {_fmt_ms(v['oracle']['total_ms'])} | {_fmt_ms(m['total_ms'])} |"
                f" {m['total_vs_postgres']:.3f}x | {m['n_regressions']} |"
            )
        L += [
            "",
            "### Amount of training data (fraction of train templates)",
            "",
            "| Fraction | Train queries | Model vs. Postgres | Mean regret | Regressions |",
            "|---|---|---|---|---|",
        ]
        for f, v in a["train_fraction"].items():
            m = v[v["model"]]
            L.append(
                f"| {float(f):.0%} | {v['n_train']} | {m['total_vs_postgres']:.3f}x |"
                f" {m['mean_regret']:.1%} | {m['n_regressions']} |"
            )
        s = a["seen_vs_unseen"]
        L += [
            "",
            "### Seen vs. unseen templates",
            "",
            "A random split puts near-duplicate variants (e.g. `16a`/`16b`) on both sides,"
            " which inflates results. Compare:",
            "",
            "| Split | Model vs. Postgres | Oracle vs. Postgres | Mean regret |",
            "|---|---|---|---|",
        ]
        for label, key in (
            ("random (leaky)", "random_split"),
            ("template (honest)", "template_split"),
        ):
            ms = s[key]
            m = ms[s["model"]]
            L.append(
                f"| {label} | {m['total_vs_postgres']:.3f}x |"
                f" {ms['oracle']['total_vs_postgres']:.3f}x | {m['mean_regret']:.1%} |"
            )
        L.append("")

    if "overhead" in r:
        o = r["overhead"]
        L += [
            "## Overhead",
            "",
            f"Over {o['n_queries']} queries: planning all arms median {o['planning_ms_median']:.1f} ms"
            f" (p95 {o['planning_ms_p95']:.1f} ms); model inference median"
            f" {o['inference_ms_median']:.1f} ms (p95 {o['inference_ms_p95']:.1f} ms).",
            "",
        ]
    return "\n".join(L)


def save_report(r: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(r, indent=2))
