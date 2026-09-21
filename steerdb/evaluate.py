"""Offline policy evaluation over the bootstrap table (every query x every arm was executed).

Because every arm's latency is known, any policy (stock, oracle, random, a model) can be
scored without re-executing queries. The safety timeout is simulated faithfully: if the chosen
plan would exceed timeout_factor x arm-0 latency, it is cancelled at the timeout and arm 0 is
rerun, so the cost is timeout + arm-0 latency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .arms import ARMS
from .models import ValueModel
from .plan_gen import Candidate
from .selector import Selector
from .store import Observation

Table = dict[str, dict[int, Observation]]
REGRESSION_THRESHOLD = 1.2  # "more than 20% slower than Postgres"


@dataclass
class PolicyResult:
    name: str
    chosen: dict[str, int] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)  # effective, incl. timeout fallback
    fallbacks: int = 0


def candidates_from_table(table: Table, query: str, arm_ids: tuple[int, ...]) -> list[Candidate]:
    by_hash: dict[str, Candidate] = {}
    for a in arm_ids:
        obs = table[query][a]
        if obs.plan_hash in by_hash:
            by_hash[obs.plan_hash].arms.append(a)
        else:
            by_hash[obs.plan_hash] = Candidate(obs.plan, obs.plan_hash, [a])
    return sorted(by_hash.values(), key=lambda c: c.arm)


def complete_queries(table: Table, queries: list[str], arm_ids: tuple[int, ...]) -> list[str]:
    return [q for q in queries if q in table and all(a in table[q] for a in arm_ids)]


def run_policy(
    name: str,
    table: Table,
    queries: list[str],
    choose: Callable[[str], int],
    selector: Selector | None = None,
) -> PolicyResult:
    selector = selector or Selector()
    res = PolicyResult(name)
    for q in queries:
        arm = choose(q)
        lat = table[q][arm].latency_ms
        base = table[q][0].latency_ms
        timeout = selector.timeout_ms(base)
        if arm != 0 and timeout is not None and lat > timeout:
            lat = timeout + base
            res.fallbacks += 1
        res.chosen[q] = arm
        res.latency[q] = lat
    return res


def stock_policy(table, queries, **_):
    return run_policy("postgres", table, queries, lambda q: 0)


def oracle_policy(table, queries, arm_ids, **_):
    return run_policy(
        "oracle", table, queries, lambda q: min(arm_ids, key=lambda a: table[q][a].latency_ms)
    )


def random_policy(table, queries, arm_ids, seed: int = 0, **_):
    rng = np.random.default_rng(seed)
    return run_policy("random", table, queries, lambda q: int(rng.choice(arm_ids)))


def model_policy(
    name: str,
    model: ValueModel,
    table: Table,
    queries: list[str],
    arm_ids: tuple[int, ...],
    selector: Selector | None = None,
) -> PolicyResult:
    selector = selector or Selector(mode="greedy")

    def choose(q: str) -> int:
        cands = candidates_from_table(table, q, arm_ids)
        if not model.trained:
            return selector.choose([c.arm for c in cands], None, trained=False).arm
        mu, sigma = model.predict([c.plan for c in cands])
        return selector.choose([c.arm for c in cands], mu, sigma).arm

    return run_policy(name, table, queries, choose, selector)


def metrics(res: PolicyResult, stock: PolicyResult, oracle: PolicyResult) -> dict:
    qs = sorted(res.latency)
    lat = np.array([res.latency[q] for q in qs])
    base = np.array([stock.latency[q] for q in qs])
    best = np.array([oracle.latency[q] for q in qs])
    speedup = base / np.maximum(lat, 1e-9)
    regret = (lat - best) / np.maximum(best, 1e-9)
    return {
        "policy": res.name,
        "n_queries": len(qs),
        "total_ms": float(lat.sum()),
        "total_vs_postgres": float(lat.sum() / base.sum()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "n_regressions": int((lat > REGRESSION_THRESHOLD * base).sum()),
        "n_wins": int((lat * REGRESSION_THRESHOLD < base).sum()),
        "fallbacks": res.fallbacks,
        "speedups": {q: float(s) for q, s in zip(qs, speedup)},
    }


def regressions(res: PolicyResult, stock: PolicyResult, oracle: PolicyResult) -> list[dict]:
    rows = []
    for q in sorted(res.latency):
        if res.latency[q] > REGRESSION_THRESHOLD * stock.latency[q]:
            rows.append(
                {
                    "query": q,
                    "chosen_arm": res.chosen[q],
                    "chosen_ms": res.latency[q],
                    "postgres_ms": stock.latency[q],
                    "oracle_arm": oracle.chosen[q],
                    "oracle_ms": oracle.latency[q],
                    "slowdown": res.latency[q] / stock.latency[q],
                }
            )
    return rows


def oracle_gap(table: Table, queries: list[str], arm_ids=tuple(a.id for a in ARMS)) -> dict:
    """Go/no-go check (design doc section 10): how much faster is the best arm than arm 0?"""
    qs = complete_queries(table, queries, arm_ids)
    stock = stock_policy(table, qs)
    oracle = oracle_policy(table, qs, arm_ids)
    total_stock = sum(stock.latency.values())
    total_oracle = sum(oracle.latency.values())
    arm_wins: dict[int, int] = {}
    for q in qs:
        arm_wins[oracle.chosen[q]] = arm_wins.get(oracle.chosen[q], 0) + 1
    return {
        "n_queries": len(qs),
        "postgres_total_ms": total_stock,
        "oracle_total_ms": total_oracle,
        "oracle_improvement": 1 - total_oracle / total_stock if total_stock else 0.0,
        "oracle_best_arm_counts": dict(sorted(arm_wins.items())),
    }
