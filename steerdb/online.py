"""Phase 3: Bao-style online learning loop.

    for each epoch, for each training query (shuffled):
        score K plans, choose via Thompson sampling (or greedy)
        execute the chosen plan (with the safety timeout), record (plan, latency)
        retrain every N queries on all experience

Two execution back-ends:
  * live   -- actually runs queries against Postgres and records to the experience store
  * replay -- looks latencies up in the bootstrap table (all arms were executed in Phase 0),
              so exploration strategies can be compared quickly and deterministically.

The loop starts "cold": its only initial experience is how stock Postgres (arm 0) ran the
training queries, which is exactly what a real deployment would have.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .arms import ARMS, DEFAULT_ARM, get_arm
from .evaluate import (
    Table,
    candidates_from_table,
    complete_queries,
    metrics,
    model_policy,
    oracle_policy,
    stock_policy,
)
from .executor import execute
from .log import log
from .models import make_model
from .plan_gen import candidate_plans
from .selector import Selector
from .store import ExperienceStore, Observation
from .training import xy
from .workload import Query


@dataclass
class EpochStats:
    epoch: int
    train_workload_ms: float
    explored: int
    fallbacks: int
    test: dict | None = None


@dataclass
class OnlineResult:
    epochs: list[EpochStats] = field(default_factory=list)
    postgres_train_ms: float = 0.0
    oracle_train_ms: float = 0.0


def online_loop(
    table: Table,
    train: list[Query],
    test: list[Query],
    model_kind: str = "treecnn",
    epochs: int = 5,
    retrain_every: int = 25,
    mode: str = "thompson",
    arm_ids: tuple[int, ...] = tuple(a.id for a in ARMS),
    conn=None,
    store: ExperienceStore | None = None,
    seed: int = 0,
    model_kwargs: dict | None = None,
    log=log,
) -> OnlineResult:
    live = conn is not None
    rng = random.Random(seed)
    selector = Selector(mode=mode, seed=seed)
    eval_selector = Selector(mode="greedy")
    arms = tuple(get_arm(a) for a in arm_ids)
    model_kwargs = model_kwargs or {}
    # Only queries with every arm measured: needed for replay, the timeout baseline and eval.
    train_names = complete_queries(table, [q.name for q in train], arm_ids)
    test_names = complete_queries(table, [q.name for q in test], arm_ids)
    train = [q for q in train if q.name in set(train_names)]

    experience: list[Observation] = [table[name][0] for name in train_names]
    model = make_model(model_kind, seed=seed, **model_kwargs)
    model.fit(*xy(experience))

    out = OnlineResult()
    out.postgres_train_ms = sum(stock_policy(table, train_names).latency.values())
    out.oracle_train_ms = sum(oracle_policy(table, train_names, arm_ids).latency.values())
    if test_names:
        stock_t = stock_policy(table, test_names)
        oracle_t = oracle_policy(table, test_names, arm_ids)

    since_retrain = 0
    for epoch in range(1, epochs + 1):
        order = list(train)
        rng.shuffle(order)
        total = 0.0
        explored = fallbacks = 0
        for q in order:
            base_ms = table[q.name][0].latency_ms
            if live:
                cands, _ = candidate_plans(conn, q.sql, arms)
            else:
                cands = candidates_from_table(table, q.name, arm_ids)
            mu, sigma = model.predict([c.plan for c in cands])
            d = selector.choose([c.arm for c in cands], mu, sigma, model.trained)
            explored += d.reason == "explore"
            cand = cands[d.index]
            timeout = selector.timeout_ms(base_ms) if d.arm != DEFAULT_ARM.id else None

            if live:
                res = execute(conn, q.sql, get_arm(d.arm), timeout, warmup=0, runs=1)
                lat, timed_out = res.latency_ms, res.timed_out
            else:
                lat = table[q.name][d.arm].latency_ms
                timed_out = timeout is not None and lat > timeout
                if timed_out:
                    lat = float(timeout)

            obs = Observation(
                q.name, d.arm, cand.plan_hash, cand.plan, lat, timed_out, "online", epoch
            )
            experience.append(obs)
            if live and store is not None:
                store.add(obs)
            cost = lat
            if timed_out:  # safety fallback: rerun stock plan
                fallbacks += 1
                if live:
                    cost += execute(conn, q.sql, DEFAULT_ARM, None, warmup=0, runs=1).latency_ms
                else:
                    cost += base_ms
            total += cost

            since_retrain += 1
            if since_retrain >= retrain_every:
                model = make_model(model_kind, seed=seed + epoch, **model_kwargs)
                model.fit(*xy(experience))
                since_retrain = 0

        stats = EpochStats(epoch, total, explored, fallbacks)
        if test_names:
            res_t = model_policy("online", model, table, test_names, arm_ids, eval_selector)
            m = metrics(res_t, stock_t, oracle_t)
            m.pop("speedups")
            stats.test = m
        out.epochs.append(stats)
        test_str = (
            f", test total {stats.test['total_ms']:.0f} ms"
            f" ({stats.test['total_vs_postgres']:.2f}x postgres)"
            if stats.test
            else ""
        )
        log(
            f"epoch {epoch}: train workload {total:.0f} ms"
            f" (postgres {out.postgres_train_ms:.0f}, oracle {out.oracle_train_ms:.0f});"
            f" explored {explored}, fallbacks {fallbacks}{test_str}"
        )
    return out
