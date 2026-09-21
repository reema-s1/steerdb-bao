"""Phase 0: run every query under every arm and record (plan, latency) -- resumable."""

from __future__ import annotations

import time

from . import config
from .arms import ARMS, Arm
from .executor import execute
from .plan_gen import explain, plan_hash
from .store import ExperienceStore, Observation
from .workload import Query


def collect(
    conn,
    store: ExperienceStore,
    queries: list[Query],
    arms: tuple[Arm, ...] = ARMS,
    warmup: int = config.WARMUP_RUNS,
    runs: int = config.MEASURED_RUNS,
    timeout_ms: int = config.STATEMENT_TIMEOUT_MS,
    log=None,
) -> None:
    """Arms whose plan is identical to an already-executed arm of the same query are not
    re-executed; they reuse that measurement (same operator tree => same execution)."""
    log = log or (lambda msg: print(msg, flush=True))
    total = len(queries) * len(arms)
    done = 0
    t_start = time.perf_counter()
    for q in queries:
        measured: dict[str, Observation] = {}
        for obs in store.observations(("bootstrap",), {q.name}):
            measured.setdefault(obs.plan_hash, obs)
        for arm in arms:
            done += 1
            if store.has(q.name, arm.id, "bootstrap"):
                continue
            plan = explain(conn, q.sql, arm)
            h = plan_hash(plan)
            if h in measured:
                src = measured[h]
                store.add(
                    Observation(q.name, arm.id, h, plan, src.latency_ms, src.timed_out, "bootstrap")
                )
                log(f"[{done}/{total}] {q.name} arm{arm.id}: same plan as arm{src.arm}, reused")
                continue
            res = execute(conn, q.sql, arm, timeout_ms, warmup, runs)
            obs = Observation(q.name, arm.id, h, plan, res.latency_ms, res.timed_out, "bootstrap")
            store.add(obs)
            measured[h] = obs
            flag = " TIMEOUT" if res.timed_out else ""
            log(f"[{done}/{total}] {q.name} arm{arm.id}: {res.latency_ms:.1f} ms{flag}")
    log(f"collection finished in {time.perf_counter() - t_start:.0f}s")
