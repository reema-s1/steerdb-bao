"""A synthetic experience store whose latencies follow a simple, learnable rule.

Stock Postgres (arm 0) picks a nested loop, which is slow for large inputs; arms that
disable nested loops pick a hash join, which is fast. So there is real headroom, and a model
that reads the join operator and row estimate can find it.
"""

from __future__ import annotations

from conftest import make_plan

from steerdb.arms import ARMS
from steerdb.plan_gen import plan_hash
from steerdb.store import ExperienceStore, Observation
from steerdb.workload import Query

ARM_JOIN = {
    0: "Nested Loop",
    1: "Hash Join",
    2: "Merge Join",
    3: "Nested Loop",
    4: "Hash Join",
    5: "Nested Loop",
    6: "Nested Loop",
    7: "Hash Join",
}


def latency(join: str, rows: float) -> float:
    if join == "Nested Loop":
        return 1.0 + rows * 0.05
    if join == "Hash Join":
        return 40.0 + rows * 0.002
    return 60.0 + rows * 0.004


def synthetic_queries(n_templates: int = 33, variants: str = "abc") -> list[Query]:
    return [Query(f"{t}{v}", t, f"SELECT {t}") for t in range(1, n_templates + 1) for v in variants]


def query_rows(q: Query) -> float:
    return 10.0 * (1 + (q.template * 7 + ord(q.name[-1])) % 40) ** 2


def build_store(queries: list[Query]) -> ExperienceStore:
    store = ExperienceStore(":memory:")
    for q in queries:
        rows = query_rows(q)
        for arm in ARMS:
            join = ARM_JOIN[arm.id]
            plan = make_plan(join=join, rows=rows, cost=rows * (1 + arm.id * 0.01))
            store.add(
                Observation(
                    q.name, arm.id, plan_hash(plan), plan, latency(join, rows), False, "bootstrap"
                )
            )
    return store
