"""Generate one candidate plan per arm with EXPLAIN (FORMAT JSON) -- planning only."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from .arms import ARMS, Arm
from .db import apply_arm

# Keys that define *what a plan does*. Costs/row estimates are excluded on purpose: two arms
# that yield the same operator tree execute identically even if the disabled-method penalty
# inflates one arm's cost numbers.
_STRUCTURAL_KEYS = (
    "Node Type",
    "Parent Relationship",
    "Join Type",
    "Relation Name",
    "Alias",
    "Index Name",
    "Strategy",
    "Scan Direction",
    "Hash Cond",
    "Merge Cond",
    "Index Cond",
    "Recheck Cond",
    "Join Filter",
    "Filter",
    "Sort Key",
    "Inner Unique",
)


@dataclass
class Candidate:
    """A distinct plan and every arm that produces it."""

    plan: dict  # the root "Plan" node
    plan_hash: str
    arms: list[int] = field(default_factory=list)

    @property
    def arm(self) -> int:
        """Representative arm (the lowest id, so arm 0 wins ties)."""
        return min(self.arms)


def _structure(node: dict) -> dict:
    out = {k: node[k] for k in _STRUCTURAL_KEYS if k in node}
    out["Plans"] = [_structure(c) for c in node.get("Plans", [])]
    return out


def plan_hash(plan: dict) -> str:
    blob = json.dumps(_structure(plan), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def explain(conn, sql: str, arm: Arm) -> dict:
    """Plan `sql` under `arm` without executing it. Returns the root Plan node."""
    apply_arm(conn, arm)
    try:
        row = conn.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchone()
    finally:
        apply_arm(conn, ARMS[0])
    doc = row[0]
    if isinstance(doc, str):
        doc = json.loads(doc)
    return doc[0]["Plan"]


def candidate_plans(conn, sql: str, arms: tuple[Arm, ...] = ARMS) -> tuple[list[Candidate], float]:
    """EXPLAIN under every arm, deduplicate identical plans. Returns (candidates, planning_ms)."""
    t0 = time.perf_counter()
    by_hash: dict[str, Candidate] = {}
    for arm in arms:
        plan = explain(conn, sql, arm)
        h = plan_hash(plan)
        if h in by_hash:
            by_hash[h].arms.append(arm.id)
        else:
            by_hash[h] = Candidate(plan, h, [arm.id])
    planning_ms = (time.perf_counter() - t0) * 1000
    return sorted(by_hash.values(), key=lambda c: c.arm), planning_ms
