"""Hint sets ("arms"): planner GUC combinations that steer what Postgres produces.

`enable_*=off` penalizes a method rather than forbidding it, so every arm always yields a
valid plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Every GUC any arm touches. Before applying an arm, all of these are reset to "on" so
# arms never leak into each other on a shared session.
PLANNER_GUCS = (
    "enable_nestloop",
    "enable_hashjoin",
    "enable_mergejoin",
    "enable_seqscan",
    "enable_indexscan",
    "enable_bitmapscan",
)


@dataclass(frozen=True)
class Arm:
    id: int
    name: str
    settings: dict[str, str] = field(default_factory=dict)
    intent: str = ""

    def set_statements(self) -> list[str]:
        """SQL to put a session into this arm's configuration (idempotent)."""
        stmts = [f"SET {g} = on" for g in PLANNER_GUCS if g not in self.settings]
        stmts += [f"SET {g} = {v}" for g, v in sorted(self.settings.items())]
        return stmts


ARMS: tuple[Arm, ...] = (
    Arm(0, "default", {}, "baseline Postgres"),
    Arm(
        1,
        "no_nestloop",
        {"enable_nestloop": "off"},
        "avoid nested loops when row estimates are too low",
    ),
    Arm(2, "no_hashjoin", {"enable_hashjoin": "off"}, "force merge or nested loop"),
    Arm(3, "no_mergejoin", {"enable_mergejoin": "off"}, "hash or nested loop only"),
    Arm(4, "hash_only", {"enable_nestloop": "off", "enable_mergejoin": "off"}, "hash joins only"),
    Arm(
        5,
        "nestloop_only",
        {"enable_hashjoin": "off", "enable_mergejoin": "off"},
        "nested loops only",
    ),
    Arm(6, "no_seqscan", {"enable_seqscan": "off"}, "favor index scans"),
    Arm(
        7,
        "no_indexscan",
        {"enable_indexscan": "off", "enable_bitmapscan": "off"},
        "favor sequential scans",
    ),
)

DEFAULT_ARM = ARMS[0]

# Arm subsets for the "number of arms K" ablation. Arm 0 is always included.
ARM_SUBSETS: dict[int, tuple[int, ...]] = {
    2: (0, 1),
    4: (0, 1, 3, 6),
    8: tuple(a.id for a in ARMS),
}


def get_arm(arm_id: int) -> Arm:
    for arm in ARMS:
        if arm.id == arm_id:
            return arm
    raise KeyError(f"unknown arm {arm_id}")


def parse_arm_ids(spec: str | None) -> tuple[int, ...]:
    """'0,1,4' -> (0, 1, 4); None/'' -> all arms. Arm 0 is always added."""
    if not spec:
        return tuple(a.id for a in ARMS)
    ids = {int(x) for x in spec.split(",") if x.strip()}
    for i in ids:
        get_arm(i)
    ids.add(0)
    return tuple(sorted(ids))
