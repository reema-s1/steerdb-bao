"""Load workloads (JOB, CEB) and split them by template, never randomly (variants leak).

Query names and their template groups:
    JOB  16b       -> template "16"
    CEB  ceb9a_07  -> template "ceb9"  (grouped by join graph, so 9a/9b never straddle a split)
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from . import config

_JOB_RE = re.compile(r"^(\d+)([a-z]*)$")
_CEB_RE = re.compile(r"^ceb(\d+)([a-z]*)_(\d+)$")


@dataclass(frozen=True)
class Query:
    name: str  # e.g. "16b", "ceb9a_07"
    template: str  # e.g. "16", "ceb9"
    sql: str


def template_of(name: str) -> str:
    if m := _JOB_RE.match(name):
        return m.group(1)
    if m := _CEB_RE.match(name):
        return f"ceb{m.group(1)}"
    raise ValueError(f"not a JOB or CEB query name: {name!r}")


def template_sort_key(template: str) -> tuple[str, int]:
    """Natural order: 1, 2, ..., 33, ceb3, ceb7, ..."""
    m = re.match(r"^([a-z]*)(\d+)$", template)
    return (m.group(1), int(m.group(2))) if m else (template, 0)


def _dirs(spec) -> list[Path]:
    if spec is None:
        spec = config.WORKLOAD_DIR
    if isinstance(spec, (list, tuple)):
        return [Path(s) for s in spec]
    return [Path(s) for s in re.split(rf"[,{re.escape(os.pathsep)}]", str(spec)) if s]


def load_workload(directories=None) -> list[Query]:
    """Load *.sql from one or more directories (list, or a string separated by ',' or os.pathsep)."""
    queries = []
    for directory in _dirs(directories):
        for path in directory.glob("*.sql"):
            try:
                template = template_of(path.stem)
            except ValueError:
                continue  # schema.sql, fkindexes.sql, ...
            sql = path.read_text(encoding="utf-8").strip().rstrip(";")
            queries.append(Query(path.stem, template, sql))
    if not queries:
        raise FileNotFoundError(
            f"no queries found in {directories or config.WORKLOAD_DIR}; run"
            " `bash workload/fetch_job.sh` first"
        )
    return sorted(queries, key=lambda q: (template_sort_key(q.template), q.name))


def templates_of(queries: list[Query]) -> list[str]:
    return sorted({q.template for q in queries}, key=template_sort_key)


def split_by_template(
    queries: list[Query], test_templates: tuple[str, ...] = config.TEST_TEMPLATES
) -> tuple[list[Query], list[Query]]:
    test_set = set(test_templates)
    train = [q for q in queries if q.template not in test_set]
    test = [q for q in queries if q.template in test_set]
    return train, test


def template_folds(queries: list[Query], k: int = 5, seed: int = 0) -> list[list[str]]:
    """Partition the templates into k groups for leave-templates-out cross-validation.

    Templates are shuffled with a fixed seed, then dealt round-robin, so every fold mixes small
    and large joins (JOB's templates grow in complexity with their number).
    """
    templates = templates_of(queries)
    random.Random(seed).shuffle(templates)
    k = min(k, len(templates))
    return [sorted(templates[i::k], key=template_sort_key) for i in range(k)]


def split_random(
    queries: list[Query], test_frac: float = 0.25, seed: int = 0
) -> tuple[list[Query], list[Query]]:
    """Leaky random split, only used to *demonstrate* the seen-template inflation."""
    rng = random.Random(seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)
    n_test = round(len(shuffled) * test_frac)
    return sorted(shuffled[n_test:], key=lambda q: q.name), sorted(
        shuffled[:n_test], key=lambda q: q.name
    )


def subsample_templates(queries: list[Query], frac: float, seed: int = 0) -> list[Query]:
    """Keep a fraction of the *templates* (all variants of each kept template)."""
    templates = templates_of(queries)
    rng = random.Random(seed)
    k = max(1, round(len(templates) * frac))
    kept = set(rng.sample(templates, k))
    return [q for q in queries if q.template in kept]
