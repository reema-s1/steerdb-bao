"""Load the JOB workload and split it by template (never randomly, see design doc section 5)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from . import config

_NAME_RE = re.compile(r"^(\d+)([a-z]*)$")


@dataclass(frozen=True)
class Query:
    name: str  # e.g. "16b"
    template: int  # e.g. 16
    sql: str


def parse_name(name: str) -> tuple[int, str]:
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"not a JOB query name: {name!r}")
    return int(m.group(1)), m.group(2)


def load_workload(directory: Path | str | None = None) -> list[Query]:
    directory = Path(directory or config.WORKLOAD_DIR)
    queries = []
    for path in directory.glob("*.sql"):
        try:
            template, _ = parse_name(path.stem)
        except ValueError:
            continue  # schema.sql, fkindexes.sql, ...
        sql = path.read_text(encoding="utf-8").strip().rstrip(";")
        queries.append(Query(path.stem, template, sql))
    if not queries:
        raise FileNotFoundError(
            f"no JOB queries found in {directory}; run `bash workload/fetch_job.sh` first"
        )
    return sorted(queries, key=lambda q: (q.template, q.name))


def split_by_template(
    queries: list[Query], test_templates: tuple[int, ...] = config.TEST_TEMPLATES
) -> tuple[list[Query], list[Query]]:
    test_set = set(test_templates)
    train = [q for q in queries if q.template not in test_set]
    test = [q for q in queries if q.template in test_set]
    return train, test


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
    templates = sorted({q.template for q in queries})
    rng = random.Random(seed)
    k = max(1, round(len(templates) * frac))
    kept = set(rng.sample(templates, k))
    return [q for q in queries if q.template in kept]
