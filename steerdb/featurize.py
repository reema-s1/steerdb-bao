"""Featurization: EXPLAIN JSON plan tree -> per-node vectors (Tree-CNN) or a flat summary (LightGBM).

Node vector layout:
    [ node-type one-hot | log1p(rows) z | log1p(cost) z | width z | relation one-hot | has_filter, has_index_cond ]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NODE_TYPES = (
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Hash Join",
    "Nested Loop",
    "Merge Join",
    "Hash",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
    "Sort",
    "Incremental Sort",
    "Aggregate",
    "Materialize",
    "Memoize",
    "Gather",
    "Gather Merge",
    "Limit",
    "Result",
    "Append",
    "Other",
)
JOIN_TYPES = frozenset({"Hash Join", "Nested Loop", "Merge Join"})

IMDB_TABLES = (
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
)

_TYPE_INDEX = {t: i for i, t in enumerate(NODE_TYPES)}
_TABLE_INDEX = {t: i for i, t in enumerate(IMDB_TABLES)}
_FILTER_KEYS = ("Filter", "Join Filter", "Hash Cond", "Merge Cond")
_INDEX_KEYS = ("Index Cond", "Recheck Cond")

N_TYPES = len(NODE_TYPES)
N_TABLES = len(IMDB_TABLES)
NUMERIC_OFFSET = N_TYPES
TABLE_OFFSET = N_TYPES + 3
FLAG_OFFSET = TABLE_OFFSET + N_TABLES
NODE_DIM = FLAG_OFFSET + 2


def node_type_index(node_type: str) -> int:
    return _TYPE_INDEX.get(node_type, _TYPE_INDEX["Other"])


def _raw_numeric(node: dict) -> tuple[float, float, float]:
    return (
        math.log1p(max(float(node.get("Plan Rows", 0.0)), 0.0)),
        math.log1p(max(float(node.get("Total Cost", 0.0)), 0.0)),
        float(node.get("Plan Width", 0.0)),
    )


def iter_nodes(plan: dict):
    yield plan
    for child in plan.get("Plans", []):
        yield from iter_nodes(child)


@dataclass
class Stats:
    """Mean/std for the three numeric node features, fitted on training plans only."""

    rows: tuple[float, float] = (0.0, 1.0)
    cost: tuple[float, float] = (0.0, 1.0)
    width: tuple[float, float] = (0.0, 1.0)

    @staticmethod
    def _z(v: float, ms: tuple[float, float]) -> float:
        return (v - ms[0]) / ms[1]


@dataclass
class TreeTensor:
    """One plan as a binary tree.

    x:   (N+1, NODE_DIM) node features; row 0 is the all-zero padding node.
    idx: (N, 3) int64 rows of (self, left, right) into x; 0 means "no child".
    """

    x: np.ndarray
    idx: np.ndarray


@dataclass
class Featurizer:
    stats: Stats = field(default_factory=Stats)
    fitted: bool = False

    # ---------------------------------------------------------------- fitting / io
    def fit(self, plans: list[dict]) -> Featurizer:
        vals = np.array([_raw_numeric(n) for p in plans for n in iter_nodes(p)], dtype=np.float64)
        if len(vals) == 0:
            raise ValueError("cannot fit featurizer on zero plans")
        mean, std = vals.mean(axis=0), vals.std(axis=0)
        std[std < 1e-6] = 1.0
        self.stats = Stats(*((float(m), float(s)) for m, s in zip(mean, std)))
        self.fitted = True
        return self

    def to_dict(self) -> dict:
        return {"rows": self.stats.rows, "cost": self.stats.cost, "width": self.stats.width}

    @classmethod
    def from_dict(cls, d: dict) -> Featurizer:
        return cls(Stats(tuple(d["rows"]), tuple(d["cost"]), tuple(d["width"])), fitted=True)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> Featurizer:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---------------------------------------------------------------- per-node
    def node_vector(self, node: dict) -> np.ndarray:
        v = np.zeros(NODE_DIM, dtype=np.float32)
        v[node_type_index(node.get("Node Type", "Other"))] = 1.0
        rows, cost, width = _raw_numeric(node)
        s = self.stats
        v[NUMERIC_OFFSET + 0] = Stats._z(rows, s.rows)
        v[NUMERIC_OFFSET + 1] = Stats._z(cost, s.cost)
        v[NUMERIC_OFFSET + 2] = Stats._z(width, s.width)
        rel = node.get("Relation Name")
        if rel in _TABLE_INDEX:
            v[TABLE_OFFSET + _TABLE_INDEX[rel]] = 1.0
        v[FLAG_OFFSET + 0] = float(any(k in node for k in _FILTER_KEYS))
        v[FLAG_OFFSET + 1] = float(any(k in node for k in _INDEX_KEYS))
        return v

    # ---------------------------------------------------------------- tree form
    def plan_to_tree(self, plan: dict) -> TreeTensor:
        vecs: list[np.ndarray] = [np.zeros(NODE_DIM, dtype=np.float32)]  # padding row 0
        triples: list[tuple[int, int, int]] = []

        def visit(vec: np.ndarray, children: list[dict]) -> int:
            me = len(vecs)
            vecs.append(vec)
            triples.append((me, 0, 0))
            slot = len(triples) - 1
            left = visit_node(children[0]) if len(children) >= 1 else 0
            if len(children) == 2:
                right = visit_node(children[1])
            elif len(children) > 2:
                # Binarize n-ary nodes (e.g. Append): extra children hang off a synthetic node.
                other = np.zeros(NODE_DIM, dtype=np.float32)
                other[_TYPE_INDEX["Other"]] = 1.0
                right = visit(other, children[1:])
            else:
                right = 0
            triples[slot] = (me, left, right)
            return me

        def visit_node(node: dict) -> int:
            return visit(self.node_vector(node), node.get("Plans", []))

        visit_node(plan)
        return TreeTensor(np.stack(vecs), np.asarray(triples, dtype=np.int64))

    # ---------------------------------------------------------------- flat form
    def flat_features(self, plan: dict) -> np.ndarray:
        return flat_features(plan)


FLAT_NAMES = [f"count[{t}]" for t in NODE_TYPES] + [
    "n_nodes",
    "depth",
    "n_joins",
    "root_log_cost",
    "root_log_rows",
    "sum_log_rows",
    "max_log_rows",
    "sum_join_log_rows",
    "max_join_log_rows",
    "n_filters",
    "n_index_conds",
]


def _depth(node: dict) -> int:
    return 1 + max((_depth(c) for c in node.get("Plans", [])), default=0)


def flat_features(plan: dict) -> np.ndarray:
    """Flattened plan summary for the LightGBM baseline (no standardization needed)."""
    counts = np.zeros(N_TYPES, dtype=np.float32)
    log_rows, join_log_rows = [], []
    n_filters = n_index = 0
    for node in iter_nodes(plan):
        nt = node.get("Node Type", "Other")
        counts[node_type_index(nt)] += 1
        lr = math.log1p(max(float(node.get("Plan Rows", 0.0)), 0.0))
        log_rows.append(lr)
        if nt in JOIN_TYPES:
            join_log_rows.append(lr)
        n_filters += any(k in node for k in _FILTER_KEYS)
        n_index += any(k in node for k in _INDEX_KEYS)
    root_rows, root_cost, _ = _raw_numeric(plan)
    extra = [
        len(log_rows),
        _depth(plan),
        len(join_log_rows),
        root_cost,
        root_rows,
        sum(log_rows),
        max(log_rows),
        sum(join_log_rows),
        max(join_log_rows, default=0.0),
        n_filters,
        n_index,
    ]
    return np.concatenate([counts, np.asarray(extra, dtype=np.float32)])
