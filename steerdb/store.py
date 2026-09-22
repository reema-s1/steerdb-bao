"""Experience store: every (query, arm, plan, latency) observation, in SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_name  TEXT    NOT NULL,
    arm         INTEGER NOT NULL,
    plan_hash   TEXT    NOT NULL,
    plan_json   TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL,
    timed_out   INTEGER NOT NULL DEFAULT 0,
    source      TEXT    NOT NULL,          -- bootstrap | online | run
    epoch       INTEGER,
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_exec_query ON executions(query_name, arm, source);
"""


@dataclass
class Observation:
    query_name: str
    arm: int
    plan_hash: str
    plan: dict
    latency_ms: float
    timed_out: bool
    source: str
    epoch: int | None = None


class ExperienceStore:
    def __init__(self, path: Path | str | None = None) -> None:
        path = str(path or config.STORE_PATH)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ writes
    def add(self, obs: Observation) -> None:
        self.db.execute(
            "INSERT INTO executions (query_name, arm, plan_hash, plan_json, latency_ms, timed_out,"
            " source, epoch, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                obs.query_name,
                obs.arm,
                obs.plan_hash,
                json.dumps(obs.plan),
                obs.latency_ms,
                int(obs.timed_out),
                obs.source,
                obs.epoch,
                time.time(),
            ),
        )
        self.db.commit()

    def backup_to(self, path: Path | str) -> None:
        """Consistent snapshot of the store (e.g. to Google Drive), written atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        dst = sqlite3.connect(str(tmp))
        try:
            self.db.backup(dst)
        finally:
            dst.close()
        os.replace(tmp, path)

    def delete_source(self, source: str) -> None:
        self.db.execute("DELETE FROM executions WHERE source = ?", (source,))
        self.db.commit()

    # ------------------------------------------------------------------ reads
    def _rows(self, where: str = "", params: tuple = ()) -> list[Observation]:
        cur = self.db.execute(
            "SELECT query_name, arm, plan_hash, plan_json, latency_ms, timed_out, source, epoch"
            f" FROM executions {where} ORDER BY id",
            params,
        )
        return [
            Observation(q, a, h, json.loads(p), lat, bool(t), s, e)
            for q, a, h, p, lat, t, s, e in cur.fetchall()
        ]

    def observations(
        self, sources: tuple[str, ...] | None = None, queries: set[str] | None = None
    ) -> list[Observation]:
        if sources:
            marks = ",".join("?" * len(sources))
            rows = self._rows(f"WHERE source IN ({marks})", tuple(sources))
        else:
            rows = self._rows()
        if queries is not None:
            rows = [r for r in rows if r.query_name in queries]
        return rows

    def has(self, query_name: str, arm: int, source: str = "bootstrap") -> bool:
        cur = self.db.execute(
            "SELECT 1 FROM executions WHERE query_name = ? AND arm = ? AND source = ? LIMIT 1",
            (query_name, arm, source),
        )
        return cur.fetchone() is not None

    def bootstrap_table(self) -> dict[str, dict[int, Observation]]:
        """query -> arm -> the latest bootstrap observation. The basis of offline evaluation."""
        table: dict[str, dict[int, Observation]] = {}
        for obs in self._rows("WHERE source = ?", ("bootstrap",)):
            table.setdefault(obs.query_name, {})[obs.arm] = obs
        return table

    def baseline_ms(self, query_name: str) -> float | None:
        """Historical arm-0 latency for a query (used for the safety timeout)."""
        cur = self.db.execute(
            "SELECT latency_ms FROM executions WHERE query_name = ? AND arm = 0"
            " AND timed_out = 0 ORDER BY id DESC LIMIT 1",
            (query_name,),
        )
        row = cur.fetchone()
        return row[0] if row else None
