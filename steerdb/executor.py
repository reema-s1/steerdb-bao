"""Execute a query under an arm with a timeout and measure wall-clock latency."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import psycopg
from psycopg import errors

from . import config
from .arms import DEFAULT_ARM, Arm
from .db import apply_arm, set_timeout


@dataclass
class ExecResult:
    latency_ms: float  # median of measured runs, or the timeout value if censored
    timed_out: bool
    runs_ms: list[float]


def _run_once(conn: psycopg.Connection, sql: str) -> float:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000


def execute(
    conn: psycopg.Connection,
    sql: str,
    arm: Arm,
    timeout_ms: int | None = None,
    warmup: int = config.WARMUP_RUNS,
    runs: int = config.MEASURED_RUNS,
) -> ExecResult:
    """Warm-up runs are discarded; the label is the median of `runs` measured runs.

    A timed-out query is recorded with latency = timeout (a censored label).
    """
    timeout_ms = timeout_ms or config.STATEMENT_TIMEOUT_MS
    apply_arm(conn, arm)
    set_timeout(conn, timeout_ms)
    measured: list[float] = []
    try:
        for i in range(warmup + runs):
            ms = _run_once(conn, sql)
            if i >= warmup:
                measured.append(ms)
    except errors.QueryCanceled:
        return ExecResult(float(timeout_ms), True, measured)
    finally:
        set_timeout(conn, 0)
        apply_arm(conn, DEFAULT_ARM)
    return ExecResult(statistics.median(measured), False, measured)
