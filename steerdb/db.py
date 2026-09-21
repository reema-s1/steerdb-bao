"""Database connection helper."""

from __future__ import annotations

import psycopg

from . import config


def connect(dsn: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(dsn or config.DSN, autocommit=True)
    for guc, value in config.SESSION_SETTINGS.items():
        conn.execute(f"SET {guc} = {value}")
    return conn


def apply_arm(conn: psycopg.Connection, arm) -> None:
    for stmt in arm.set_statements():
        conn.execute(stmt)


def set_timeout(conn: psycopg.Connection, timeout_ms: int | None) -> None:
    conn.execute(f"SET statement_timeout = {int(timeout_ms or 0)}")
