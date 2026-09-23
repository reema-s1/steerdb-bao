"""Central configuration. Everything overridable through environment variables."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DSN = os.environ.get("STEERDB_DSN", "postgresql://postgres:steerdb@localhost:5433/imdb")
RUNS_DIR = Path(os.environ.get("STEERDB_RUNS", REPO_ROOT / "runs"))
STORE_PATH = Path(os.environ.get("STEERDB_STORE", RUNS_DIR / "experience.sqlite"))
MODELS_DIR = RUNS_DIR / "models"
# One or more query directories, separated by "," (e.g. "workload/job,workload/ceb").
WORKLOAD_DIR = os.environ.get("STEERDB_WORKLOAD", str(REPO_ROOT / "workload" / "job"))

# Session settings pinned on every connection, independent of server config, so that
# plans are deterministic and latencies are comparable:
#  - geqo_seed pinned: queries with >= geqo_threshold (12) relations use the genetic optimizer,
#    which is deterministic for a fixed seed. (Exhaustive search on JOB's 17-way joins costs
#    seconds of planning per arm, x8 arms -- far too slow for an inference path.)
#  - no parallel workers / no JIT -> less timing noise
SESSION_SETTINGS: dict[str, str] = {
    "geqo_seed": "0",
    "max_parallel_workers_per_gather": "0",
    "jit": "off",
}

# Measurement hygiene: warm cache, discard a warm-up run, take the median
WARMUP_RUNS = int(os.environ.get("STEERDB_WARMUP", "1"))
MEASURED_RUNS = int(os.environ.get("STEERDB_RUNS_PER_QUERY", "3"))
STATEMENT_TIMEOUT_MS = int(os.environ.get("STEERDB_TIMEOUT_MS", str(5 * 60 * 1000)))

# Template split: train on templates 1-25, test on unseen templates 26-33.
TEST_TEMPLATES = tuple(str(t) for t in range(26, 34))
