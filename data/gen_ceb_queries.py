"""Generate CEB-style queries (Cardinality Estimation Benchmark, Negi et al.) from CEB's templates.

CEB's pre-generated query files are no longer downloadable (dead Dropbox links), but its
templates and generator are in the MIT-licensed repo. This driver runs CEB's own
`QueryGenerator` against the loaded IMDB database (it samples predicate values from the
data) and writes plain SQL files that `steerdb` loads like JOB:

    workload/ceb/ceb<template>_<n>.sql   e.g. ceb3b_07.sql (CEB template 3b, sample 7)

    bash data/fetch_ceb.sh                       # clone CEB (pinned) + install its generator deps
    python data/gen_ceb_queries.py --per-template 5 --out workload/ceb_pilot
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from steerdb import config  # noqa: E402

# The complex (7-16 table) CEB-IMDb templates published in the CEB repo. The repo's other set,
# templates/simple_imdb_train_all, is 3-table joins only and has no plan-choice headroom.
COMPLEX_TEMPLATES = ("3b", "7a", "8a", "9a", "9b", "10a", "11a", "11b")


def _scale_thresholds(obj, scale: float):
    """Multiply every `thresholds` / `min_count` in a template (for tiny test databases)."""
    if scale == 1.0:
        return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "thresholds":
                obj[k] = [max(1, int(x * scale)) for x in v]
            elif k == "min_count":
                obj[k] = max(1, int(v * scale))
            else:
                _scale_thresholds(v, scale)
    elif isinstance(obj, list):
        for v in obj:
            _scale_thresholds(v, scale)
    return obj


class GaveUp(Exception):
    pass


def _cap_attempts(qg, max_attempts: int) -> None:
    """Upstream retries forever when a template can't be satisfied; stop after max_attempts."""
    inner = qg._gen_query_str
    count = 0

    def wrapped(preds):
        nonlocal count
        count += 1
        if count > max_attempts:
            raise GaveUp
        return inner(preds)

    qg._gen_query_str = wrapped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dsn", default=config.DSN)
    ap.add_argument("--ceb", default=str(ROOT / "data" / "raw" / "CEB"), help="CEB checkout")
    ap.add_argument("--templates", default="all", help="e.g. 3b,7a (default: all 8)")
    ap.add_argument("--per-template", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "workload" / "ceb"))
    ap.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="testing only: scale the templates' target row counts to a small (synthetic) DB",
    )
    args = ap.parse_args()

    ceb = Path(args.ceb)
    sys.path.insert(0, str(ceb))
    import query_gen.query_generator as qgen
    import toml

    # Upstream commented out `from nltk.tokenize import word_tokenize` but still calls it for
    # ILIKE predicates; a regex tokenizer is equivalent for picking substrings.
    qgen.word_tokenize = lambda text: re.findall(r"\w+|[^\w\s]", text)
    qgen.np = np  # also used without being imported upstream
    QueryGenerator = qgen.QueryGenerator

    dsn = urlparse(args.dsn)
    db = dict(
        user=dsn.username or "postgres",
        db_host=dsn.hostname or "localhost",
        port=str(dsn.port or 5432),
        pwd=dsn.password or "",
        db_name=dsn.path.lstrip("/") or "imdb",
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = None if args.templates == "all" else set(args.templates.split(","))

    total_new = 0
    for name in COMPLEX_TEMPLATES:
        if wanted and name not in wanted:
            continue
        # Incremental: keep existing queries (e.g. the pilot's) and only add new distinct ones up
        # to --per-template. Regenerating from scratch would not reproduce them, because some
        # templates sample values with Postgres's random().
        existing = sorted(out.glob(f"ceb{name}_*.sql"))
        have = {p.read_text(encoding="utf-8").strip() for p in existing}
        need = args.per_template - len(existing)
        if need <= 0:
            print(f"  {name}: already has {len(existing)} queries", flush=True)
            continue
        # Seed per template and per starting size: independent of other templates, and new
        # draws when extending an existing set.
        random.seed(f"{args.seed}-{name}-{len(existing)}")
        fn = next((ceb / "templates" / name).glob("*.toml"))
        cwd = os.getcwd()
        os.chdir(ceb)  # CEB's helpers assume the repo root as working directory
        try:
            template = _scale_thresholds(toml.load(fn), args.threshold_scale)
            qg = QueryGenerator(
                template, db["user"], db["db_host"], db["port"], db["pwd"], db["db_name"]
            )
            _cap_attempts(qg, max_attempts=200 * need + 2000)
            # Over-generate a little: duplicates of existing queries are dropped below.
            sqls = qg.gen_queries(need + max(2, need // 5))
        except GaveUp:
            print(f"  {name}: SKIPPED, no valid predicates found on this database", flush=True)
            continue
        except Exception:  # research code: report and keep going with the other templates
            print(f"  {name}: SKIPPED, generator error:", flush=True)
            traceback.print_exc()
            continue
        finally:
            os.chdir(cwd)
        fresh = [s.strip().rstrip(";") for s in dict.fromkeys(sqls)]
        fresh = [s for s in fresh if s not in have][:need]
        for i, sql in enumerate(fresh, len(existing) + 1):
            (out / f"ceb{name}_{i:02d}.sql").write_text(sql + "\n", encoding="utf-8")
        total_new += len(fresh)
        print(f"  {name}: {len(existing)} existing + {len(fresh)} new", flush=True)
    print(f"wrote {total_new} new queries to {out}")


if __name__ == "__main__":
    main()
