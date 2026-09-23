# steerdb — Bao-style learned query-plan steering for PostgreSQL

A Tree-CNN over `EXPLAIN` plans (PyTorch) that picks which planner hint set to run each query
under, following **Bao** (Marcus et al., SIGMOD 2021) and the tree-convolution model of **Neo**
(VLDB 2019), with a safety guard and a cross-validated evaluation harness.

**Headline result (honest):** on the Join Order Benchmark the best-arm ceiling is only **8.5%**,
so steering breaks even (**1.04× of stock Postgres**); on a harder cardinality-stressing
workload the ceiling is **68.5%** and a single query is **33× faster** under the right hint set,
but the learner does not yet capture it from ~150 training queries. The safety guard bounds a
wrong choice at ~3× instead of the **2.5×-slower** behaviour an unguarded model showed.
Full numbers: [docs/results.md](docs/results.md).

steerdb doesn't replace the Postgres optimizer. For each query it asks Postgres for up to 8
candidate plans, one per **hint set** (for example `enable_nestloop=off`), then uses a small
learned model to predict which plan will actually run fastest. Every candidate is still a real
Postgres plan, so a bad choice has a bounded cost. A safety guard falls back to the stock plan
when the model isn't confident or a chosen plan runs too long.

```
SQL ─▶ plan under K hint sets (EXPLAIN, no execution) ─▶ featurize plan trees
    ─▶ Tree-CNN predicts latency (+ uncertainty) ─▶ pick (argmin / Thompson) + safety guard
    ─▶ execute ─▶ record (plan, latency) ─▶ periodic retrain
```

Results, ablations and figures are in [docs/results.md](docs/results.md).

## Quickstart

Requirements: Docker, Python 3.10+, bash, and about 12 GB of free disk for IMDB.

```bash
pip install -e ".[dev]"                     # CPU-only torch is enough

docker compose up -d                        # Postgres 16 on localhost:5433, fixed config
bash data/load_imdb.sh                      # fetch JOB + IMDB (~1.2 GB download), load, index, ANALYZE

steerdb collect                             # Phase 0: 113 queries x 8 arms (resumable, hours)
steerdb oracle-gap                          # go/no-go: is the best arm meaningfully faster?
steerdb train --model lgbm                  # Phase 1 baseline
steerdb train --model treecnn               # Phase 2 main model
steerdb run --file workload/job/29a.sql     # route one query end to end
python experiments/run_eval.py --overhead   # every table and figure in docs/results.md
```

Every command reads `STEERDB_DSN` (default `postgresql://postgres:steerdb@localhost:5433/imdb`).
All generated state goes to `runs/`: the experience store `runs/experience.sqlite` and models
in `runs/models/`.

### Smoke test without the 3.6 GB dataset

`data/make_synthetic_imdb.py` builds a small database with the same 21-table schema, the real
dimension values that JOB predicates look for, and skewed foreign keys. The whole pipeline runs
on it in a few minutes. CI does this on every push. The numbers it produces say nothing about
real performance.

```bash
bash workload/fetch_job.sh
python data/make_synthetic_imdb.py --scale 0.5
steerdb collect --warmup 0 --runs 1 --timeout-ms 5000
steerdb bench --no-ablations --out runs/eval --markdown runs/results.md
```

### Running everything on Google Colab (no local disk or Docker needed)

[notebooks/run_on_colab.ipynb](notebooks/run_on_colab.ipynb) runs the full pipeline on the
real IMDB/JOB data on a free Colab CPU runtime:

1. Installs Postgres 16 with [scripts/colab_postgres.sh](scripts/colab_postgres.sh), using the
   same settings as `docker-compose.yml`.
2. Loads IMDB with `PG_MODE=local bash data/load_imdb.sh`.
3. Generates CEB queries (see [Workloads](#workloads)) and keeps them in Drive.
4. Runs `steerdb collect`, then `train` and `experiments/run_eval.py`.

A settings cell at the top controls two runs. The **pilot** (5 CEB queries per template)
stops after printing the oracle gap per workload. The **full run** (`CEB_PER_TEMPLATE = 60`,
`RUN_EVALUATION = True`) collects only the new queries and then trains and evaluates.

Anything that must survive a session reset lives in `MyDrive/steerdb/`: the cached 1.2 GB
download, the experience store (`collect --backup` copies it there after every query), models,
and results. After a disconnect, choose *Run all* again. IMDB is reloaded (about 30 min) and
collection resumes where it stopped. Colab VMs are shared, so latencies are noisier than on a
quiet dedicated machine; the report records the machine it ran on.

### Training on a free GPU (Colab / Kaggle)

Training needs only the experience store, not Postgres. Collect locally, then train anywhere.
[notebooks/train_gpu.ipynb](notebooks/train_gpu.ipynb) doesn't reimplement anything: it clones the
repo (or unpacks a zip made with `git archive -o steerdb.zip HEAD`), runs `pip install -e .`, and
calls the CLI:

```bash
steerdb train --model treecnn --device cuda      # --device auto (default) picks the GPU if present
steerdb bench --out runs/eval                     # every Tree-CNN in the evaluation uses the GPU
```

Upload `runs/experience.sqlite` to the notebook, then download `runs/models/` back into `runs/`.
Weights are saved as CPU tensors, so a GPU-trained model loads on a laptop without CUDA.

## Workloads

- **JOB** (113 queries, 33 templates; Leis et al., VLDB 2015), fetched by
  `workload/fetch_job.sh` at a pinned commit.
- **CEB-style** queries from the 8 complex (7-16 table) IMDB templates of the Cardinality
  Estimation Benchmark (Negi et al.; MIT license). CEB's pre-generated query files are no
  longer downloadable (dead Dropbox links), so [data/gen_ceb_queries.py](data/gen_ceb_queries.py)
  runs CEB's own generator (`data/fetch_ceb.sh`, pinned commit) against the loaded IMDB
  database. It samples real predicate values, writes `workload/ceb/ceb<template>_<n>.sql`, and
  only ever adds queries, so a pilot set stays fixed when the workload grows. It also patches
  two missing upstream imports and caps the generator's retry loop.

Point any command at several workloads with `--workload workload/job,workload/ceb` (or the
`STEERDB_WORKLOAD` variable). One experience store holds both workloads.

Why CEB: on JOB with Postgres 16, foreign-key indexes and a warm cache, stock Postgres was
already the best arm for 105 of 113 queries. The best-arm oracle was only 8.5% faster in total,
below this project's 15% go/no-go bar. CEB was designed to produce the cardinality
misestimates that make plan choice matter.

## Components

| Module | Role |
|---|---|
| [steerdb/arms.py](steerdb/arms.py) | The 8 hint sets. Each arm resets every planner GUC it doesn't set, so arms never leak into each other |
| [steerdb/plan_gen.py](steerdb/plan_gen.py) | `EXPLAIN (FORMAT JSON)` per arm, deduplicated by a structural plan hash that ignores cost numbers |
| [steerdb/featurize.py](steerdb/featurize.py) | Plan node → vector (node type, standardized log rows/cost, width, relation, filter flags); plan → binary tree tensor; flat summary for LightGBM |
| [steerdb/models/lgbm_baseline.py](steerdb/models/lgbm_baseline.py) | Phase 1: LightGBM ensemble on the flat summary |
| [steerdb/models/tree_conv.py](steerdb/models/tree_conv.py) | Phase 2: Tree-CNN (3 tree-conv layers → dynamic max-pool → 2 FC), 161k parameters, ensemble for uncertainty |
| [steerdb/selector.py](steerdb/selector.py) | Greedy argmin or Thompson sampling, plus the safety guard |
| [steerdb/executor.py](steerdb/executor.py) | Runs a query under an arm with a timeout (warm-up + median of 3; timeouts become censored labels) |
| [steerdb/store.py](steerdb/store.py) | SQLite experience store |
| [steerdb/router.py](steerdb/router.py) | Inference path used by `steerdb run` |
| [steerdb/online.py](steerdb/online.py) | Online loop: choose via Thompson sampling, execute, record, retrain every 25 queries. Runs live or as a fast replay of Phase 0 latencies |
| [steerdb/evaluate.py](steerdb/evaluate.py), [steerdb/report.py](steerdb/report.py) | Offline evaluation of all policies, metrics, ablations, markdown report |

### Hint sets

| Arm | Setting | Intent |
|---|---|---|
| 0 | default | baseline Postgres |
| 1 | `enable_nestloop=off` | avoid nested loops when row estimates are too low |
| 2 | `enable_hashjoin=off` | force merge or nested loop |
| 3 | `enable_mergejoin=off` | hash or nested loop only |
| 4 | `enable_nestloop=off, enable_mergejoin=off` | hash joins only |
| 5 | `enable_hashjoin=off, enable_mergejoin=off` | nested loops only |
| 6 | `enable_seqscan=off` | favor index scans |
| 7 | `enable_indexscan=off, enable_bitmapscan=off` | favor sequential scans |

### Safety guard

1. An untrained model always picks arm 0.
2. **Pessimistic deviation.** The router leaves arm 0 only if the predicted speedup still
   exceeds 5% after subtracting one standard deviation of the ensemble's disagreement:
   `(mu_0 - mu_c) - k * sqrt(sd_c^2 + sd_0^2) > log(1.05)`, with `k = 1`. Both values were
   chosen by leave-template-out cross-validation on training templates only. On JOB this guard
   brought the Tree-CNN from 2.47× slower than Postgres to 1.01×.
3. A non-default plan runs with `statement_timeout = 2 × arm-0 latency`. If it times out, it
   is cancelled, arm 0 is rerun, and the failure is recorded as a negative example. The offline
   evaluation simulates this cost exactly (timeout + arm-0 latency). An earlier 1-second
   minimum timeout turned small mistakes on 100 ms queries into 10× slowdowns, so it was removed.
4. Every test query where steerdb is more than 20% slower than Postgres is listed in the results.

## Evaluation methodology

- **Unseen templates, cross-validated.** Templates are shuffled and dealt into 5 folds. Each
  fold is predicted by models trained only on the other folds, so every query is evaluated once
  by a model that never saw its template. CEB queries are grouped by join graph (`ceb9a_*` and
  `ceb9b_*` share a fold). Variants such as `16a`/`16b`/`16c` are near-duplicates, so a random
  split would leak; the report shows that leaky split next to the honest one.
- **Why not one fixed split.** The conventional split (train JOB 1–25, test 26–33) turned out
  lopsided: JOB's templates grow with their number, so 18 of the 23 test queries join 12 or
  more tables, versus 2 of the 90 training queries. That tests extrapolation to unseen plan
  shapes, not the typical case, and it leaves just 23 queries to measure with.
- **Offline policy evaluation.** Phase 0 executes every query under every arm, so any policy
  (stock, oracle, random, cost-only, LightGBM, Tree-CNN) can be scored from recorded latencies
  without rerunning queries.
- **Baselines:** stock Postgres (arm 0), best-arm oracle (the upper bound), random arm,
  cost-only re-ranking (lowest optimizer cost), and LightGBM.
- **Metrics:** total workload time; p50/p95/p99; regret vs. the oracle; per-query speedup
  distribution; number of queries more than 20% slower than Postgres; planning and inference
  overhead.
- **Ablations:** number of arms (2/4/8), amount of training data (25/50/100% of templates),
  Thompson vs. greedy online learning, and seen vs. unseen templates.

### Measurement setup

Fixed server settings are in [docker-compose.yml](docker-compose.yml): `shared_buffers=2GB`,
`work_mem=64MB`, `effective_cache_size=4GB`, `random_page_cost=1.1`. Every connection also pins
three settings, whatever the server config ([steerdb/config.py](steerdb/config.py)):

- `geqo_seed=0`: queries with 12 or more relations go through the genetic optimizer, which is deterministic for a fixed seed. Exhaustive search on JOB's 17-way joins took over 11 s to plan across 8 arms.
- `max_parallel_workers_per_gather=0` and `jit=off`: less timing noise.

Labels come from warm-cache runs: 1 discarded warm-up, then the median of 3. The statement
timeout is 5 minutes. Use a single client with nothing else running. Record your hardware in
`docs/results.md` (the report fills in platform and CPU automatically).

## CLI

```
steerdb [--workload DIRS] collect [--queries 1a,2b] [--arms 0,1,4] [--warmup 1] [--runs 3] [--timeout-ms 300000] [--backup PATH]
steerdb oracle-gap  [--queries ...]
steerdb train       --model lgbm|treecnn [--epochs N] [--seed S] [--out DIR] [--device auto|cpu|cuda]
steerdb run         "SQL" | --file q.sql  [--model DIR] [--min-gain 0.05] [--no-execute]
steerdb online      [--model treecnn] [--mode thompson|greedy] [--online-epochs 5] [--retrain-every 25] [--live]
steerdb bench       [--model treecnn] [--folds 5] [--no-ablations] [--overhead] [--out DIR] [--markdown FILE]
```

## Development

```bash
pytest -q                 # unit tests, no database needed
ruff check . && ruff format --check .
```

The tests cover the featurization worked example, arm definitions, plan-hash
deduplication, selector and guard behavior, Tree-CNN mechanics (child gathering, batching,
per-plan pooling), model save/load, the offline evaluator (including the simulated timeout
fallback), and the online loop on a synthetic latency table with known headroom.

## Limitations

- Hint sets only reshape what Postgres already considers, so the best-arm oracle caps the gain.
  On JOB (Postgres 16, warm cache) that cap was only 8.5%.
  Check `steerdb oracle-gap` before trusting any model result.
- About 900 labelled executions is a small dataset. Ensembles and the LightGBM baseline are
  there as sanity checks against overfitting.
- Read-only, single-node workloads only. No DDL or updates, and no C extension.

## References

- R. Marcus et al., *Bao: Making Learned Query Optimization Practical*, SIGMOD 2021
- R. Marcus et al., *Neo: A Learned Query Optimizer*, VLDB 2019
- Z. Yang et al., *Balsa: Learning a Query Optimizer Without Expert Demonstrations*, SIGMOD 2022
- V. Leis et al., *How Good Are Query Optimizers, Really?*, VLDB 2015 (Join Order Benchmark)
- P. Negi et al., *Flow-Loss: Learning Cardinality Estimates That Matter*, VLDB 2021 (introduces CEB); templates and generator from github.com/learnedsystems/CEB
- L. Mou et al., *Convolutional Neural Networks over Tree Structures for Programming Language Processing*, AAAI 2016
