#!/usr/bin/env bash
# Fetch the 113 Join Order Benchmark queries (Leis et al., VLDB 2015) plus the IMDB schema,
# pinned to a fixed commit of github.com/gregrahn/join-order-benchmark.
#   queries      -> workload/job/*.sql
#   schema files -> data/raw/{schema,fkindexes}.sql
set -euo pipefail

COMMIT=a39603662e023e449cb2121997a5034df9e02ebf
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching JOB @ ${COMMIT:0:7}"
curl -fsSL "https://github.com/gregrahn/join-order-benchmark/archive/${COMMIT}.tar.gz" \
  | tar -xz -C "$TMP"
SRC="$TMP/join-order-benchmark-$COMMIT"

mkdir -p "$ROOT/workload/job" "$ROOT/data/raw"
cp "$SRC"/[0-9]*.sql "$ROOT/workload/job/"
cp "$SRC/schema.sql" "$SRC/fkindexes.sql" "$ROOT/data/raw/"

N=$(ls "$ROOT"/workload/job/[0-9]*.sql | wc -l | tr -d ' ')
echo "$N queries in workload/job/"
[ "$N" -eq 113 ] || { echo "expected 113 queries" >&2; exit 1; }
