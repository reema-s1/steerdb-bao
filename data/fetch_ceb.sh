#!/usr/bin/env bash
# Fetch the CEB repo (Cardinality Estimation Benchmark, MIT license) at a pinned commit and
# install what its query generator needs. Queries are then generated against the loaded IMDB
# database with data/gen_ceb_queries.py (CEB's pre-generated query files are no longer online).
set -euo pipefail

COMMIT=9eaaadbdeaba6ae48cbf80b39602b483525ddcf7
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/raw/CEB"

if [ ! -d "$DEST/.git" ]; then
  git clone -q https://github.com/learnedsystems/CEB.git "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout -q "$COMMIT"
python -m pip install -q pygtrie klepto sqlparse networkx psycopg2-binary toml scikit-learn
echo "CEB @ ${COMMIT:0:7} in $DEST"
