#!/usr/bin/env bash
# Load the IMDB dataset used by the Join Order Benchmark into the docker-compose Postgres.
#
#   bash data/load_imdb.sh
#
# Needs ~12 GB free disk (1.2 GB download, 3.6 GB CSV, ~6 GB in Postgres with indexes).
# Idempotent: the public schema is dropped and recreated on every run.
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash on Windows: don't rewrite /data/raw paths

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
URL="${IMDB_URL:-http://event.cwi.nl/da/job/imdb.tgz}"
TABLES="aka_name aka_title cast_info char_name comp_cast_type company_name company_type
complete_cast info_type keyword kind_type link_type movie_companies movie_info movie_info_idx
movie_keyword movie_link name person_info role_type title"

cd "$ROOT"
mkdir -p "$RAW"

[ -f "$RAW/schema.sql" ] && [ -f workload/job/1a.sql ] || bash workload/fetch_job.sh

if [ ! -f "$RAW/title.csv" ]; then
  [ -f "$RAW/imdb.tgz" ] || { echo "downloading $URL"; curl -fL -o "$RAW/imdb.tgz" "$URL"; }
  echo "extracting"
  tar -xzf "$RAW/imdb.tgz" -C "$RAW"
  if [ ! -f "$RAW/title.csv" ]; then  # archive may contain a sub-directory
    mv "$(dirname "$(find "$RAW" -name title.csv | head -1)")"/*.csv "$RAW/"
  fi
fi

docker compose up -d --wait postgres
PSQL=(docker compose exec -T postgres psql -U postgres -d imdb -v ON_ERROR_STOP=1 -q)

echo "creating schema"
{ echo "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; cat "$RAW/schema.sql"; } | "${PSQL[@]}"

for t in $TABLES; do
  echo "loading $t"
  echo "COPY $t FROM '/data/raw/$t.csv' WITH (FORMAT csv, ESCAPE '\\');" | "${PSQL[@]}"
done

echo "creating foreign-key indexes"
"${PSQL[@]}" < "$RAW/fkindexes.sql"
echo "VACUUM ANALYZE"
echo "VACUUM ANALYZE;" | "${PSQL[@]}"

echo "row counts:"
for t in $TABLES; do
  printf '  %-16s %s\n' "$t" "$(echo "SELECT count(*) FROM $t;" | "${PSQL[@]}" -tA)"
done
echo "done. DSN: postgresql://postgres:steerdb@localhost:5433/imdb"
