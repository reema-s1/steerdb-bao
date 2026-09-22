#!/usr/bin/env bash
# Load the IMDB dataset used by the Join Order Benchmark into Postgres.
#
#   bash data/load_imdb.sh                    # into the docker-compose Postgres (default)
#   PG_MODE=local bash data/load_imdb.sh      # into any reachable Postgres via psql + $STEERDB_DSN
#                                             # (no Docker; used by the Colab notebook)
#
# Optional environment:
#   IMDB_TGZ    path to an already-downloaded imdb.tgz (e.g. cached on Google Drive)
#   IMDB_RAW    directory for the extracted CSVs (default data/raw; local mode only)
#   DELETE_CSV  1 = delete the CSVs after loading to free ~3.6 GB
#
# Needs ~12 GB free disk (1.2 GB download, 3.6 GB CSV, ~6 GB in Postgres with indexes).
# Idempotent: the public schema is dropped and recreated on every run.
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash on Windows: don't rewrite /data/raw paths

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCHEMA_DIR="$ROOT/data/raw"
PG_MODE="${PG_MODE:-docker}"
RAW="${IMDB_RAW:-$ROOT/data/raw}"
TGZ="${IMDB_TGZ:-$RAW/imdb.tgz}"
URL="${IMDB_URL:-http://event.cwi.nl/da/job/imdb.tgz}"
TABLES="aka_name aka_title cast_info char_name comp_cast_type company_name company_type
complete_cast info_type keyword kind_type link_type movie_companies movie_info movie_info_idx
movie_keyword movie_link name person_info role_type title"

cd "$ROOT"
mkdir -p "$RAW" "$SCHEMA_DIR"

[ -f "$SCHEMA_DIR/schema.sql" ] && [ -f workload/job/1a.sql ] || bash workload/fetch_job.sh

if [ ! -f "$RAW/title.csv" ]; then
  if [ ! -f "$TGZ" ]; then
    echo "downloading $URL -> $TGZ"
    mkdir -p "$(dirname "$TGZ")"
    curl -fL -o "$TGZ.part" "$URL" && mv "$TGZ.part" "$TGZ"
  fi
  echo "extracting into $RAW"
  tar -xzf "$TGZ" -C "$RAW"
  if [ ! -f "$RAW/title.csv" ]; then  # archive may contain a sub-directory
    mv "$(dirname "$(find "$RAW" -name title.csv | head -1)")"/*.csv "$RAW/"
  fi
fi

case "$PG_MODE" in
  docker)
    [ "$RAW" = "$ROOT/data/raw" ] || { echo "IMDB_RAW is only supported with PG_MODE=local" >&2; exit 1; }
    docker compose up -d --wait postgres
    PSQL=(docker compose exec -T postgres psql -U postgres -d imdb -v ON_ERROR_STOP=1 -q)
    CSV_PATH=/data/raw  # mounted into the container
    COPY_CMD="COPY"
    DSN_MSG="postgresql://postgres:steerdb@localhost:5433/imdb"
    ;;
  local)
    DSN="${STEERDB_DSN:?set STEERDB_DSN for PG_MODE=local}"
    PSQL=(psql "$DSN" -v ON_ERROR_STOP=1 -q)
    CSV_PATH="$RAW"
    COPY_CMD='\copy'  # client-side: the server needs no access to the files
    DSN_MSG="$DSN"
    ;;
  *) echo "PG_MODE must be docker or local" >&2; exit 1 ;;
esac

echo "creating schema"
{ echo "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; cat "$SCHEMA_DIR/schema.sql"; } | "${PSQL[@]}"

for t in $TABLES; do
  echo "loading $t"
  echo "$COPY_CMD $t FROM '$CSV_PATH/$t.csv' WITH (FORMAT csv, ESCAPE '\\');" | "${PSQL[@]}"
done

echo "creating foreign-key indexes"
"${PSQL[@]}" < "$SCHEMA_DIR/fkindexes.sql"
echo "VACUUM ANALYZE"
echo "VACUUM ANALYZE;" | "${PSQL[@]}"

echo "row counts:"
for t in $TABLES; do
  printf '  %-16s %s\n' "$t" "$(echo "SELECT count(*) FROM $t;" | "${PSQL[@]}" -tA)"
done

if [ "${DELETE_CSV:-0}" = "1" ]; then
  echo "deleting CSVs in $RAW"
  rm -f "$RAW"/*.csv
fi
echo "done. DSN: $DSN_MSG"
