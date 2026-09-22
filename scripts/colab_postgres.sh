#!/usr/bin/env bash
# Install and start PostgreSQL 16 on a Debian/Ubuntu VM without Docker (Google Colab, Kaggle),
# configured like docker-compose.yml. Run as root (the default on Colab/Kaggle). Idempotent.
#
#   bash scripts/colab_postgres.sh
#   export STEERDB_DSN=postgresql://postgres:steerdb@localhost:5432/imdb
set -euo pipefail

PGVER=16
export DEBIAN_FRONTEND=noninteractive

if [ ! -x "/usr/lib/postgresql/$PGVER/bin/postgres" ]; then
  echo "installing PostgreSQL $PGVER from apt.postgresql.org"
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates lsb-release >/dev/null
  install -d /usr/share/postgresql-common/pgdg
  curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    https://www.postgresql.org/media/keys/ACCC4CF8.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc]" \
    "https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list
  apt-get update -qq
  apt-get install -y -qq "postgresql-$PGVER" >/dev/null
fi

service postgresql start >/dev/null

as_pg() { su postgres -c "$1"; }
as_pg "psql -q -c \"ALTER USER postgres PASSWORD 'steerdb'\""
if ! as_pg "psql -tAc \"SELECT 1 FROM pg_database WHERE datname = 'imdb'\"" | grep -q 1; then
  as_pg "createdb imdb"
fi

# Same server settings as docker-compose.yml.
as_pg "psql -q" <<'SQL'
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET effective_cache_size = '4GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET max_parallel_workers_per_gather = 0;
ALTER SYSTEM SET jit = off;
ALTER SYSTEM SET random_page_cost = 1.1;
SQL
service postgresql restart >/dev/null

psql "postgresql://postgres:steerdb@localhost:5432/imdb" -tAc "SELECT version()"
echo "ready: postgresql://postgres:steerdb@localhost:5432/imdb"
