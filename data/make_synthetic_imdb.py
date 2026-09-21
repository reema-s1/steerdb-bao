"""Generate a small synthetic database with the IMDB/JOB schema, for smoke tests.

Real results need the real IMDB data (data/load_imdb.sh). This exists so the whole pipeline
(collect -> train -> run -> bench) can be exercised in minutes on any machine: same 21 tables,
same foreign-key indexes, the real dimension-table values JOB predicates look for, and skewed
foreign keys so the planner's estimates are wrong in realistic ways.

    python data/make_synthetic_imdb.py --dsn postgresql://postgres@localhost:5499/imdb --scale 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from steerdb import config  # noqa: E402

# Rows per table at scale 1.
SIZES = {
    "title": 60_000,
    "name": 50_000,
    "char_name": 30_000,
    "company_name": 10_000,
    "keyword": 8_000,
    "aka_name": 20_000,
    "aka_title": 10_000,
    "cast_info": 400_000,
    "movie_info": 200_000,
    "movie_info_idx": 60_000,
    "movie_keyword": 160_000,
    "movie_companies": 90_000,
    "person_info": 80_000,
    "complete_cast": 12_000,
    "movie_link": 6_000,
}

# Dimension tables: (value column, real values). Their size is len(values).
DIMS = {
    "kind_type": (
        "kind",
        [
            "movie",
            "tv series",
            "tv movie",
            "video movie",
            "tv mini series",
            "video game",
            "episode",
        ],
    ),
    "company_type": (
        "kind",
        [
            "distributors",
            "production companies",
            "special effects companies",
            "miscellaneous companies",
        ],
    ),
    "comp_cast_type": ("kind", ["cast", "crew", "complete", "complete+verified"]),
    "role_type": (
        "role",
        [
            "actor",
            "actress",
            "producer",
            "writer",
            "cinematographer",
            "composer",
            "costume designer",
            "director",
            "editor",
            "miscellaneous crew",
            "production designer",
            "guest",
        ],
    ),
    "link_type": (
        "link",
        [
            "follows",
            "followed by",
            "remake of",
            "remade as",
            "references",
            "referenced in",
            "spoofs",
            "spoofed in",
            "features",
            "featured in",
            "spin off from",
            "spin off",
            "version of",
            "similar to",
            "edited into",
            "edited from",
            "alternate language version of",
            "unknown link",
        ],
    ),
    "info_type": (
        "info",
        [
            "runtimes",
            "color info",
            "genres",
            "languages",
            "certificates",
            "sound mix",
            "tech info",
            "countries",
            "taglines",
            "keywords",
            "alternate versions",
            "crazy credits",
            "goofs",
            "soundtrack",
            "quotes",
            "release dates",
            "trivia",
            "locations",
            "mini biography",
            "birth notes",
            "birth date",
            "height",
            "death date",
            "spouse",
            "other works",
            "birth name",
            "salary history",
            "nick names",
            "books",
            "agent address",
            "biographical movies",
            "portrayed in",
            "where now",
            "trade mark",
            "interviews",
            "article",
            "magazine cover photo",
            "pictorial",
            "death notes",
            "LD disc format",
            "plot",
            "votes distribution",
            "rating",
            "votes",
            "budget",
            "gross",
            "opening weekend",
            "rentals",
            "admissions",
            "filming dates",
            "production dates",
            "studios",
            "top 250 rank",
            "bottom 10 rank",
        ],
    ),
}

PARENT = {
    "movie_id": "title",
    "linked_movie_id": "title",
    "episode_of_id": "title",
    "person_id": "name",
    "person_role_id": "char_name",
    "company_id": "company_name",
    "keyword_id": "keyword",
    "role_id": "role_type",
    "kind_id": "kind_type",
    "company_type_id": "company_type",
    "info_type_id": "info_type",
    "link_type_id": "link_type",
    "subject_id": "comp_cast_type",
    "status_id": "comp_cast_type",
}

VOCAB = [
    "Drama",
    "Horror",
    "Action",
    "Sci-Fi",
    "Thriller",
    "Comedy",
    "Documentary",
    "USA",
    "Germany",
    "Sweden",
    "Japan",
    "France",
    "English",
    "German",
    "(co-production)",
    "(presents)",
    "(as Metro-Goldwyn-Mayer Pictures)",
    "(voice)",
    "(uncredited)",
    "(producer)",
    "(writer)",
    "[us]",
    "[de]",
    "[gb]",
    "[jp]",
    "character-name-in-title",
    "sequel",
    "superhero",
    "marvel-cinematic-universe",
    "murder",
    "violence",
    "blood",
    "based-on-novel",
    "based-on-comic",
    "Warner Bros",
    "Tim",
    "Bert",
    "Queen",
    "Iron Man",
    "Avatar",
    "Shrek",
    "Dark",
    "Champion",
    "Money",
    "Fight",
    "Love",
    "Kung Fu Panda",
    "Batman",
    "Joker",
    "Angela",
    "Robert",
    "Anne",
    "Internet:",
    "USA:2000",
    "Japan:2005",
    "PCS:Spherical",
    "OFM:35 mm",
    "top",
    "rank",
    "7.0",
    "8.5",
    "1000",
    "$ 1,000,000",
    "Women",
    "Man",
    "The",
    "A",
    "Of",
    "Episode",
    "Season",
    "Film",
]


def _sql_array(values: list[str]) -> str:
    return "ARRAY[" + ",".join("'" + v.replace("'", "''") + "'" for v in values) + "]"


def column_expr(table: str, col: str, dtype: str, maxlen: int | None, nullable: bool, sizes) -> str:
    if col == "id":
        return "g"
    if table in DIMS and col == DIMS[table][0]:
        return f"({_sql_array(DIMS[table][1])})[g]"
    if col in PARENT:
        # Skewed: low ids are much more popular, which breaks the uniformity assumption.
        return f"1 + floor(power(random(), 2.5) * {sizes[PARENT[col]]})::int"
    if col == "production_year":
        expr = "1900 + floor(power(random(), 0.4) * 116)::int"
    elif col == "md5sum":
        expr = "md5(random()::text)"
    elif dtype == "integer":
        expr = "floor(random() * 100)::int"
    else:
        vocab = _sql_array(VOCAB)
        n = len(VOCAB)
        pick = f"({vocab})[1 + floor(random() * {n})::int]"
        expr = f"{pick} || ' ' || {pick}"
        if maxlen:
            expr = f"left({expr}, {maxlen})"
    if nullable:
        expr = f"CASE WHEN random() < 0.3 THEN NULL ELSE {expr} END"
    return expr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dsn", default=config.DSN)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--schema", default=str(ROOT / "data" / "raw" / "schema.sql"))
    ap.add_argument("--fkindexes", default=str(ROOT / "data" / "raw" / "fkindexes.sql"))
    args = ap.parse_args()

    sizes = {t: max(1, int(n * args.scale)) for t, n in SIZES.items()}
    sizes.update({t: len(v[1]) for t, v in DIMS.items()})

    t0 = time.perf_counter()
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
        conn.execute(Path(args.schema).read_text())
        cols = conn.execute(
            "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable"
            " FROM information_schema.columns WHERE table_schema = 'public'"
            " ORDER BY table_name, ordinal_position"
        ).fetchall()
        by_table: dict[str, list] = {}
        for t, c, dt, ml, nl in cols:
            by_table.setdefault(t, []).append((c, dt, ml, nl == "YES"))
        for table, columns in by_table.items():
            names = ", ".join(c for c, *_ in columns)
            exprs = ", ".join(column_expr(table, c, dt, ml, nl, sizes) for c, dt, ml, nl in columns)
            conn.execute(
                f"INSERT INTO {table} ({names}) SELECT {exprs}"
                f" FROM generate_series(1, {sizes[table]}) AS g"
            )
            print(f"  {table:<16} {sizes[table]:>8} rows")
        conn.execute(Path(args.fkindexes).read_text())
        conn.execute("VACUUM ANALYZE")
    print(f"synthetic IMDB ready in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
