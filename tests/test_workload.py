import pytest
from synthetic import synthetic_queries

from steerdb.workload import (
    load_workload,
    split_by_template,
    split_random,
    subsample_templates,
    template_folds,
    template_of,
)


def test_template_of():
    assert template_of("16b") == "16"
    assert template_of("1a") == "1"
    assert template_of("ceb9a_07") == "ceb9"
    assert template_of("ceb9b_12") == "ceb9"  # same join graph -> same group
    assert template_of("ceb11b_01") == "ceb11"
    with pytest.raises(ValueError):
        template_of("schema")


def test_template_split_never_shares_templates():
    train, test = split_by_template(synthetic_queries())
    assert {q.template for q in test} == {str(t) for t in range(26, 34)}
    assert not {q.template for q in train} & {q.template for q in test}


def test_random_split_leaks_templates():
    """The reason we split by template: a random split shares templates across sides."""
    train, test = split_random(synthetic_queries(), test_frac=0.25, seed=0)
    assert {q.template for q in train} & {q.template for q in test}
    assert len(train) + len(test) == 99


def test_subsample_keeps_whole_templates():
    qs = synthetic_queries(20)
    sub = subsample_templates(qs, 0.25, seed=1)
    assert len({q.template for q in sub}) == 5
    assert len(sub) == 15


def test_load_workload(tmp_path):
    (tmp_path / "2b.sql").write_text("SELECT 2;\n")
    (tmp_path / "10a.sql").write_text("SELECT 10")
    (tmp_path / "schema.sql").write_text("CREATE TABLE x ();")
    qs = load_workload(tmp_path)
    assert [q.name for q in qs] == ["2b", "10a"]
    assert qs[0].sql == "SELECT 2"


def test_load_workload_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_workload(tmp_path)


def test_load_multiple_dirs_and_natural_order(tmp_path):
    job, ceb = tmp_path / "job", tmp_path / "ceb"
    job.mkdir()
    ceb.mkdir()
    for n in ("10a", "2b"):
        (job / f"{n}.sql").write_text("SELECT 1")
    (ceb / "ceb3b_01.sql").write_text("SELECT 3")
    qs = load_workload(f"{job},{ceb}")
    assert [q.name for q in qs] == ["2b", "10a", "ceb3b_01"]
    assert [q.template for q in qs] == ["2", "10", "ceb3"]
    assert [q.name for q in load_workload([job, ceb])] == [q.name for q in qs]


def test_template_folds_partition_templates():
    qs = synthetic_queries(33)
    folds = template_folds(qs, k=5, seed=0)
    flat = [t for f in folds for t in f]
    assert sorted(flat) == sorted({q.template for q in qs})  # every template exactly once
    assert len(folds) == 5 and all(len(f) in (6, 7) for f in folds)
    # shuffled: the high-numbered (large-join) templates are not all in one fold
    assert max(sum(int(t) >= 26 for t in f) for f in folds) < 8
