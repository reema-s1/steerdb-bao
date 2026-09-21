import pytest
from synthetic import synthetic_queries

from steerdb.workload import (
    load_workload,
    parse_name,
    split_by_template,
    split_random,
    subsample_templates,
)


def test_parse_name():
    assert parse_name("16b") == (16, "b")
    assert parse_name("1a") == (1, "a")
    with pytest.raises(ValueError):
        parse_name("schema")


def test_template_split_never_shares_templates():
    train, test = split_by_template(synthetic_queries())
    assert {q.template for q in test} == set(range(26, 34))
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
