from synthetic import build_store, synthetic_queries

from steerdb.report import cv_splits, evaluate_folds, full_report, to_markdown
from steerdb.workload import Query

FAST = {"treecnn": {"epochs": 2, "n_members": 1}, "lgbm": {"n_members": 2, "n_estimators": 30}}


def test_cv_evaluates_every_query_once_with_unseen_templates():
    qs = synthetic_queries(12)
    splits = cv_splits(qs, k=4)
    for train, test in splits:
        assert not {q.template for q in train} & {q.template for q in test}
    res = evaluate_folds(
        build_store(qs), splits, kinds=("lgbm",), model_kwargs=FAST, log=lambda *_: None
    )
    assert res["n_test"] == len(qs) and res["n_folds"] == 4
    assert set(res["chosen"]["lgbm"]) == {q.name for q in qs}
    m = res["metrics"]
    assert m["oracle"]["total_ms"] <= m["postgres"]["total_ms"]
    assert m["lgbm"]["total_vs_postgres"] < 1.0  # the synthetic rule is learnable


def test_full_report_with_two_workloads_renders():
    job = synthetic_queries(10)
    ceb = [Query(f"ceb{t}a_0{i}", f"ceb{t}", "SELECT 1") for t in (3, 7) for i in (1, 2)]
    store = build_store(job)
    for obs in build_store(ceb).observations():  # same latency rule, CEB-style names
        store.add(obs)
    r = full_report(
        store,
        job + ceb,
        main_model="lgbm",
        ablations=False,
        online_epochs=0,
        folds=3,
        model_kwargs=FAST,
        log=lambda *_: None,
    )
    assert r["protocol"]["workloads"] == {"ceb": 4, "job": 30}
    assert set(r["oracle_gap"]) == {"all", "ceb", "job"}
    assert set(r["main"]["by_workload"]) == {"ceb", "job"}
    md = to_markdown(r)
    assert "leave-templates-out" in md and "### CEB only" in md
