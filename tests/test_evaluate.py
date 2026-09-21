from synthetic import build_store, latency, query_rows, synthetic_queries

from steerdb.evaluate import (
    candidates_from_table,
    metrics,
    model_policy,
    oracle_gap,
    oracle_policy,
    random_policy,
    run_policy,
    stock_policy,
)
from steerdb.models import make_model
from steerdb.online import online_loop
from steerdb.selector import Selector
from steerdb.training import train_model
from steerdb.workload import split_by_template

ARMS = tuple(range(8))


def _setup():
    queries = synthetic_queries()
    store = build_store(queries)
    train, test = split_by_template(queries)
    return queries, store, store.bootstrap_table(), train, test


def test_candidates_are_deduplicated():
    _, _, table, _, _ = _setup()
    cands = candidates_from_table(table, "1a", ARMS)
    assert len(cands) == 3
    assert cands[0].arm == 0 and cands[0].arms == [0, 3, 5, 6]


def test_stock_and_oracle():
    _, _, table, _, test = _setup()
    names = [q.name for q in test]
    stock = stock_policy(table, names)
    oracle = oracle_policy(table, names, ARMS)
    for q in test:
        assert stock.latency[q.name] == latency("Nested Loop", query_rows(q))
        assert oracle.latency[q.name] <= stock.latency[q.name]
    m = metrics(oracle, stock, oracle)
    assert m["mean_regret"] == 0
    assert m["n_regressions"] == 0
    assert m["total_vs_postgres"] < 1


def test_safety_timeout_is_simulated():
    _, _, table, _, _ = _setup()
    # On the smallest query the nested loop (arm 0) is fast and the merge join (arm 2) is far
    # slower than 2x, so forcing arm 2 must hit the timeout and pay timeout + arm 0.
    q = min(table, key=lambda name: table[name][0].latency_ms)
    base = table[q][0].latency_ms
    assert table[q][2].latency_ms > 2 * base
    sel = Selector(timeout_factor=2.0, timeout_floor_ms=1.0)
    res = run_policy("forced", table, [q], lambda _: 2, sel)
    assert res.fallbacks == 1
    assert res.latency[q] == sel.timeout_ms(base) + base
    # Arm 0 itself is never "timed out" by the guard.
    assert run_policy("stock", table, [q], lambda _: 0, sel).fallbacks == 0


def test_oracle_gap_reports_headroom():
    queries, _, table, _, _ = _setup()
    gap = oracle_gap(table, [q.name for q in queries])
    assert gap["n_queries"] == len(queries)
    assert gap["oracle_improvement"] > 0.15
    assert sum(gap["oracle_best_arm_counts"].values()) == len(queries)


def test_trained_model_beats_postgres_on_unseen_templates():
    _, store, table, train, test = _setup()
    model = train_model("lgbm", store, {q.name for q in train}, n_members=3, n_estimators=100)
    names = [q.name for q in test]
    stock, oracle = stock_policy(table, names), oracle_policy(table, names, ARMS)
    m = metrics(model_policy("lgbm", model, table, names, ARMS), stock, oracle)
    rnd = metrics(random_policy(table, names, ARMS), stock, oracle)
    assert m["total_vs_postgres"] < 0.8
    assert m["total_ms"] < rnd["total_ms"]


def test_untrained_model_policy_is_stock():
    _, _, table, _, test = _setup()
    names = [q.name for q in test]
    res = model_policy("untrained", make_model("lgbm"), table, names, ARMS)
    assert set(res.chosen.values()) == {0}


def test_online_replay_improves_over_postgres():
    _, _, table, train, test = _setup()
    res = online_loop(
        table,
        train,
        test,
        model_kind="lgbm",
        epochs=3,
        retrain_every=20,
        model_kwargs={"n_members": 3, "n_estimators": 60},
        log=lambda *_: None,
    )
    assert len(res.epochs) == 3
    assert res.epochs[-1].train_workload_ms < res.postgres_train_ms
    assert res.epochs[-1].test["total_vs_postgres"] < 1.0
