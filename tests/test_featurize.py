import math

import numpy as np
import pytest

from steerdb.featurize import (
    FLAG_OFFSET,
    FLAT_NAMES,
    NODE_DIM,
    NUMERIC_OFFSET,
    TABLE_OFFSET,
    Featurizer,
    Stats,
    flat_features,
    node_type_index,
)


def test_worked_example_from_design_doc(plan_3way):
    """Design doc section 6.3: Hash Join, rows=42000, cost=18500.3, width=64."""
    f = Featurizer(Stats(rows=(8.2, 3.1), cost=(9.1, 2.7), width=(48, 30)), fitted=True)
    hash_join = plan_3way["Plans"][0]
    v = f.node_vector(hash_join)

    assert v.shape == (NODE_DIM,)
    assert node_type_index("Hash Join") == 3
    expected_onehot = np.zeros(NUMERIC_OFFSET)
    expected_onehot[3] = 1
    np.testing.assert_array_equal(v[:NUMERIC_OFFSET], expected_onehot)
    assert v[NUMERIC_OFFSET + 0] == pytest.approx(0.79, abs=0.01)
    assert v[NUMERIC_OFFSET + 1] == pytest.approx(0.27, abs=0.01)
    assert v[NUMERIC_OFFSET + 2] == pytest.approx(0.53, abs=0.01)
    assert v[TABLE_OFFSET:FLAG_OFFSET].sum() == 0  # joins have no relation
    assert v[FLAG_OFFSET] == 1  # Hash Cond counts as a filter
    assert v[FLAG_OFFSET + 1] == 0


def test_scan_node_relation_and_index_flag(plan_3way):
    f = Featurizer()
    index_scan = plan_3way["Plans"][0]["Plans"][1]["Plans"][0]
    v = f.node_vector(index_scan)
    from steerdb.featurize import IMDB_TABLES

    assert v[TABLE_OFFSET + IMDB_TABLES.index("title")] == 1
    assert v[TABLE_OFFSET:FLAG_OFFSET].sum() == 1
    assert v[FLAG_OFFSET + 1] == 1


def test_unknown_node_type_maps_to_other():
    assert node_type_index("Custom Scan") == node_type_index("Other")


def test_plan_to_tree_structure(plan_3way):
    """Aggregate -> HashJoin -> (SeqScan, Hash -> IndexScan), preorder numbering from 1."""
    t = Featurizer().plan_to_tree(plan_3way)
    assert t.x.shape == (6, NODE_DIM)  # 5 nodes + padding row
    np.testing.assert_array_equal(t.x[0], 0)
    np.testing.assert_array_equal(
        t.idx,
        [
            [1, 2, 0],  # Aggregate: left = Hash Join
            [2, 3, 4],  # Hash Join: Seq Scan, Hash
            [3, 0, 0],  # Seq Scan
            [4, 5, 0],  # Hash: Index Scan
            [5, 0, 0],
        ],
    )
    assert t.x[1, node_type_index("Aggregate")] == 1
    assert t.x[3, node_type_index("Seq Scan")] == 1


def test_nary_nodes_are_binarized():
    leaf = {"Node Type": "Seq Scan", "Relation Name": "title"}
    plan = {"Node Type": "Append", "Plans": [leaf, leaf, leaf]}
    t = Featurizer().plan_to_tree(plan)
    # Append, leaf, Other(leaf, leaf), leaf, leaf
    assert t.idx.shape == (5, 3)
    assert t.idx[0].tolist() == [1, 2, 3]
    assert t.x[3, node_type_index("Other")] == 1
    assert t.idx[2].tolist() == [3, 4, 5]


def test_fit_standardizes(plan_3way, plan_factory):
    plans = [plan_3way, plan_factory(rows=10), plan_factory(rows=1e6)]
    f = Featurizer().fit(plans)
    rows = np.array([f.node_vector(n)[NUMERIC_OFFSET] for p in plans for n in _nodes(p)])
    assert rows.mean() == pytest.approx(0, abs=1e-5)
    assert rows.std() == pytest.approx(1, abs=1e-5)
    g = Featurizer.from_dict(f.to_dict())
    np.testing.assert_allclose(g.node_vector(plan_3way), f.node_vector(plan_3way))


def test_flat_features(plan_3way):
    x = flat_features(plan_3way)
    feats = dict(zip(FLAT_NAMES, x))
    assert len(x) == len(FLAT_NAMES)
    assert feats["count[Hash Join]"] == 1
    assert feats["n_nodes"] == 5
    assert feats["depth"] == 4
    assert feats["n_joins"] == 1
    assert feats["root_log_cost"] == pytest.approx(math.log1p(18600.12), rel=1e-5)
    assert feats["max_join_log_rows"] == pytest.approx(math.log1p(42000), rel=1e-5)


def _nodes(p):
    yield p
    for c in p.get("Plans", []):
        yield from _nodes(c)
