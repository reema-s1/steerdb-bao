import pytest

from steerdb.arms import ARM_SUBSETS, ARMS, PLANNER_GUCS, get_arm, parse_arm_ids
from steerdb.plan_gen import plan_hash


def test_eight_arms_with_default_first():
    assert len(ARMS) == 8
    assert [a.id for a in ARMS] == list(range(8))
    assert ARMS[0].settings == {}


def test_arm_settings_only_touch_known_gucs():
    for arm in ARMS:
        assert set(arm.settings) <= set(PLANNER_GUCS)
        assert all(v == "off" for v in arm.settings.values())


def test_arms_are_distinct():
    keys = {tuple(sorted(a.settings.items())) for a in ARMS}
    assert len(keys) == len(ARMS)


def test_set_statements_reset_every_other_guc():
    stmts = get_arm(4).set_statements()
    assert len(stmts) == len(PLANNER_GUCS)
    assert "SET enable_nestloop = off" in stmts
    assert "SET enable_mergejoin = off" in stmts
    assert "SET enable_hashjoin = on" in stmts
    assert all(s.endswith("= on") for s in get_arm(0).set_statements())


def test_design_doc_table():
    assert get_arm(1).settings == {"enable_nestloop": "off"}
    assert get_arm(5).settings == {"enable_hashjoin": "off", "enable_mergejoin": "off"}
    assert get_arm(7).settings == {"enable_indexscan": "off", "enable_bitmapscan": "off"}


def test_parse_arm_ids_always_includes_default():
    assert parse_arm_ids("3,1") == (0, 1, 3)
    assert parse_arm_ids(None) == tuple(range(8))
    with pytest.raises(KeyError):
        parse_arm_ids("9")


def test_subsets_contain_default():
    for k, subset in ARM_SUBSETS.items():
        assert len(subset) == k and 0 in subset


def test_plan_hash_ignores_costs_but_not_structure(plan_factory):
    a = plan_factory(join="Hash Join", cost=100)
    b = plan_factory(join="Hash Join", cost=1e10 + 100)  # disabled-method penalty
    c = plan_factory(join="Nested Loop", cost=100)
    assert plan_hash(a) == plan_hash(b)
    assert plan_hash(a) != plan_hash(c)
