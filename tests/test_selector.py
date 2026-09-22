import math

import numpy as np
import pytest

from steerdb.selector import Selector


def test_untrained_model_uses_default_arm():
    d = Selector().choose([0, 1, 4], mu=None, trained=False)
    assert (d.arm, d.reason) == (0, "untrained")


def test_greedy_picks_lowest_prediction():
    mu = np.log([100.0, 20.0, 50.0])
    d = Selector(mode="greedy", min_gain=0.05).choose([0, 1, 4], mu)
    assert (d.index, d.arm, d.reason) == (1, 1, "model")
    assert d.predicted_ms == pytest.approx([100.0, 20.0, 50.0])


def test_guard_keeps_default_when_gain_is_too_small():
    mu = np.log([100.0, 97.0])  # predicted 3% faster, below the 5% bar
    d = Selector(mode="greedy", min_gain=0.05).choose([0, 2], mu)
    assert (d.arm, d.reason) == (0, "guard")


def test_default_need_not_be_first_candidate():
    mu = np.log([30.0, 100.0])
    d = Selector(min_gain=0.5).choose([3, 0], mu)  # 3.3x predicted speedup
    assert d.arm == 3
    d = Selector(min_gain=0.5).choose([3, 0], np.log([90.0, 100.0]))
    assert (d.arm, d.reason) == (0, "guard")


def test_thompson_exploits_when_confident():
    mu = np.log([100.0, 10.0])
    sel = Selector(mode="thompson", seed=0)
    picks = [sel.choose([0, 1], mu, np.array([0.01, 0.01])).arm for _ in range(50)]
    assert picks.count(1) == 50


def test_thompson_explores_under_uncertainty():
    mu = np.log([100.0, 110.0])
    sel = Selector(mode="thompson", seed=0)
    decisions = [sel.choose([0, 1], mu, np.array([1.0, 1.0])) for _ in range(400)]
    frac = sum(d.arm == 1 for d in decisions) / len(decisions)
    assert 0.3 < frac < 0.6
    assert any(d.reason == "explore" for d in decisions)


def test_timeout_is_factor_of_baseline_with_floor():
    sel = Selector(timeout_factor=2.0, timeout_floor_ms=1000)
    assert sel.timeout_ms(5000) == 10000
    assert sel.timeout_ms(100) == 1000
    assert sel.timeout_ms(None) is None


def test_predicted_ms_roundtrip():
    d = Selector().choose([0, 1], np.array([math.log(7.0), math.log(3.0)]))
    assert d.predicted_ms == [pytest.approx(7.0), pytest.approx(3.0)]


def test_uncertainty_guard_needs_confident_gain():
    mu = np.log([100.0, 50.0])  # predicted 2x faster ...
    confident = Selector(min_gain=0.05, guard_k=1.0).choose([0, 1], mu, np.array([0.05, 0.05]))
    assert confident.arm == 1
    # ... but the ensemble disagrees wildly: log(2) - 1.0 * sqrt(2) * 0.8 < log(1.05)
    unsure = Selector(min_gain=0.05, guard_k=1.0).choose([0, 1], mu, np.array([0.8, 0.8]))
    assert (unsure.arm, unsure.reason) == (0, "guard")
    # guard_k = 0 recovers the plain point-estimate rule
    assert Selector(min_gain=0.05, guard_k=0.0).choose([0, 1], mu, np.array([0.8, 0.8])).arm == 1


def test_guard_picks_best_pessimistic_margin_not_best_mean():
    mu = np.log([100.0, 40.0, 45.0])
    sd = np.array([0.0, 1.0, 0.05])  # arm 1 has the best mean but is very uncertain
    assert Selector(min_gain=0.05, guard_k=1.0).choose([0, 1, 4], mu, sd).arm == 4


def test_default_timeout_has_no_large_floor():
    assert Selector().timeout_ms(100) == 200
    assert Selector().timeout_ms(0.1) == 1  # never 0: statement_timeout = 0 means "no limit"
