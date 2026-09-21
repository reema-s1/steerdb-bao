import numpy as np
import pytest
import torch
from synthetic import build_store, synthetic_queries

from steerdb.featurize import NODE_DIM, Featurizer
from steerdb.models import CostModel, load_model, make_model
from steerdb.models.tree_conv import TreeCNN, TreeConv, collate
from steerdb.training import dedupe, xy


def test_tree_conv_gathers_self_and_children():
    conv = TreeConv(2, 1)
    with torch.no_grad():
        conv.linear.weight[:] = torch.tensor([[1.0, 0, 10.0, 0, 100.0, 0]])
        conv.linear.bias.zero_()
    x = torch.tensor([[0.0, 0], [1, 0], [2, 0], [3, 0]])  # row 0 = padding
    idx = torch.tensor([[1, 2, 3], [2, 0, 0], [3, 0, 0]])
    out = conv(x, idx)
    assert out[0].item() == 0  # padding stays zero
    assert out[1:, 0].tolist() == [1 + 20 + 300, 2, 3]


def test_collate_offsets_children(plan_3way, plan_factory):
    f = Featurizer()
    trees = [f.plan_to_tree(plan_3way), f.plan_to_tree(plan_factory())]
    x, idx, pid, n = collate(trees)
    assert n == 2
    assert x.shape == (1 + 5 + 4, NODE_DIM)
    assert idx[:, 0].tolist() == list(range(1, 10))  # self indices are contiguous
    assert idx[5].tolist() == [6, 7, 0]  # second tree's root: Aggregate -> join at 7
    assert pid.tolist() == [0] * 5 + [1] * 4


def test_tree_cnn_output_shape_and_pooling_is_per_plan(plan_3way, plan_factory):
    f = Featurizer()
    net = TreeCNN()
    x, idx, pid, n = collate([f.plan_to_tree(plan_3way), f.plan_to_tree(plan_factory())])
    out = net(x, idx, pid, n)
    assert out.shape == (2,)
    # Each plan's prediction must not depend on the other plans in the batch.
    x1, idx1, pid1, n1 = collate([f.plan_to_tree(plan_3way)])
    assert net(x1, idx1, pid1, n1)[0].item() == pytest.approx(out[0].item(), abs=1e-5)


def test_parameter_count_in_design_range():
    n = sum(p.numel() for p in TreeCNN().parameters())
    assert 100_000 <= n <= 300_000


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("lgbm", {"n_members": 3, "n_estimators": 100}),
        ("treecnn", {"n_members": 2, "epochs": 40}),
    ],
)
def test_models_learn_synthetic_rule_and_roundtrip(kind, kwargs, tmp_path):
    store = build_store(synthetic_queries(12))
    plans, lat = xy(store.observations(("bootstrap",)))
    model = make_model(kind, seed=0, **kwargs)
    model.fit(plans, lat)
    mu, sigma = model.predict(plans)
    assert mu.shape == sigma.shape == (len(plans),)
    assert (sigma >= 0).all()
    corr = np.corrcoef(mu, np.log(lat))[0, 1]
    assert corr > 0.9, corr

    model.save(tmp_path / kind)
    again = load_model(tmp_path / kind)
    np.testing.assert_allclose(again.predict(plans)[0], mu, rtol=1e-4, atol=1e-4)


def test_cost_model_ranks_by_total_cost(plan_factory):
    mu, _ = CostModel().predict([plan_factory(cost=10), plan_factory(cost=1000)])
    assert mu[0] < mu[1]


def test_dedupe_drops_bootstrap_copies():
    store = build_store(synthetic_queries(1, "a"))
    obs = store.observations(("bootstrap",))
    assert len(obs) == 8
    assert len(dedupe(obs)) == 3  # Nested Loop / Hash Join / Merge Join plans
