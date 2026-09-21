"""Phase 1 baseline: LightGBM ensemble on a flattened plan summary."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ..featurize import flat_features
from . import ValueModel, to_log


class LGBMModel(ValueModel):
    kind = "lgbm"

    def __init__(self, n_members: int = 5, n_estimators: int = 300, seed: int = 0) -> None:
        super().__init__()
        self.n_members = n_members
        self.n_estimators = n_estimators
        self.seed = seed
        self.boosters: list[lgb.Booster] = []

    def _params(self, member: int) -> dict:
        return {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "seed": self.seed + member,
            "verbosity": -1,
            "num_threads": 0,
        }

    def fit(self, plans, latencies_ms) -> None:
        X = np.stack([flat_features(p) for p in plans])
        y = to_log(latencies_ms)
        self.boosters = [
            lgb.train(self._params(m), lgb.Dataset(X, y), num_boost_round=self.n_estimators)
            for m in range(self.n_members)
        ]
        self.trained = True

    def predict(self, plans):
        X = np.stack([flat_features(p) for p in plans])
        preds = np.stack([b.predict(X) for b in self.boosters])
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(self.boosters):
            b.save_model(str(directory / f"member{i}.txt"))
        meta = {
            "kind": self.kind,
            "n_members": self.n_members,
            "n_estimators": self.n_estimators,
            "seed": self.seed,
        }
        (directory / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: Path) -> LGBMModel:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        m = cls(meta["n_members"], meta["n_estimators"], meta["seed"])
        m.boosters = [
            lgb.Booster(model_file=str(directory / f"member{i}.txt")) for i in range(m.n_members)
        ]
        m.trained = True
        return m
