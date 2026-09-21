"""Value models: predict log-latency (and its uncertainty) for a plan.

All models share one interface:
    fit(plans, latencies_ms)              -> None
    predict(plans) -> (mu, sigma)         both in log(ms) space, shape (n,)
    save(dir) / load_model(dir)
Uncertainty comes from an ensemble: sigma is the std of member predictions.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

MIN_LATENCY_MS = 0.01


def to_log(latencies_ms) -> np.ndarray:
    return np.log(np.maximum(np.asarray(latencies_ms, dtype=np.float64), MIN_LATENCY_MS))


class ValueModel(ABC):
    kind: str = "base"

    def __init__(self) -> None:
        self.trained = False

    @abstractmethod
    def fit(self, plans: list[dict], latencies_ms: list[float]) -> None: ...

    @abstractmethod
    def predict(self, plans: list[dict]) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    def predict_ms(self, plans: list[dict]) -> np.ndarray:
        mu, _ = self.predict(plans)
        return np.exp(mu)


class CostModel(ValueModel):
    """No learning: rank plans by the optimizer's own Total Cost ("cost-only re-ranking")."""

    kind = "cost"

    def __init__(self) -> None:
        super().__init__()
        self.trained = True

    def fit(self, plans, latencies_ms) -> None:
        pass

    def predict(self, plans):
        mu = np.array([math.log1p(float(p.get("Total Cost", 0.0))) for p in plans])
        return mu, np.zeros_like(mu)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meta.json").write_text(json.dumps({"kind": self.kind}))


def make_model(kind: str, **kwargs) -> ValueModel:
    if kind == "lgbm":
        from .lgbm_baseline import LGBMModel

        return LGBMModel(**kwargs)
    if kind == "treecnn":
        from .tree_conv import TreeCNNModel

        return TreeCNNModel(**kwargs)
    if kind == "cost":
        return CostModel()
    raise ValueError(f"unknown model kind {kind!r} (expected lgbm | treecnn | cost)")


def load_model(directory: Path) -> ValueModel:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    kind = meta["kind"]
    if kind == "lgbm":
        from .lgbm_baseline import LGBMModel

        return LGBMModel.load(directory)
    if kind == "treecnn":
        from .tree_conv import TreeCNNModel

        return TreeCNNModel.load(directory)
    if kind == "cost":
        return CostModel()
    raise ValueError(f"unknown model kind {kind!r} in {directory}")
