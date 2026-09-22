"""Phase 2 main model: Tree Convolutional Network (Neo / Bao style) over plan trees.

    3 x tree-conv (node, left child, right child; weights shared over the tree)
    -> dynamic max-pool over all nodes -> 2 FC layers -> predicted log-latency
An ensemble of independently seeded members (each on a bootstrap resample) gives uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..featurize import NODE_DIM, Featurizer, TreeTensor
from . import ValueModel, to_log


class TreeConv(nn.Module):
    """out(n) = W [x(n) ; x(left(n)) ; x(right(n))] + b, with x(0) = 0 as the padding node."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * in_dim, out_dim)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        gathered = x[idx].reshape(idx.shape[0], -1)  # (N, 3*in)
        out = self.linear(gathered)
        # Re-attach the zero padding row so the next layer can index children the same way.
        return torch.cat([out.new_zeros(1, out.shape[1]), out], dim=0)


class TreeCNN(nn.Module):
    def __init__(self, in_dim: int = NODE_DIM, channels=(256, 128, 64), hidden: int = 32) -> None:
        super().__init__()
        dims = (in_dim, *channels)
        self.convs = nn.ModuleList(TreeConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.act = nn.LeakyReLU()
        self.head = nn.Sequential(
            nn.Linear(channels[-1], hidden), nn.LeakyReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x, idx, plan_ids, n_plans: int) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, idx)
            h = torch.cat([h[:1], self.act(h[1:])], dim=0)  # keep padding row exactly zero
        nodes = h[1:]  # (N, C); row k is node k+1 == idx[k, 0]
        pooled = torch.full((n_plans, nodes.shape[1]), float("-inf"), device=nodes.device)
        pooled = pooled.scatter_reduce(
            0, plan_ids[:, None].expand(-1, nodes.shape[1]), nodes, reduce="amax"
        )
        return self.head(pooled).squeeze(-1)


def collate(trees: list[TreeTensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack many trees into one big forest sharing a single zero padding row."""
    xs = [np.zeros((1, trees[0].x.shape[1]), dtype=np.float32)]
    idxs, plan_ids = [], []
    offset = 0
    for i, t in enumerate(trees):
        n = t.idx.shape[0]
        xs.append(t.x[1:])
        shifted = t.idx.copy()
        shifted[shifted > 0] += offset
        idxs.append(shifted)
        plan_ids.append(np.full(n, i, dtype=np.int64))
        offset += n
    return (
        torch.from_numpy(np.concatenate(xs)),
        torch.from_numpy(np.concatenate(idxs)),
        torch.from_numpy(np.concatenate(plan_ids)),
        len(trees),
    )


def resolve_device(device: str = "auto") -> torch.device:
    """'auto' -> CUDA if available, else CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no GPU is available")
    return torch.device(device)


class TreeCNNModel(ValueModel):
    kind = "treecnn"

    def __init__(
        self,
        n_members: int = 3,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.device = resolve_device(device)
        self.n_members = n_members
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.featurizer = Featurizer()
        self.members: list[TreeCNN] = []
        self.y_mean, self.y_std = 0.0, 1.0

    def n_parameters(self) -> int:
        return sum(p.numel() for p in TreeCNN().parameters())

    def fit(self, plans, latencies_ms) -> None:
        if self.device.type != "cpu":
            self._fit(plans, latencies_ms)
            return
        # Tiny tensors: many threads cost more in synchronization than they save.
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, prev_threads))
        try:
            self._fit(plans, latencies_ms)
        finally:
            torch.set_num_threads(prev_threads)

    def _fit(self, plans, latencies_ms) -> None:
        self.featurizer = Featurizer().fit(plans)
        trees = [self.featurizer.plan_to_tree(p) for p in plans]
        y = to_log(latencies_ms)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)
        y_norm = torch.tensor(
            (y - self.y_mean) / self.y_std, dtype=torch.float32, device=self.device
        )

        self.members = []
        for m in range(self.n_members):
            torch.manual_seed(self.seed + m)
            rng = np.random.default_rng(self.seed + m)
            sample = (
                rng.integers(0, len(trees), len(trees))
                if self.n_members > 1
                else np.arange(len(trees))
            )
            net = TreeCNN().to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            net.train()
            for _ in range(self.epochs):
                order = rng.permutation(sample)
                for start in range(0, len(order), self.batch_size):
                    batch = order[start : start + self.batch_size]
                    x, idx, pid, n = self._to_device(collate([trees[i] for i in batch]))
                    loss = nn.functional.mse_loss(net(x, idx, pid, n), y_norm[batch])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            self.members.append(net)
        self.trained = True

    @torch.no_grad()
    def predict(self, plans):
        trees = [self.featurizer.plan_to_tree(p) for p in plans]
        x, idx, pid, n = self._to_device(collate(trees))
        preds = torch.stack([net(x, idx, pid, n) for net in self.members]).cpu().numpy()
        preds = preds * self.y_std + self.y_mean
        return preds.mean(axis=0), preds.std(axis=0)

    def _to_device(self, batch):
        x, idx, pid, n = batch
        return x.to(self.device), idx.to(self.device), pid.to(self.device), n

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Always save CPU tensors, so a model trained on a GPU (Colab/Kaggle) loads anywhere.
        states = [{k: v.cpu() for k, v in net.state_dict().items()} for net in self.members]
        torch.save(states, directory / "members.pt")
        meta = {
            "kind": self.kind,
            "n_members": self.n_members,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "featurizer": self.featurizer.to_dict(),
        }
        (directory / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: Path, device: str = "auto") -> TreeCNNModel:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        m = cls(
            meta["n_members"],
            meta["epochs"],
            meta["batch_size"],
            meta["lr"],
            meta["weight_decay"],
            meta["seed"],
            device,
        )
        m.y_mean, m.y_std = meta["y_mean"], meta["y_std"]
        m.featurizer = Featurizer.from_dict(meta["featurizer"])
        m.members = []
        for sd in torch.load(directory / "members.pt", map_location="cpu", weights_only=True):
            net = TreeCNN()
            net.load_state_dict(sd)
            net.to(m.device)
            net.eval()
            m.members.append(net)
        m.trained = True
        return m
