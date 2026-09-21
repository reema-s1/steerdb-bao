"""Figures for the write-up, generated from results.json."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POLICIES = ("postgres", "random", "cost-only", "lgbm", "treecnn", "oracle")
COLORS = {
    "postgres": "#6b7280",
    "random": "#d1d5db",
    "cost-only": "#a8a29e",
    "lgbm": "#60a5fa",
    "treecnn": "#2563eb",
    "oracle": "#16a34a",
}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def total_latency_bar(report: dict, path: Path) -> Path:
    ms = report["main"]["metrics"]
    names = [p for p in POLICIES if p in ms]
    totals = [ms[p]["total_ms"] / 1000 for p in names]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(names, totals, color=[COLORS[p] for p in names])
    for b, p in zip(bars, names):
        ax.annotate(
            f"{ms[p]['total_vs_postgres']:.2f}x",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("total workload latency (s)")
    ax.set_title("Unseen templates: total latency (label = vs. Postgres)")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def speedup_box(report: dict, path: Path) -> Path:
    ms = report["main"]["metrics"]
    names = [p for p in POLICIES if p in ms and p != "postgres"]
    data = [list(ms[p]["speedups"].values()) for p in names]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bp = ax.boxplot(data, patch_artist=True, whis=(5, 95))
    ax.set_xticks(range(1, len(names) + 1), names)
    for patch, p in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[p])
    ax.axhline(1.0, color="#111827", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("per-query speedup vs. Postgres (log)")
    ax.set_title("Distribution of wins and losses (>1 = faster than Postgres)")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def learning_curve(report: dict, path: Path) -> Path | None:
    online = report.get("online")
    if not online:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for mode, o in online.items():
        xs = [e["epoch"] for e in o["epochs"]]
        ys = [e["train_workload_ms"] / 1000 for e in o["epochs"]]
        ax.plot(xs, ys, marker="o", label=f"steerdb ({mode})")
    o = next(iter(online.values()))
    ax.axhline(o["postgres_train_ms"] / 1000, color=COLORS["postgres"], ls="--", label="postgres")
    ax.axhline(o["oracle_train_ms"] / 1000, color=COLORS["oracle"], ls="--", label="oracle")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train workload latency (s)")
    ax.set_title("Online learning curve")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def arms_ablation(report: dict, path: Path) -> Path | None:
    abl = report.get("ablations", {}).get("arms")
    if not abl:
        return None
    ks = list(abl)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for key, label in (("oracle", "oracle"), (None, "model")):
        ys = [abl[k][key or abl[k]["model"]]["total_vs_postgres"] for k in ks]
        ax.plot(
            [int(k) for k in ks],
            ys,
            marker="o",
            label=label,
            color=COLORS["oracle"] if key else COLORS["treecnn"],
        )
    ax.axhline(1.0, color=COLORS["postgres"], ls="--", label="postgres")
    ax.set_xlabel("number of arms K")
    ax.set_ylabel("total latency vs. Postgres")
    ax.set_title("Ablation: number of arms")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def make_all(report: dict, out_dir: Path, rel_to: Path | None = None) -> list[str]:
    out_dir = Path(out_dir)
    paths = [
        total_latency_bar(report, out_dir / "total_latency.png"),
        speedup_box(report, out_dir / "speedup_box.png"),
        learning_curve(report, out_dir / "learning_curve.png"),
        arms_ablation(report, out_dir / "arms_ablation.png"),
    ]
    paths = [p for p in paths if p is not None]
    if rel_to is None:
        return [str(p) for p in paths]
    return [Path(os.path.relpath(p, rel_to)).as_posix() for p in paths]
