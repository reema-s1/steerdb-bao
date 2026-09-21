"""Choose a plan among candidates: greedy argmin, Thompson sampling, plus the safety guard.

Safety guard (design doc section 6.6):
  1. Untrained model -> arm 0.
  2. Only deviate from arm 0 if the predicted speedup is at least `min_gain`
     (exploit mode only; exploration is allowed to try uncertain arms).
  3. The chosen plan runs with statement_timeout = timeout_factor x arm-0 latency; on timeout
     the caller cancels, reruns arm 0 and records the failure as a negative example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .arms import DEFAULT_ARM


@dataclass
class Decision:
    index: int  # index into the candidate list
    arm: int
    reason: str  # "model" | "explore" | "untrained" | "guard"
    predicted_ms: list[float] = field(default_factory=list)


class Selector:
    def __init__(
        self,
        mode: str = "greedy",
        min_gain: float = 0.05,
        timeout_factor: float = 2.0,
        timeout_floor_ms: float = 1000.0,
        seed: int = 0,
    ) -> None:
        if mode not in ("greedy", "thompson"):
            raise ValueError(f"unknown selector mode {mode!r}")
        self.mode = mode
        self.min_gain = min_gain
        self.timeout_factor = timeout_factor
        self.timeout_floor_ms = timeout_floor_ms
        self.rng = np.random.default_rng(seed)

    def choose(
        self,
        arms: list[int],
        mu: np.ndarray | None,
        sigma: np.ndarray | None = None,
        trained: bool = True,
    ) -> Decision:
        """arms[i] is the representative arm of candidate i; mu/sigma are log-ms predictions."""
        default_idx = arms.index(DEFAULT_ARM.id) if DEFAULT_ARM.id in arms else 0
        if not trained or mu is None:
            return Decision(default_idx, arms[default_idx], "untrained")

        mu = np.asarray(mu, dtype=np.float64)
        predicted_ms = [float(v) for v in np.exp(mu)]
        if self.mode == "thompson":
            sigma = np.zeros_like(mu) if sigma is None else np.asarray(sigma, dtype=np.float64)
            samples = self.rng.normal(mu, np.maximum(sigma, 1e-9))
            idx = int(np.argmin(samples))
            reason = "model" if idx == int(np.argmin(mu)) else "explore"
            return Decision(idx, arms[idx], reason, predicted_ms)

        idx = int(np.argmin(mu))
        if idx != default_idx:
            predicted_speedup = math.exp(mu[default_idx] - mu[idx])
            if predicted_speedup < 1.0 + self.min_gain:
                return Decision(default_idx, arms[default_idx], "guard", predicted_ms)
        return Decision(idx, arms[idx], "model", predicted_ms)

    def timeout_ms(self, baseline_ms: float | None) -> int | None:
        """statement_timeout for the chosen plan, or None if arm 0's latency is unknown."""
        if baseline_ms is None:
            return None
        return int(max(self.timeout_factor * baseline_ms, self.timeout_floor_ms))
