"""Choose a plan among candidates: greedy argmin, Thompson sampling, plus the safety guard.

Safety guard:
  1. Untrained model -> arm 0.
  2. Pessimistic deviation (exploit mode): leave arm 0 only if the predicted speedup still
     exceeds `min_gain` after subtracting `guard_k` standard deviations of the ensemble's
     disagreement:  (mu_0 - mu_c) - guard_k * sqrt(sd_c^2 + sd_0^2) > log(1 + min_gain).
     Defaults come from leave-template-out cross-validation on JOB's training templates.
     (Exploration in Thompson mode is allowed to try uncertain arms.)
  3. The chosen plan runs with statement_timeout = timeout_factor x arm-0 latency (no large
     floor: with a 1 s floor a mistake on a 100 ms query cost 10x); on timeout the caller
     cancels, reruns arm 0 and records the failure as a negative example.
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
        guard_k: float = 1.0,
        timeout_factor: float = 2.0,
        timeout_floor_ms: float = 1.0,
        seed: int = 0,
    ) -> None:
        if mode not in ("greedy", "thompson"):
            raise ValueError(f"unknown selector mode {mode!r}")
        self.mode = mode
        self.min_gain = min_gain
        self.guard_k = guard_k
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

        sd = np.zeros_like(mu) if sigma is None else np.asarray(sigma, dtype=np.float64)
        margin = (mu[default_idx] - mu) - self.guard_k * np.sqrt(sd**2 + sd[default_idx] ** 2)
        margin[default_idx] = -np.inf
        idx = int(np.argmax(margin))
        if len(arms) == 1 or margin[idx] <= math.log(1.0 + self.min_gain):
            best = int(np.argmin(mu))
            reason = "model" if best == default_idx else "guard"
            return Decision(default_idx, arms[default_idx], reason, predicted_ms)
        return Decision(idx, arms[idx], "model", predicted_ms)

    def timeout_ms(self, baseline_ms: float | None) -> int | None:
        """statement_timeout for the chosen plan, or None if arm 0's latency is unknown."""
        if baseline_ms is None:
            return None
        return int(max(self.timeout_factor * baseline_ms, self.timeout_floor_ms))
