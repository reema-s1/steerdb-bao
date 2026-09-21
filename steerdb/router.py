"""Inference path: plan -> score -> choose -> execute (with the safety timeout fallback)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .arms import ARMS, DEFAULT_ARM, Arm, get_arm
from .executor import ExecResult, execute
from .models import ValueModel
from .plan_gen import Candidate, candidate_plans
from .selector import Decision, Selector


@dataclass
class RouteResult:
    candidates: list[Candidate]
    decision: Decision
    planning_ms: float
    inference_ms: float
    result: ExecResult | None = None
    fell_back: bool = False  # chosen plan hit the safety timeout, arm 0 was rerun
    failed_result: ExecResult | None = None


class Router:
    def __init__(
        self,
        conn,
        model: ValueModel | None,
        selector: Selector | None = None,
        arms: tuple[Arm, ...] = ARMS,
    ) -> None:
        self.conn = conn
        self.model = model
        self.selector = selector or Selector(mode="greedy")
        self.arms = arms

    def choose(self, sql: str) -> RouteResult:
        cands, planning_ms = candidate_plans(self.conn, sql, self.arms)
        t0 = time.perf_counter()
        trained = self.model is not None and self.model.trained
        mu = sigma = None
        if trained:
            mu, sigma = self.model.predict([c.plan for c in cands])
        inference_ms = (time.perf_counter() - t0) * 1000
        decision = self.selector.choose([c.arm for c in cands], mu, sigma, trained)
        return RouteResult(cands, decision, planning_ms, inference_ms)

    def run(self, sql: str, baseline_ms: float | None = None, **exec_kwargs) -> RouteResult:
        route = self.choose(sql)
        arm = get_arm(route.decision.arm)
        timeout = self.selector.timeout_ms(baseline_ms) if arm.id != DEFAULT_ARM.id else None
        res = execute(self.conn, sql, arm, timeout, **exec_kwargs)
        if res.timed_out and arm.id != DEFAULT_ARM.id:
            route.failed_result = res
            route.fell_back = True
            res = execute(self.conn, sql, DEFAULT_ARM, None, **exec_kwargs)
        route.result = res
        return route


def format_route(route: RouteResult) -> str:
    lines = [f"{'arms':<14}{'plan':<18}{'optimizer cost':>16}{'predicted ms':>14}"]
    preds = route.decision.predicted_ms or [np.nan] * len(route.candidates)
    for i, (c, p) in enumerate(zip(route.candidates, preds)):
        mark = " <- chosen" if i == route.decision.index else ""
        arms = ",".join(map(str, c.arms))
        lines.append(
            f"{arms:<14}{c.plan_hash:<18}{c.plan.get('Total Cost', 0):>16.1f}{p:>14.1f}{mark}"
        )
    lines.append(
        f"decision: arm {route.decision.arm} ({route.decision.reason}); "
        f"planning {route.planning_ms:.1f} ms, inference {route.inference_ms:.1f} ms"
    )
    if route.fell_back:
        lines.append("safety timeout hit: chosen plan cancelled, reran arm 0")
    if route.result is not None:
        flag = " (TIMEOUT)" if route.result.timed_out else ""
        lines.append(f"execution: {route.result.latency_ms:.1f} ms{flag}")
    return "\n".join(lines)
