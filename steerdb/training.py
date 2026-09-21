"""Turn stored experience into training sets and fit value models."""

from __future__ import annotations

from .models import ValueModel, make_model
from .store import ExperienceStore, Observation


def dedupe(observations: list[Observation]) -> list[Observation]:
    """Bootstrap rows for arms that share a plan are copies of one measurement; keep one."""
    seen: set[tuple[str, str]] = set()
    out = []
    for obs in observations:
        if obs.source == "bootstrap":
            key = (obs.query_name, obs.plan_hash)
            if key in seen:
                continue
            seen.add(key)
        out.append(obs)
    return out


def xy(observations: list[Observation]) -> tuple[list[dict], list[float]]:
    obs = dedupe(observations)
    return [o.plan for o in obs], [o.latency_ms for o in obs]


def train_model(
    kind: str,
    store: ExperienceStore,
    query_names: set[str],
    arm_ids: tuple[int, ...] | None = None,
    **model_kwargs,
) -> ValueModel:
    observations = store.observations(("bootstrap",), query_names)
    if arm_ids is not None:
        observations = [o for o in observations if o.arm in arm_ids]
    if not observations:
        raise RuntimeError(
            "no bootstrap experience for the training queries; run `steerdb collect`"
        )
    plans, lat = xy(observations)
    model = make_model(kind, **model_kwargs)
    model.fit(plans, lat)
    return model
