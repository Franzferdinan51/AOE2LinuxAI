"""Idle-villager dispatch — procedural because it reads the entity list."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..entity_utils import (
    RESOURCE_KINDS,
    ResourceKind,
    first_center_of_class,
    nearest_class_of_kind,
)
from . import allocation

if TYPE_CHECKING:
    from .state import PolicyState

_IDLE_DISPATCH_PER_DECISION = 1
_IDLE_DISPATCH_MAX = 6
_IDLE_COUNT_SUSPECT_STREAK = 4

_FARM_BUILD_KEY = "a"

_FARM_WOOD_COST = 60
_WOOD_BANK_MARGIN = 20


def distribute_idle(
    entities: list[object],
    state: PolicyState,
    wood_target: int | None,
    strategist_allocation: allocation.Allocation | None = None,
) -> list[dict[str, object]]:
    """Route idle villagers one at a time toward the target allocation."""
    if not state.idle_present:
        return []
    batch = _idle_batch_size(state)
    if batch == 0:
        return []

    target_mix = allocation.for_state(state, strategist_allocation, wood_target)
    jobs = dict(state.villager_jobs)
    origin = _tc_origin(entities)
    actions: list[dict[str, object]] = []
    farm_queued = False
    for _ in range(batch):
        kind = allocation.next_kind(target_mix, jobs)
        jobs = allocation.with_one_more(jobs, kind)
        if kind == "food" and not farm_queued and nearest_class_of_kind(entities, "food") is None:
            actions.append(
                {
                    "type": "build",
                    "building_key": _FARM_BUILD_KEY,
                    "intent": "Build farm for idle villager (no forage/huntables visible)",
                }
            )
            farm_queued = True
            continue
        target = resolve_idle_target(entities, kind, origin)
        if target is None:
            break
        actions.append(
            {
                "type": "press",
                "key": ".",
                "rescan": True,
                "intent": f"Select next idle villager → {kind}",
            }
        )
        actions.append(
            {
                "type": "right_click",
                "target_class": target,
                "intent": f"Send idle villager to {target} ({kind})",
            }
        )
    return actions


def _idle_batch_size(state: PolicyState) -> int:
    """The badge count when trusted, else the blind batch. 0 means none idle."""
    if state.idle_count is None:
        return _IDLE_DISPATCH_PER_DECISION
    batch = min(state.idle_count, _IDLE_DISPATCH_MAX)
    if state.idle_streak >= _IDLE_COUNT_SUSPECT_STREAK:
        batch = max(batch, _IDLE_DISPATCH_PER_DECISION)
    return batch


def farm_bank_target() -> int:
    """The fallback target once a mill stands."""
    return _FARM_WOOD_COST + _WOOD_BANK_MARGIN


def _tc_origin(entities: list[object]) -> tuple[float, float]:
    """Town Center centre if detected, else the origin."""
    return first_center_of_class(entities, "town_center") or (0.0, 0.0)


def resolve_idle_target(
    entities: list[object], kind: ResourceKind, origin: tuple[float, float]
) -> str | None:
    """The requested kind if visible, else the next by gather priority."""
    target = nearest_class_of_kind(entities, kind, origin)
    if target is not None:
        return target
    for fallback in RESOURCE_KINDS:
        if fallback == kind:
            continue
        target = nearest_class_of_kind(entities, fallback, origin)
        if target is not None:
            return target
    return None
