"""Fork primitive for event-sourced branching (Phase 5)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from evaluation.event_log import Event, ForkPayload, Payload, TurnStartPayload
from evaluation.world_sim import WorldState
from pydantic import TypeAdapter

if TYPE_CHECKING:
    import duckdb
    from evaluation.event_log import EventSink


MutationFn: TypeAlias = Callable[[WorldState], WorldState]

_PAYLOAD_ADAPTER: TypeAdapter[Payload] = TypeAdapter(Payload)


class ForkError(Exception):
    """Raised when fork() cannot reconstruct state from the requested event."""


def _load_turn_start(conn, run_id, t) -> TurnStartPayload:
    row = conn.execute(
        "SELECT payload_json FROM events "
        "WHERE run_id = ? AND kind = 'turn_start' AND t = ? LIMIT 1",
        (run_id, t),
    ).fetchone()
    if row is None:
        raise ForkError(f"no turn_start event at run_id={run_id!r} t={t}")
    payload = _PAYLOAD_ADAPTER.validate_json(row[0])
    if not isinstance(payload, TurnStartPayload):
        raise ForkError(f"unexpected payload kind at run_id={run_id!r} t={t}")
    return payload


def _diff_summary(before: WorldState, after: WorldState) -> str:
    parts: list[str] = []
    def _check(name: str, a, b) -> None:
        if a != b:
            parts.append(f"{name}={a!r}\u2192{b!r}")
    _check("food", before.food, after.food)
    _check("wood", before.wood, after.wood)
    _check("gold", before.gold, after.gold)
    _check("stone", before.stone, after.stone)
    _check("population", before.population, after.population)
    _check("pop_cap", before.pop_cap, after.pop_cap)
    _check("age", before.age, after.age)
    _check("buildings", before.buildings, after.buildings)
    _check("villager_queue", before.villager_queue, after.villager_queue)
    _check("age_up_ticks_remaining", before.age_up_ticks_remaining, after.age_up_ticks_remaining)
    _check("turn", before.turn, after.turn)
    return " ".join(parts)


def fork(conn, parent_run_id, parent_t, sink, mutation_fn=None):
    """Branch from (parent_run_id, parent_t) into a fresh run."""
    from datetime import UTC, datetime
    turn_start = _load_turn_start(conn, parent_run_id, parent_t)
    if turn_start.state is None:
        raise ForkError(f"turn_start at run_id={parent_run_id!r} t={parent_t} has no state snapshot")
    parent_state = turn_start.state.to_world_state()
    forked_state = mutation_fn(parent_state) if mutation_fn is not None else parent_state
    mutation_summary = _diff_summary(parent_state, forked_state) if mutation_fn is not None else ""
    new_run_id = uuid.uuid4().hex
    sink.emit(
        Event(
            run_id=new_run_id,
            agent_id="",
            t=parent_t,
            payload=ForkPayload(
                parent_run_id=parent_run_id,
                parent_t=parent_t,
                mutation_summary=mutation_summary,
            ),
            ts=datetime.now(UTC),
        )
    )
    return new_run_id, forked_state
