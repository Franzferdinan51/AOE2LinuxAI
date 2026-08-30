"""Event-log domain types — frozen Pydantic payloads + the EventSink Protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Discriminator, TypeAdapter

if TYPE_CHECKING:
    from datetime import datetime

    from core.world_state import WorldState


SCHEMA_VERSION = 1


class WorldStateSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    food: float
    wood: float
    gold: float
    stone: float
    population: int
    pop_cap: int
    age: str
    buildings: list[str]
    villager_queue: list[int]
    age_up_ticks_remaining: int
    turn: int

    @classmethod
    def from_world_state(cls, state: WorldState) -> WorldStateSnapshot:
        return cls(
            food=state.food, wood=state.wood, gold=state.gold, stone=state.stone,
            population=state.population, pop_cap=state.pop_cap, age=state.age,
            buildings=list(state.buildings), villager_queue=list(state.villager_queue),
            age_up_ticks_remaining=state.age_up_ticks_remaining, turn=state.turn,
        )

    def to_world_state(self) -> WorldState:
        from core.world_state import WorldState as _WorldState
        return _WorldState(
            food=self.food, wood=self.wood, gold=self.gold, stone=self.stone,
            population=self.population, pop_cap=self.pop_cap, age=self.age,
            buildings=list(self.buildings), villager_queue=list(self.villager_queue),
            age_up_ticks_remaining=self.age_up_ticks_remaining, turn=self.turn,
        )


class _PayloadBase(BaseModel):
    model_config = ConfigDict(frozen=True)


class TurnStartPayload(_PayloadBase):
    kind: Literal["turn_start"] = "turn_start"
    turn_num: int
    state: WorldStateSnapshot | None = None
    profile_name: str | None = None


class ObservationPayload(_PayloadBase):
    kind: Literal["observation"] = "observation"
    entity_count: int
    classes: list[str]


class LlmPromptPayload(_PayloadBase):
    kind: Literal["llm_prompt"] = "llm_prompt"
    state_summary: str


class LlmResponsePayload(_PayloadBase):
    kind: Literal["llm_response"] = "llm_response"
    actions: list[dict[str, object]]
    reasoning: str
    cost_usd: float


class ActionPayload(_PayloadBase):
    kind: Literal["action"] = "action"
    index_in_turn: int
    action: dict[str, object]


class ActionResultPayload(_PayloadBase):
    kind: Literal["action_result"] = "action_result"
    index_in_turn: int
    action_type: str
    state_changed: bool


class WorldMutationPayload(_PayloadBase):
    kind: Literal["world_mutation"] = "world_mutation"
    before_summary: str
    after_summary: str
    reason: str = ""


class ForkPayload(_PayloadBase):
    kind: Literal["fork"] = "fork"
    parent_run_id: str
    parent_t: int
    mutation_summary: str = ""


class MetricPayload(_PayloadBase):
    kind: Literal["metric"] = "metric"
    name: str
    value: float


Payload = Annotated[
    TurnStartPayload | ObservationPayload | LlmPromptPayload | LlmResponsePayload
    | ActionPayload | ActionResultPayload | WorldMutationPayload
    | ForkPayload | MetricPayload,
    Discriminator("kind"),
]


_PAYLOAD_ADAPTER: Final[TypeAdapter[Payload]] = TypeAdapter(Payload)


EventRow = tuple[str, str, int, str, str, "datetime", int]


@dataclass(frozen=True, slots=True)
class Event:
    run_id: str
    agent_id: str
    t: int
    payload: Payload
    ts: datetime
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_row(cls, row: EventRow) -> Event:
        run_id, agent_id, t, _kind, payload_json, ts, schema_version = row
        return cls(
            run_id=run_id, agent_id=agent_id, t=t,
            payload=_PAYLOAD_ADAPTER.validate_json(payload_json),
            ts=ts, schema_version=schema_version,
        )


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class NullEventSink:
    def emit(self, event: Event) -> None:
        return None


__all__ = [
    "SCHEMA_VERSION", "ActionPayload", "ActionResultPayload", "Event",
    "EventRow", "EventSink", "ForkPayload", "LlmPromptPayload",
    "LlmResponsePayload", "MetricPayload", "NullEventSink",
    "ObservationPayload", "Payload", "TurnStartPayload",
    "WorldMutationPayload", "WorldStateSnapshot",
]
