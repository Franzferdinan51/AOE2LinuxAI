"""Global game-loop state shared across the perceive/act clocks.

Keeps the last detected entities, world snapshot, and recent history in one
place so the executor and the strategist read the same view.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import WorldState

# Module-level state with a single re-entrant lock. The perceive clock writes,
# the act clock reads. The strategist phase reads atomically.
_state_lock = threading.RLock()
_detected_entities: list[dict] = []
_world_state: WorldState | None = None
_recent_turns: list[dict] = []
_recent_actions: list[dict] = []
# Mid-turn rescan callback (set by the executor module). Tests can swap it for
# a deterministic function via ``set_rescan_fn``.
_rescan_fn = None
_rescan_full_fn = None


def set_detected_entities(entities: list) -> None:
    """Replace the global detected-entities slot."""
    with _state_lock:
        # Normalize to dicts so downstream code can use ``.get`` uniformly.
        global _detected_entities
        _detected_entities = [e if isinstance(e, dict) else _entity_to_dict(e) for e in entities]


def get_detected_entities() -> list[dict]:
    """Snapshot of the latest detected entities (already a copy)."""
    with _state_lock:
        return list(_detected_entities)


def clear_detected_entities() -> None:
    with _state_lock:
        global _detected_entities
        _detected_entities = []


def _entity_to_dict(entity: Any) -> dict:
    """Best-effort dict normalization for non-dict DetectedEntity objects."""
    return {
        "id": getattr(entity, "id", "unknown"),
        "class": getattr(entity, "class_name", "unknown"),
        "center": tuple(getattr(entity, "center", (0, 0))),
        "confidence": float(getattr(entity, "confidence", 0.0)),
    }


def set_world_state(state: WorldState | None) -> None:
    with _state_lock:
        global _world_state
        _world_state = state


def get_world_state() -> WorldState | None:
    with _state_lock:
        return _world_state


def push_recent_turn(turn: dict, actions: list[dict]) -> None:
    """Keep a rolling window of the last 5 turns for prompt context."""
    with _state_lock:
        global _recent_turns, _recent_actions
        _recent_turns.append(turn)
        _recent_actions.append({"turn": turn.get("iteration"), "actions": actions})
        # Cap to 5 entries so the prompt stays bounded.
        _recent_turns = _recent_turns[-5:]
        _recent_actions = _recent_actions[-5:]


def get_recent_turns() -> list[dict]:
    with _state_lock:
        return list(_recent_turns)


def get_recent_actions() -> list[dict]:
    with _state_lock:
        return list(_recent_actions)


def set_rescan_fn(fn) -> None:
    """Register the executor's mid-turn rescan coroutine.

    The detector uses this to trigger a rescan without depending on the
    executor directly (which would be a circular import).
    """
    global _rescan_fn
    _rescan_fn = fn


def set_rescan_full_fn(fn) -> None:
    """Register the executor's full-rescan coroutine (forces a fresh detection)."""
    global _rescan_full_fn
    _rescan_full_fn = fn


def get_rescan_fn():
    return _rescan_fn


def get_rescan_full_fn():
    return _rescan_full_fn


def reset_for_test() -> None:
    """Clear all module state — for test isolation."""
    with _state_lock:
        global _detected_entities, _world_state, _recent_turns, _recent_actions
        _detected_entities = []
        _world_state = None
        _recent_turns = []
        _recent_actions = []
