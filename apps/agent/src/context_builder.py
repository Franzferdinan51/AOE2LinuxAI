"""ContextBuilder — assemble the LLM-facing world-state summary.

The agent and the strategist use slightly different prompts but share the
same underlying scene description, so this lives in its own module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .entity_utils import EntitySummary
    from .models import WorldState


ENTITY_DISPLAY_LIMIT = 25


@dataclass(frozen=True)
class ContextBundle:
    """Everything the LLM needs to plan the next action."""

    scene_text: str
    entity_text: str
    history_text: str
    goal_text: str
    memory_text: str

    def render(self) -> str:
        return "\n\n".join(
            part
            for part in (
                self.scene_text,
                self.entity_text,
                self.history_text,
                self.goal_text,
                self.memory_text,
            )
            if part
        )


def build_context(
    world: WorldState,
    entities: Iterable[dict],
    history: Sequence[dict],
    goals: Sequence[dict],
    memories: Sequence[str],
) -> ContextBundle:
    """Roll up all LLM inputs into a single renderable bundle."""
    from .entity_utils import build_entity_summary

    summary = build_entity_summary(list(entities))
    return ContextBundle(
        scene_text=_scene_text(world),
        entity_text=summary.compact,
        history_text=_history_text(history),
        goal_text=_goal_text(goals),
        memory_text=_memory_text(memories),
    )


def _scene_text(world: WorldState) -> str:
    age = getattr(world, "current_age", "Unknown")
    turn = getattr(world, "turn", 0)
    resources = getattr(world, "resources", {})
    if not isinstance(resources, dict):
        resources = {}
    res_line = ", ".join(f"{k}={v}" for k, v in sorted(resources.items()))
    return f"Age: {age} | Turn: {turn} | Resources: {res_line}"


def _history_text(history: Sequence[dict]) -> str:
    if not history:
        return ""
    recent = history[-5:]
    lines = [f"- turn {h.get('turn', '?')}: {h.get('summary', '')}" for h in recent]
    return "Recent turns:\n" + "\n".join(lines)


def _goal_text(goals: Sequence[dict]) -> str:
    if not goals:
        return ""
    lines = [f"- [{g.get('priority', 'P?')}] {g.get('description', '')}" for g in goals]
    return "Active goals:\n" + "\n".join(lines)


def _memory_text(memories: Sequence[str]) -> str:
    if not memories:
        return ""
    return "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories)
