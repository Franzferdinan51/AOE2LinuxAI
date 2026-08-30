"""Goal model and reactive-tier trigger evaluation.

Goals are short-lived intentions ("build a house", "age up to Feudal"). The
agent opens and closes them as the world state changes. Reactive triggers
(auto-close / auto-open) keep the planner focused on the next bottleneck
without waiting for the LLM to be asked.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


GoalKind = Literal["build", "age_up", "produce", "scout", "attack", "defend", "gather", "research"]


# Classes that justify raising a defend / attack goal.
THREAT_CLASSES: frozenset[str] = frozenset(
    {"scout_cavalry", "knight", "militia", "man_at_arms", "spearman", "archer", "skirmisher"}
)


@dataclass
class Goal:
    """A single open or pending intention.

    `id` is a stable snake_case identifier — it's what the planner emits as
    ``[applied: goal_id]`` so the goal-log audit can replay exactly which goals
    drove each action.
    """

    id: str
    kind: GoalKind
    description: str
    priority: int = 50
    opened_at_turn: int = 0
    closes_at_turn: int | None = None
    satisfied: bool = False
    source: str = "agent"  # "agent" | "strategist" | "reactive"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "priority": self.priority,
            "opened_at_turn": self.opened_at_turn,
            "closes_at_turn": self.closes_at_turn,
            "satisfied": self.satisfied,
            "source": self.source,
            "metadata": self.metadata,
        }


def goal_priority(kind: GoalKind) -> int:
    """Default priority by kind — the planner keeps the queue sorted on this."""
    return {
        "age_up": 95,
        "defend": 90,
        "attack": 85,
        "build": 70,
        "produce": 65,
        "research": 60,
        "gather": 55,
        "scout": 40,
    }.get(kind, 50)


def deduplicate_open(goals: Iterable[Goal]) -> list[Goal]:
    """Drop duplicate ids — the LLM sometimes re-opens the same goal in the same turn."""
    seen: set[str] = set()
    out: list[Goal] = []
    for g in goals:
        if g.id in seen:
            continue
        seen.add(g.id)
        out.append(g)
    return out
