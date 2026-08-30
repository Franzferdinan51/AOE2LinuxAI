"""Memory chain — ordered list of memory fragments with precedence semantics.

A "memory chain" is the planner's view of which memories apply to the
current state, in priority order. It is built every turn from the agent's
working memory + the persisted memory store.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .goals import Goal


@dataclass
class MemoryChain:
    """The active memory chain — used by the planner to decide which goal to advance."""

    items: list[dict] = field(default_factory=list)

    def add(self, item: dict) -> None:
        self.items.append(item)

    def truncate(self, n: int) -> None:
        self.items = self.items[:n]

    def filter_by(self, key: str, value: object) -> "MemoryChain":
        return MemoryChain(items=[i for i in self.items if i.get(key) == value])

    def render(self) -> str:
        if not self.items:
            return ""
        return "\n".join(f"- {i.get('id', '?')}: {i.get('description', '')}" for i in self.items)


def build_chain(goals: Iterable[Goal], memories: Iterable[str]) -> MemoryChain:
    """Build the planner-facing memory chain from open goals + loaded memories."""
    chain = MemoryChain()
    for g in sorted(goals, key=lambda x: -x.priority):
        chain.add({"id": g.id, "description": g.description, "kind": "goal", "priority": g.priority})
    for m in memories:
        chain.add({"id": m, "description": m, "kind": "memory"})
    return chain
