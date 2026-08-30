"""WorldState dataclass — the canonical mid-game state of the synth arena."""

from __future__ import annotations

from dataclasses import dataclass

AGE_SEQUENCE: list[str] = ["Dark Age", "Feudal Age", "Castle Age", "Imperial Age"]


@dataclass
class WorldState:
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
    turn: int = 0
