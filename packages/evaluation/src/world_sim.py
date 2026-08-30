"""Synthetic WorldState simulator for the arena.

The synth arena needs to evaluate many LLM strategies against a ground-truth
world without running the actual game. This module is the simulator: pure
Python, deterministic given a seed, fast enough to drive tens of thousands
of turns through `synth_game_loop` in seconds.

Every external action is validated by `apply_action` against the simulator's
own rules — a woolly action like `press f` outside the build menu just no-ops,
matching the real game. `tick()` then evolves resources, completes villagers,
ages up if the cost is banked, etc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, cast

from core.world_state import AGE_SEQUENCE, WorldState


VillagerActionKind = Literal["gather", "build", "queue", "research", "idle", "none"]

DEFAULT_VILLAGERS = 4
DEFAULT_POP_CAP = 200
FOOD_PER_VILLAGER_TICK = 5.0
WOOD_PER_VILLAGER_TICK = 5.0
GOLD_PER_VILLAGER_TICK = 0.0
STONE_PER_VILLAGER_TICK = 0.0
FOOD_PER_FARM_TICK = 25.0
BUILDING_WOOD_COST = {
    "house": 25, "farm": 60, "mill": 100, "mining_camp": 100,
    "lumber_camp": 100, "barracks": 175, "archery_range": 175,
    "stable": 175, "blacksmith": 150, "market": 175,
}
RESEARCH_COST = {
    "loom": {"gold": 50}, "wheelbarrow": {"food": 175, "wood": 50},
    "horse_collar": {"food": 75, "wood": 75},
    "double_bit_axe": {"food": 100, "wood": 50},
    "gold_mining": {"food": 100, "wood": 75},
    "castle_age": {"food": 800, "gold": 200},
    "imperial_age": {"food": 1000, "gold": 800},
}
AGE_FEUDAL_PREREQS = {"barracks", "archery_range", "stable", "blacksmith", "market"}
CASTLE_PREREQ_COUNT = 2
FEUDAL_PREREQ_COUNT = 2
AGE_UP_TICKS = 100


@dataclass
class SimConfig:
    seed: int = 0
    ticks_per_villager: int = 4
    initial_villagers: int = DEFAULT_VILLAGERS
    initial_food: int = 200
    initial_wood: int = 200
    initial_gold: int = 100
    initial_stone: int = 200
    initial_age: str = "Dark Age"


@dataclass
class TickResult:
    food_gained: float = 0.0
    wood_gained: float = 0.0
    gold_gained: float = 0.0
    stone_gained: float = 0.0
    villagers_completed: int = 0
    age_advanced_to: str | None = None


@dataclass
class ActionResult:
    success: bool
    detail: str
    state_changed: bool = False
    kind: VillagerActionKind = "none"


def init_from_fixture(inputs: dict) -> WorldState:
    """Build a WorldState from a scenario fixture's `inputs:` block."""
    resources = inputs.get("resources", {})
    detected = inputs.get("detected_entities", [])
    buildings = [e.get("class") for e in detected if e.get("class", "").endswith("_camp") or e.get("class") in ("mill", "barracks", "archery_range", "stable", "blacksmith", "market")]
    pop = inputs.get("population", "0/0")
    if isinstance(pop, str) and "/" in pop:
        cur, cap = pop.split("/")
        pop_i, cap_i = int(cur), int(cap)
    else:
        pop_i, cap_i = 0, DEFAULT_POP_CAP
    age = inputs.get("age", "Dark Age")
    return WorldState(
        food=float(resources.get("food", 200)),
        wood=float(resources.get("wood", 200)),
        gold=float(resources.get("gold", 0)),
        stone=float(resources.get("stone", 0)),
        population=pop_i,
        pop_cap=cap_i or DEFAULT_POP_CAP,
        age=age,
        buildings=list(dict.fromkeys(buildings)),
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


def state_to_fixture_inputs(state: WorldState, base_inputs: dict | None = None) -> dict:
    """Project a WorldState back to the fixture `inputs:` shape used by the executor."""
    inputs = dict(base_inputs or {})
    inputs["age"] = state.age
    inputs["resources"] = {
        "food": int(state.food), "wood": int(state.wood),
        "gold": int(state.gold), "stone": int(state.stone),
    }
    inputs["population"] = f"{state.population}/{state.pop_cap}"
    return inputs


def apply_action(state: WorldState, action: dict) -> WorldState:
    """Apply one LLM action to the simulator state. Returns the new state."""
    new_state = _clone_state(state)
    action_type = action.get("type", "")
    if action_type == "queue_villager":
        if new_state.food >= 50 and new_state.population < new_state.pop_cap:
            new_state.food -= 50
            new_state.villager_queue.append(25)
            return new_state
        return state
    if action_type == "build":
        building_key = action.get("building_key", "")
        menu = action.get("menu", "q")
        cls = _menu_building(menu, building_key)
        cost = BUILDING_WOOD_COST.get(cls or "", 0)
        if cls and new_state.wood >= cost:
            new_state.wood -= cost
            new_state.buildings.append(cls)
            if cls == "house":
                new_state.pop_cap += 5
            return new_state
        return state
    if action_type == "research":
        tech = action.get("tech", "")
        cost_dict = RESEARCH_COST.get(tech, {})
        if cost_dict and all(new_state.resources_get(k, 0) >= v for k, v in cost_dict.items()):
            for k, v in cost_dict.items():
                setattr(new_state, k, getattr(new_state, k) - v)
            new_state.age_up_ticks_remaining = AGE_UP_TICKS
            return new_state
        return state
    if action_type in ("press", "wait", "detect"):
        return state
    return state


def _clone_state(state: WorldState) -> WorldState:
    return WorldState(
        food=state.food, wood=state.wood, gold=state.gold, stone=state.stone,
        population=state.population, pop_cap=state.pop_cap, age=state.age,
        buildings=list(state.buildings), villager_queue=list(state.villager_queue),
        age_up_ticks_remaining=state.age_up_ticks_remaining, turn=state.turn,
    )


def _menu_building(menu: str, key: str) -> str | None:
    MENUS = {
        "q": {"q": "house", "w": "mill", "e": "mining_camp", "r": "lumber_camp", "a": "farm"},
        "w": {"q": "barracks", "w": "archery_range", "e": "stable"},
        "v": {"d": "market"},
    }
    return MENUS.get(menu, {}).get(key)


def evaluate_end_state(spec: dict, state: WorldState) -> list[str]:
    """Compare end_state spec against final state, returning failure strings."""
    failures: list[str] = []
    if "age" in spec:
        if state.age != spec["age"]:
            failures.append(f"end_state age: expected {spec['age']!r}, got {state.age!r}")
    if "population" in spec:
        target = spec["population"]
        if isinstance(target, (int, float)) and state.population < target:
            failures.append(f"end_state population: expected ≥ {target}, got {state.population}")
    return failures


def tick(state: WorldState) -> WorldState:
    """Advance the world by one tick: gather, queue drain, age-up."""
    import copy
    new_state = copy.deepcopy(state)
    new_state.turn += 1
    n_villagers = new_state.population
    new_state.food += n_villagers * FOOD_PER_VILLAGER_TICK
    new_state.wood += n_villagers * WOOD_PER_VILLAGER_TICK
    new_state.gold += n_villagers * GOLD_PER_VILLAGER_TICK
    new_state.stone += n_villagers * STONE_PER_VILLAGER_TICK
    n_farms = new_state.buildings.count("farm")
    if n_farms:
        new_state.food += n_farms * FOOD_PER_FARM_TICK
    new_state.villager_queue = [t - 1 for t in new_state.villager_queue if t > 1]
    completed = sum(1 for t in state.villager_queue if t == 1)
    new_state.population += completed
    if new_state.age_up_ticks_remaining > 0:
        new_state.age_up_ticks_remaining -= 1
        if new_state.age_up_ticks_remaining == 0 and new_state.age != AGE_SEQUENCE[-1]:
            idx = AGE_SEQUENCE.index(new_state.age)
            new_state.age = AGE_SEQUENCE[idx + 1]
    return new_state


__all__ = [
    "AGE_FEUDAL_PREREQS", "AGE_SEQUENCE", "ActionResult",
    "BUILDING_WOOD_COST", "CASTLE_PREREQ_COUNT", "DEFAULT_POP_CAP",
    "DEFAULT_VILLAGERS", "FEUDAL_PREREQ_COUNT", "RESEARCH_COST",
    "SimConfig", "TickResult", "VillagerActionKind", "apply_action",
    "evaluate_end_state", "init_from_fixture", "state_to_fixture_inputs", "tick",
]
