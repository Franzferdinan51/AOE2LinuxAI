"""Action executor module for AoE2 LLM Agent.

Dispatches validated actions to per-type handler functions.
"""

import asyncio
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import structlog
from pydantic import BaseModel

from .config import config
from .entity_utils import CLASSES_BY_KIND, nearest_center_of_classes
from .io import InputInjector, select_input_backend
from .io.linux_input import SilentInputInjector
from .models import Action, validate_action
from .window import ensure_game_focused, get_game_window_rect

log = structlog.stdlib.get_logger()


def _now() -> float:
    return time.monotonic()


# Pluggable input backend — defaults to silent (records-only) so the agent
# never fires real clicks during tests. `set_input_injector()` swaps it
# before any action runs.
_input: InputInjector = SilentInputInjector()


def set_input_injector(injector: InputInjector | None = None) -> None:
    """Override the action layer's input injector."""
    global _input
    _input = injector if injector is not None else select_input_backend()


def get_input_injector() -> InputInjector:
    """Return the currently-installed input injector."""
    return _input

# Module-level state (updated per-action batch)
_window_offset: tuple[int, int] = (0, 0)
_detected_entities: list[dict] = []
_rescan_fn: Callable[[], Awaitable[None]] | None = None
_rescan_full_fn: Callable[[], Awaitable[None]] | None = None


@dataclass
class ActionResult:
    """Result of executing a single action."""

    success: bool
    detail: str


def set_rescan_fn(fn: Callable[[], Awaitable[None]]) -> None:
    """Set the rescan callback for mid-turn screenshot+detection."""
    global _rescan_fn
    _rescan_fn = fn


def set_rescan_full_fn(fn: Callable[[], Awaitable[None]]) -> None:
    """Set the full detection callback for thorough SAHI scan."""
    global _rescan_full_fn
    _rescan_full_fn = fn


def get_rescan_fn() -> Callable[[], Awaitable[None]] | None:
    """Return the registered fast-rescan callback, or None if unset."""
    return _rescan_fn


def set_detected_entities(entities: Sequence[object]) -> None:
    """Cache detected entities for target_id/target_class resolution."""
    global _detected_entities
    normalized: list[dict] = []
    for e in entities:
        to_dict = getattr(e, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if isinstance(converted, dict):
                normalized.append(converted)
        elif isinstance(e, dict):
            normalized.append(e)
        else:
            log.warning("detected_entity_unrecognized_type", entity_type=type(e).__name__)
    _detected_entities = normalized
    record_building_sightings(str(e.get("class", "")) for e in normalized)
    log.debug("detected_entities_set", count=len(_detected_entities))


def get_detected_entities() -> list[dict]:
    """Return the current detected entity list."""
    return _detected_entities


def clear_detected_entities() -> None:
    """Clear the cached detected entities."""
    global _detected_entities
    _detected_entities = []


# Build gates: per-game state feeding build_rejection + placement settlement
_HOUSE_HEADROOM_MAX = 4
_GAME_POP_CAP_LIMIT = 200
_PLACEMENT_INCOME_SLACK = 20
_INCOME_EMA_WEIGHT = 0.5
_PLACEMENT_SETTLE_SECONDS = 30.0
_HOUSE_CLASS = "house"
_HOUSE_CAP_STEP = 5
_HOUSE_SETTLE_SECONDS = 50.0
_SIGHTING_MIN_FRAMES = 3

ECON_MENU = "q"
MILITARY_MENU = "w"
ADVANCED_MENU = "v"

_MENU_BUILDINGS: dict[str, dict[str, str]] = {
    ECON_MENU: {
        "q": "house", "w": "mill", "e": "mining_camp", "r": "lumber_camp",
        "a": "farm", "s": "blacksmith", "t": "dock",
    },
    MILITARY_MENU: {"q": "barracks", "w": "archery_range", "e": "stable"},
    ADVANCED_MENU: {"d": "market"},
}

_MENU_NAMES: dict[str, str] = {
    ECON_MENU: "Open economic build menu",
    MILITARY_MENU: "Open military build menu",
    ADVANCED_MENU: "Open advanced build menu",
}


def building_class(menu: str, key: str) -> str | None:
    """The class one menu key places, or None if the menu has no such entry."""
    return _MENU_BUILDINGS.get(menu, {}).get(key)


FEUDAL_PREREQ_CLASSES: frozenset[str] = frozenset(
    {"barracks", "archery_range", "stable", "blacksmith", "market"}
)
CASTLE_PREREQ_COUNT = 2

# Technologies
_RESEARCH_CONFIRM_FRACTION = 0.5
_RESEARCH_SETTLE_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class Tech:
    """One researchable item."""

    goto_key: str
    research_key: str
    goto_modifiers: tuple[str, ...] = ()
    requires: str = ""
    food: int = 0
    gold: int = 0
    wood: int = 0


_TECHS: dict[str, Tech] = {
    "castle_age": Tech(goto_key="h", research_key="z", food=800, gold=200),
    "loom": Tech(goto_key="h", research_key="a", gold=50),
    "wheelbarrow": Tech(goto_key="h", research_key="s", food=175, wood=50),
    "horse_collar": Tech(goto_key="i", goto_modifiers=("ctrl",), research_key="q", requires="mill", food=75, wood=75),
    "double_bit_axe": Tech(goto_key="z", goto_modifiers=("ctrl",), research_key="q", requires="lumber_camp", food=100, wood=50),
    "gold_mining": Tech(goto_key="g", goto_modifiers=("ctrl",), research_key="q", requires="mining_camp", food=100, wood=75),
}


@dataclass
class _PendingResearch:
    """A research awaiting confirmation from the HUD resource drop."""
    name: str
    tech: Tech
    before: dict[str, int]
    settle_deadline: float = 0.0


# Circuit breaker (T-530)
_MISSING_STREAK_LIMIT = 3
_MISSING_SUPPRESS_SECONDS = 50.0
# Villager-order ledger (T-531)
_STARTING_VILLAGERS = 4
_VILLAGER_FOOD_COST = 50
_VILLAGER_ORDER_TARGET_BY_AGE: dict[str, int] = {"Dark Age": 30, "Feudal Age": 35}
_NEXT_AGE: dict[str, str] = {"Dark Age": "Feudal Age", "Feudal Age": "Castle Age"}

SelectionMode = Literal["click", "idle_press", "unknown"]
Verdict = Literal["confirmed", "missing", "undecided"]


@dataclass
class _PendingPlacement:
    building_class: str
    wood_cost: int
    wood_before: int
    noted_at_snapshot: int
    cap_before: int = 0
    selected_by: SelectionMode = "unknown"
    point: tuple[int, int] = (0, 0)
    settle_deadline: float = 0.0

    @property
    def is_house(self) -> bool:
        return self.building_class == _HOUSE_CLASS


@dataclass
class _BuildGates:
    population: tuple[int, int] | None = None
    resources: dict[str, int] | None = None
    idle_present: bool | None = None
    selected_by: SelectionMode = "unknown"
    buildings_confirmed: set[str] = field(default_factory=set)
    building_sightings: dict[str, int] = field(default_factory=dict)
    pending_placements: list[_PendingPlacement] = field(default_factory=list)
    wood_income_per_snapshot: float | None = None
    missing_streaks: dict[str, int] = field(default_factory=dict)
    suppressed_until: dict[str, float] = field(default_factory=dict)
    pending_research: list[_PendingResearch] = field(default_factory=list)
    researched: set[str] = field(default_factory=set)
    research_blocked_until: dict[str, float] = field(default_factory=dict)
    snapshot_count: int = 0
    villagers_ordered: int = _STARTING_VILLAGERS
    current_age: str = "Dark Age"


_build_gates = _BuildGates()


def observe_hud(population, population_cap, resources, *, idle_present=None):
    """Feed this turn's HUD reading into the build gates."""
    _build_gates.snapshot_count += 1
    _observe_wood_income(resources.get("wood"))
    _settle_pending_placements(resources.get("wood"), population_cap)
    _settle_pending_research(resources)
    _build_gates.population = (population, population_cap)
    _build_gates.resources = dict(resources)
    _build_gates.idle_present = idle_present


def observe_age(age: str) -> None:
    """Sync the validated age into the gates."""
    if age:
        _build_gates.current_age = age


def _observe_wood_income(wood_now):
    wood_before = (_build_gates.resources or {}).get("wood")
    if wood_now is None or wood_before is None or _build_gates.pending_placements:
        return
    delta = max(wood_now - wood_before, 0)
    previous = _build_gates.wood_income_per_snapshot
    if previous is None:
        _build_gates.wood_income_per_snapshot = float(delta)
    else:
        _build_gates.wood_income_per_snapshot = _INCOME_EMA_WEIGHT * delta + (1 - _INCOME_EMA_WEIGHT) * previous


def _expected_income(noted_at_snapshot: int) -> float:
    ema = _build_gates.wood_income_per_snapshot
    if ema is None:
        return 0.0
    elapsed = max(_build_gates.snapshot_count - noted_at_snapshot, 1)
    return ema * elapsed


def _note_pending_placement(building_key, *, menu=ECON_MENU, point=(0, 0)):
    cls = building_class(menu, building_key)
    cost = _WOOD_COST_BY_CLASS.get(cls or "")
    wood_before = (_build_gates.resources or {}).get("wood")
    if cls is None or cost is None or wood_before is None:
        log.debug("placement_pending_dropped", building_key=building_key)
        return
    _, cap_before = _build_gates.population or (0, 0)
    is_house = cls == _HOUSE_CLASS
    _build_gates.pending_placements.append(
        _PendingPlacement(
            building_class=cls, wood_cost=cost, wood_before=wood_before,
            noted_at_snapshot=_build_gates.snapshot_count, cap_before=cap_before,
            selected_by=_build_gates.selected_by, point=point,
            settle_deadline=_now() + (_HOUSE_SETTLE_SECONDS if is_house else _PLACEMENT_SETTLE_SECONDS),
        )
    )


def _settle_pending_placements(wood_now, cap_now):
    if not _build_gates.pending_placements or wood_now is None:
        return
    still_pending = []
    spend_by_baseline = {}
    claimed_cap = {}
    for pending in _build_gates.pending_placements:
        verdict = _house_verdict(pending, cap_now, claimed_cap) if pending.is_house else _wood_verdict(pending, wood_now, spend_by_baseline)
        if verdict == "undecided" and _now() < pending.settle_deadline:
            still_pending.append(pending)
            continue
        evidence = _settlement_evidence(pending, wood_now, cap_now)
        if verdict == "confirmed":
            record_confirmed_buildings([pending.building_class])
            log.info("build_purchase_confirmed", **evidence)
        else:
            _note_missing_settlement(pending.building_class)
            log.warning("build_purchase_missing", **evidence, selected_by=pending.selected_by, x=pending.point[0], y=pending.point[1])
    _build_gates.pending_placements = still_pending


def _house_verdict(pending, cap_now, claimed):
    already = claimed.get(pending.cap_before, 0)
    if cap_now - pending.cap_before - already < _HOUSE_CAP_STEP:
        return "undecided"
    claimed[pending.cap_before] = already + _HOUSE_CAP_STEP
    return "confirmed"


def _wood_verdict(pending, wood_now, spend_by_baseline):
    if wood_now == pending.wood_before:
        return "undecided"
    spent = spend_by_baseline.get(pending.wood_before, 0)
    income = _expected_income(pending.noted_at_snapshot)
    budget = pending.wood_before - spent - pending.wood_cost + _PLACEMENT_INCOME_SLACK
    if wood_now - income > budget:
        return "missing"
    spend_by_baseline[pending.wood_before] = spent + pending.wood_cost
    return "confirmed"


def _settlement_evidence(pending, wood_now, cap_now):
    if pending.is_house:
        return {"building": pending.building_class, "cap_before": pending.cap_before, "cap_now": cap_now}
    return {"building": pending.building_class, "wood_before": pending.wood_before, "wood_now": wood_now, "cost": pending.wood_cost, "income_estimate": round(_expected_income(pending.noted_at_snapshot), 1)}


def _note_pending_research(name, tech):
    before = _build_gates.resources
    if before is None:
        log.debug("research_pending_dropped", tech=name)
        return
    _build_gates.pending_research.append(
        _PendingResearch(name=name, tech=tech, before=dict(before), settle_deadline=_now() + _RESEARCH_SETTLE_SECONDS)
    )


def _settle_pending_research(resources):
    if not _build_gates.pending_research:
        return
    still_pending = []
    for pending in _build_gates.pending_research:
        verdict = _research_verdict(pending, resources)
        if verdict == "undecided" and _now() < pending.settle_deadline:
            still_pending.append(pending)
            continue
        if verdict == "confirmed":
            _build_gates.researched.add(pending.name)
            log.info("research_confirmed", tech=pending.name)
        else:
            _build_gates.research_blocked_until[pending.name] = _now() + _MISSING_SUPPRESS_SECONDS
            log.warning("research_missing", tech=pending.name, retry_in_s=round(_MISSING_SUPPRESS_SECONDS), **_research_costs(pending.tech))
    _build_gates.pending_research = still_pending


def _research_verdict(pending, resources):
    shortfalls = []
    for kind, price in _research_costs(pending.tech).items():
        now = resources.get(kind)
        was = pending.before.get(kind)
        if now is None or was is None or now == was:
            return "undecided"
        shortfalls.append(was - now < price * _RESEARCH_CONFIRM_FRACTION)
    return "missing" if any(shortfalls) else "confirmed"


def _research_costs(tech):
    return {kind: price for kind, price in (("food", tech.food), ("gold", tech.gold), ("wood", tech.wood)) if price}


def _is_pop_capped():
    if _build_gates.population is None:
        return False
    population, cap = _build_gates.population
    return cap > 0 and population >= cap


def is_pop_capped():
    """Whether the HUD says no villager can be queued."""
    return _is_pop_capped()


def _clear_missing_streak(building_class):
    _build_gates.missing_streaks.pop(building_class, None)
    _build_gates.suppressed_until.pop(building_class, None)


def _note_missing_settlement(building_class):
    """Count a vanished placement; suppress the class after a streak (T-530)."""
    if building_class == _HOUSE_CLASS and _is_pop_capped():
        return
    streak = _build_gates.missing_streaks.get(building_class, 0) + 1
    _build_gates.missing_streaks[building_class] = streak
    if streak >= _MISSING_STREAK_LIMIT:
        _build_gates.suppressed_until[building_class] = _now() + _MISSING_SUPPRESS_SECONDS
        log.warning("build_suppressed", building=building_class, missing_streak=streak, retry_in_s=round(_MISSING_SUPPRESS_SECONDS))


def record_building_sightings(classes):
    """Count one detection frame's building sightings — informational ONLY."""
    for cls in set(classes) & GATE_BUILDING_CLASSES:
        _build_gates.building_sightings[cls] = _build_gates.building_sightings.get(cls, 0) + 1


def record_confirmed_buildings(classes):
    """Remember gate-relevant building classes the agent PROVABLY owns."""
    proven = set(classes) & GATE_BUILDING_CLASSES
    _build_gates.buildings_confirmed.update(proven)
    for cls in proven:
        _clear_missing_streak(cls)


def confirmed_buildings():
    return frozenset(_build_gates.buildings_confirmed)


def villagers_ordered():
    return _build_gates.villagers_ordered


def villager_queue_rejection():
    ordered = _build_gates.villagers_ordered
    target = _VILLAGER_ORDER_TARGET_BY_AGE.get(_build_gates.current_age)
    if target is not None and ordered >= target:
        reason = (
            f"villager target reached ({ordered} ordered, incl. the TC queue) — "
            f"keep villagers busy and bank resources for the {_NEXT_AGE[_build_gates.current_age]} instead"
        )
    else:
        food = (_build_gates.resources or {}).get("food")
        if food is None or food >= _VILLAGER_FOOD_COST:
            return None
        reason = f"villager costs {_VILLAGER_FOOD_COST} food, you have {food}"
    log.info("villager_queue_rejected", reason=reason, ordered=ordered)
    return reason


def sighted_buildings():
    return frozenset(cls for cls, frames in _build_gates.building_sightings.items() if frames >= _SIGHTING_MIN_FRAMES)


def blocked_actions():
    """Refusals the LLM cannot work out for itself."""
    now = _now()
    blocked = [
        f"{cls} (suppressed {round(until - now)}s)"
        for cls, until in sorted(_build_gates.suppressed_until.items())
        if now < until
    ]
    blocked += [
        f"{name} (retryable in {round(until - now)}s)"
        for name, until in sorted(_build_gates.research_blocked_until.items())
        if now < until
    ]
    blocked += [f"{name} (already researched)" for name in sorted(_build_gates.researched)]
    return blocked


def pending_placement_counts():
    return Counter(p.building_class for p in _build_gates.pending_placements)


def reset_build_gates():
    global _build_gates
    _build_gates = _BuildGates()


def build_rejection(building_key, intent="", *, menu=ECON_MENU):
    reason = _rejection_reason(building_key, menu)
    if reason is not None:
        log.info("build_rejected", building_key=building_key, reason=reason, intent=intent)
    return reason


def _committed_wood():
    return sum(p.wood_cost for p in _build_gates.pending_placements)


def _rejection_reason(building_key, menu):
    cls = building_class(menu, building_key)
    if cls is None:
        return None
    suppressed_until = _build_gates.suppressed_until.get(cls, 0.0)
    if _now() < suppressed_until:
        streak = _build_gates.missing_streaks.get(cls, 0)
        return (
            f"{cls} builds suppressed for {round(suppressed_until - _now())} more seconds: "
            f"{streak} placements in a row vanished without the wood being spent"
        )
    if cls in _UNIQUE_BUILDING_CLASSES:
        if cls in _build_gates.buildings_confirmed:
            return f"{cls} already built — one is enough"
        if any(p.building_class == cls for p in _build_gates.pending_placements):
            return f"{cls} placement already pending wood-delta settlement — don't double-build"
    if cls == "house" and _build_gates.population is not None:
        population, cap = _build_gates.population
        if cap >= _GAME_POP_CAP_LIMIT:
            return f"house skipped: population cap {cap} is already the game maximum"
        headroom = cap - population
        if headroom > _HOUSE_HEADROOM_MAX:
            return f"house skipped: population {population}/{cap} leaves {headroom} headroom"
    prereq = _BUILD_PREREQ_CLASS.get(building_key)
    if prereq is not None and prereq not in _build_gates.buildings_confirmed:
        return f"{cls} unavailable: requires a completed {prereq}"
    cost = _WOOD_COST_BY_CLASS.get(cls)
    if cost is not None and _build_gates.resources is not None:
        wood = _build_gates.resources.get("wood")
        committed = _committed_wood()
        if wood is not None and wood - committed < cost:
            spare = wood - committed
            if committed:
                return f"{cls} unavailable: costs {cost} wood and only {spare} is uncommitted"
            return f"{cls} unavailable: costs {cost} wood, you have {wood}"
    if building_key in _RESOURCE_REQUIRED_KEYS and _resource_anchor(building_key) is None:
        classes = ", ".join(sorted(_BUILD_ANCHOR_CLASSES[building_key]))
        return f"{cls} skipped: no {classes} visible to build against"
    return None


# Coordinate resolution
def _resolve_target_id(target_id):
    for entity in _detected_entities:
        if entity.get("id") == target_id:
            center = entity.get("center")
            if center:
                return (int(center[0]), int(center[1]))
    return None


def _resolve_target_class(target_class):
    for entity in _detected_entities:
        if entity.get("class") == target_class:
            center = entity.get("center")
            if center:
                return (int(center[0]), int(center[1]))
    return None


def _to_int(value):
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"Expected int-coercible value, got {type(value).__name__}")


def _resolve_coords(action_dict):
    """Resolve action coordinates from auto_placement, targets, or x/y fields."""
    if action_dict.get("auto_placement"):
        key = str(action_dict.get("building_key", ""))
        placement = default_build_placement(key)
        if placement is None:
            return (f"no visible resource to anchor the {building_class(ECON_MENU, key) or key} on", None)
        return ("", placement)
    target_id = action_dict.get("target_id")
    if target_id:
        coords = _resolve_target_id(str(target_id))
        if coords is None:
            log.warning("target_id_not_found", target_id=target_id)
            return (f"target_id '{target_id}' not found in detected entities", None)
        return ("", coords)
    target_class = action_dict.get("target_class")
    if target_class:
        coords = _resolve_target_class(str(target_class))
        if coords is None:
            log.warning("target_class_not_found", target_class=target_class)
            return (f"target_class '{target_class}' not found in detected entities", None)
        return ("", coords)
    x, y = action_dict.get("x"), action_dict.get("y")
    if x is not None and y is not None:
        ix, iy = _to_int(x), _to_int(y)
        if ix == 0 and iy == 0:
            log.warning("placeholder_coords_rejected")
            return ("(0, 0) placeholder coordinates rejected", None)
        return ("", (ix, iy))
    return ("no coordinates, target_id, or target_class provided", None)


def can_resolve(action_dict):
    """Whether a targeted action still resolves against the current entity cache."""
    if not (action_dict.get("target_id") or action_dict.get("target_class")):
        return True
    error, _coords = _resolve_coords(action_dict)
    return not error


def _translate(x, y):
    return (x + _window_offset[0], y + _window_offset[1])


BUILD_PLACEMENT_KEYWORDS = ("place", "build")
CAMERA_KEYS = frozenset({"h", ".", ","})
STALE_COORDS_DETAIL = (
    "raw x/y coordinates go stale once the camera moves (a '.'/'h'/',' press "
    "re-centers the view) — use target_class or target_id instead"
)
BUILD_RETRY_RADIUS = 130
BUILD_RETRY_ATTEMPTS = 6
BUILD_SETTLE_DELAY = 0.15
BUILD_RETRY_DELAY = 0.1
RESCAN_SETTLE_DELAY = 0.3
DEFAULT_WAIT_MS = 100

BUILD_RING_RADII = (280, 400, 520)
RESOURCE_RING_RADII = (150, 210, 270)
BUILD_RING_DIRECTIONS = 8
BUILD_CLUTTER_RADIUS = 160

UI_MARGIN_TOP = 160
UI_MARGIN_BOTTOM = 240
UI_MARGIN_SIDE = 40

_BUILD_ANCHOR_CLASSES: dict[str, frozenset[str]] = {
    "r": CLASSES_BY_KIND["wood"],
    "e": CLASSES_BY_KIND["gold"] | CLASSES_BY_KIND["stone"],
    "w": frozenset({"berry_bush"}),
}
_ANCHOR_OPTIONAL_KEYS = frozenset({"w"})
_RESOURCE_REQUIRED_KEYS = frozenset(_BUILD_ANCHOR_CLASSES) - _ANCHOR_OPTIONAL_KEYS

GATE_BUILDING_CLASSES: frozenset[str] = frozenset(cls for menu in _MENU_BUILDINGS.values() for cls in menu.values())

_WOOD_COST_BY_CLASS: dict[str, int] = {
    "house": 25, "farm": 60, "mill": 100, "mining_camp": 100, "lumber_camp": 100,
    "blacksmith": 150, "dock": 150, "barracks": 175, "archery_range": 175,
    "stable": 175, "market": 175,
}

_BUILD_WOOD_COST = {key: _WOOD_COST_BY_CLASS[cls] for key, cls in _MENU_BUILDINGS[ECON_MENU].items()}

_BUILD_PREREQ_CLASS: dict[str, str] = {"a": "mill"}
_UNIQUE_BUILDING_CLASSES: frozenset[str] = frozenset({"mill", "lumber_camp"})

_build_retry_total_seconds = 0.0
_build_retry_count = 0

_DEFAULT_SCREEN = (3024, 1672)
_MIN_WINDOW_DIM = 320


def _window_size():
    rect = get_game_window_rect()
    if rect is not None:
        try:
            width, height = int(rect[2]), int(rect[3])
            if width >= _MIN_WINDOW_DIM and height >= _MIN_WINDOW_DIM:
                return width, height
        except (TypeError, ValueError, IndexError):
            pass
    return _DEFAULT_SCREEN


def _play_area_bounds():
    width, height = _window_size()
    return (UI_MARGIN_SIDE, UI_MARGIN_TOP, width - UI_MARGIN_SIDE, height - UI_MARGIN_BOTTOM)


def _in_play_area(x, y):
    min_x, min_y, max_x, max_y = _play_area_bounds()
    return min_x <= x <= max_x and min_y <= y <= max_y


def _clutter_score(point):
    px, py = point
    r2 = BUILD_CLUTTER_RADIUS * BUILD_CLUTTER_RADIUS
    return sum(
        1 for entity in _detected_entities
        if (center := entity.get("center")) and (px - center[0]) ** 2 + (py - center[1]) ** 2 <= r2
    )


def _open_ground_candidates(anchor, radii=BUILD_RING_RADII):
    ax, ay = anchor
    min_x, min_y, max_x, max_y = _play_area_bounds()
    candidates = []
    for radius in radii:
        for i in range(BUILD_RING_DIRECTIONS):
            angle = 2.0 * math.pi * i / BUILD_RING_DIRECTIONS
            cx = int(ax + radius * math.cos(angle))
            cy = int(ay + radius * math.sin(angle))
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                candidates.append((cx, cy))
    candidates.sort(key=_clutter_score)
    return candidates


def _home_anchor():
    tc = _resolve_target_class("town_center")
    if tc is not None:
        return tc
    width, height = _window_size()
    return (width // 2, height // 2)


def _resource_anchor(building_key):
    classes = _BUILD_ANCHOR_CLASSES.get(building_key)
    if classes is None:
        return None
    center = nearest_center_of_classes(_detected_entities, classes, _home_anchor())
    return None if center is None else (int(center[0]), int(center[1]))


def default_build_placement(building_key):
    resource = _resource_anchor(building_key)
    if resource is not None:
        candidates = _open_ground_candidates(resource, RESOURCE_RING_RADII)
        point = candidates[0] if candidates else resource
        log.debug("anchored_placement", building_key=building_key, anchor=resource, point=point, offset=round(math.dist(resource, point)), candidates=len(candidates))
        return point
    if building_key in _RESOURCE_REQUIRED_KEYS:
        return None
    anchor = _home_anchor()
    candidates = _open_ground_candidates(anchor)
    return candidates[0] if candidates else anchor


def build_menu_steps(building_key, intent, *, menu=ECON_MENU, menu_intent=""):
    return [
        {"type": "press", "key": menu, "intent": menu_intent or _MENU_NAMES[menu]},
        {"type": "press", "key": building_key, "intent": f"Select building ({intent})"},
        {"type": "click", "auto_placement": True, "building_key": building_key, "menu": menu, "intent": f"Place building ({intent})"},
        {"type": "press", "key": "h", "intent": f"Select TC to clear build UI ({intent})"},
    ]


def research_steps(name, intent):
    tech = _TECHS[name]
    return [
        {"type": "press", "key": tech.goto_key, "modifiers": list(tech.goto_modifiers), "rescan": True, "intent": f"Go to the {name} building ({intent})"},
        {"type": "press", "key": tech.research_key, "intent": f"Research {name} ({intent})"},
    ]


def research_rejection(name):
    reason = _research_rejection_reason(name)
    if reason is not None:
        log.info("research_rejected", tech=name, reason=reason)
    return reason


def _research_rejection_reason(name):
    tech = _TECHS.get(name)
    if tech is None:
        return f"unknown technology {name!r}; known: {', '.join(sorted(_TECHS))}"
    if name in _build_gates.researched:
        return f"{name} is already researched"
    blocked_until = _build_gates.research_blocked_until.get(name, 0.0)
    if _now() < blocked_until:
        return f"{name} did not take last time: retryable in {round(blocked_until - _now())} seconds"
    if any(p.name == name for p in _build_gates.pending_research):
        return f"{name} is already awaiting HUD settlement"
    if tech.requires and tech.requires not in _build_gates.buildings_confirmed:
        return f"{name} requires a {tech.requires} to be built first"
    resources = _build_gates.resources
    if resources is None:
        return None
    for kind, price in _research_costs(tech).items():
        have = resources.get(kind)
        if have is not None and have < price:
            return f"{name} unavailable: costs {price} {kind}, you have {have}"
    return None


def _select_villager_step(intent):
    nothing_is_idle = _build_gates.idle_present is False
    _build_gates.selected_by = "click" if nothing_is_idle else "idle_press"
    if nothing_is_idle:
        return {"type": "click", "target_class": "villager", "intent": f"Select villager ({intent})"}
    return {"type": "press", "key": ".", "rescan": True, "intent": f"Select idle villager ({intent})"}


def build_steps(building_key, intent, *, menu=ECON_MENU):
    return [_select_villager_step(intent), *build_menu_steps(building_key, intent, menu=menu)]


async def _handle_click(action_dict, intent):
    fail_detail, coords = _resolve_coords(action_dict)
    if coords is None:
        log.warning("click_no_coords", action=action_dict)
        return ActionResult(False, fail_detail)
    x, y = coords
    screen_x, screen_y = _translate(x, y)
    _input.click(screen_x, screen_y)
    log.info("click", x=x, y=y, screen_x=screen_x, screen_y=screen_y, target_id=action_dict.get("target_id", ""), intent=intent)
    if any(word in intent.lower() for word in BUILD_PLACEMENT_KEYWORDS):
        return await _finish_build_placement(action_dict, (x, y), (screen_x, screen_y))
    return ActionResult(True, "ok")


async def _finish_build_placement(action_dict, point, screen_point):
    global _build_retry_total_seconds, _build_retry_count
    x, y = point
    screen_x, screen_y = screen_point
    retry_start = time.monotonic()
    await asyncio.sleep(BUILD_SETTLE_DELAY)
    offsets = _compass_offsets(BUILD_RETRY_RADIUS, BUILD_RETRY_ATTEMPTS)
    for dx, dy in offsets:
        _input.click(screen_x + dx, screen_y + dy)
        await asyncio.sleep(BUILD_RETRY_DELAY)
    _input.right_click(screen_x, screen_y)
    elapsed = time.monotonic() - retry_start
    _build_retry_total_seconds += elapsed
    _build_retry_count += 1
    log.debug("build_placement_retry", x=x, y=y, offsets=offsets, elapsed_s=round(elapsed, 3), total_count=_build_retry_count, total_seconds=round(_build_retry_total_seconds, 1))
    building_key = action_dict.get("building_key")
    menu = str(action_dict.get("menu") or ECON_MENU)
    if isinstance(building_key, str):
        cls = building_class(menu, building_key)
        landed = await _verify_build_placement(cls, point)
        if landed is True and cls is not None:
            record_confirmed_buildings([cls])
        elif landed is False:
            _note_pending_placement(building_key, menu=menu, point=point)
            return ActionResult(True, "placement not visually confirmed; will be settled against the wood spend next turn")
    return ActionResult(True, "ok")


def _compass_offsets(radius, count):
    return [(int(radius * math.cos(a)), int(radius * math.sin(a))) for i in range(count) for a in (2.0 * math.pi * i / count,)]


def _count_class_near(class_name, point):
    px, py = point
    r2 = BUILD_CLUTTER_RADIUS * BUILD_CLUTTER_RADIUS
    return sum(1 for entity in _detected_entities if entity.get("class") == class_name and (center := entity.get("center")) and (px - center[0]) ** 2 + (py - center[1]) ** 2 <= r2)


async def _verify_build_placement(expected, point):
    if expected is None or _rescan_fn is None:
        return None
    before = _count_class_near(expected, point)
    await asyncio.sleep(RESCAN_SETTLE_DELAY)
    await _rescan_fn()
    landed = _count_class_near(expected, point) > before
    if landed:
        log.info("build_placement_verified", building=expected, x=point[0], y=point[1])
    else:
        log.info("build_placement_unconfirmed", building=expected, x=point[0], y=point[1])
    return landed


_ACTOR_CLASSES = frozenset({"villager", "town_center"})


def _re_resolve_from_intent(x, y, intent):
    intent_lower = intent.lower()
    for entity in _detected_entities:
        cls = entity.get("class", "")
        if cls and cls not in _ACTOR_CLASSES and cls in intent_lower:
            resolved = _resolve_target_class(cls)
            if resolved:
                log.debug("coords_re_resolved", cls=cls, old_x=x, old_y=y, new_x=resolved[0], new_y=resolved[1])
                return resolved
            break
    return (x, y)


async def _handle_right_click(action_dict, intent):
    fail_detail, coords = _resolve_coords(action_dict)
    if coords is None:
        log.warning("right_click_no_coords", action=action_dict)
        return ActionResult(False, fail_detail)
    x, y = coords
    if not action_dict.get("target_id") and not action_dict.get("target_class"):
        x, y = _re_resolve_from_intent(x, y, intent)
    if not _in_play_area(x, y):
        log.warning("right_click_off_map", x=x, y=y, intent=intent)
        return ActionResult(False, f"({x}, {y}) is in the HUD margin, not on the map")
    screen_x, screen_y = _translate(x, y)
    _input.right_click(screen_x, screen_y)
    log.info("right_click", x=x, y=y, screen_x=screen_x, screen_y=screen_y, target_id=action_dict.get("target_id", ""), intent=intent)
    return ActionResult(True, "ok")


async def _handle_press(action_dict, intent):
    key = str(action_dict["key"])
    raw_modifiers = action_dict.get("modifiers", [])
    modifiers = list(raw_modifiers) if isinstance(raw_modifiers, list) else []
    if modifiers:
        _input.hotkey(modifiers, key)
        log.info("press", key=key, modifiers=modifiers, intent=intent)
    else:
        _input.press(key)
        log.info("press", key=key, intent=intent)
    if action_dict.get("rescan") and _rescan_fn:
        await asyncio.sleep(RESCAN_SETTLE_DELAY)
        await _rescan_fn()
        log.info("rescan_after_press", key=key)
    return ActionResult(True, "ok")


async def _handle_drag(action_dict, intent):
    sx = _to_int(action_dict["start_x"]); sy = _to_int(action_dict["start_y"])
    ex = _to_int(action_dict["end_x"]); ey = _to_int(action_dict["end_y"])
    screen_sx, screen_sy = _translate(sx, sy)
    screen_ex, screen_ey = _translate(ex, ey)
    _input.move_to(screen_sx, screen_sy)
    _input.drag((screen_sx, screen_sy), (screen_ex, screen_ey), duration=0.2)
    log.info("drag", start_x=sx, start_y=sy, end_x=ex, end_y=ey, intent=intent)
    return ActionResult(True, "ok")


async def _handle_scroll(action_dict, intent):
    clicks = _to_int(action_dict["clicks"])
    x, y = action_dict.get("x"), action_dict.get("y")
    if x is not None and y is not None:
        screen_x, screen_y = _translate(_to_int(x), _to_int(y))
        _input.scroll(clicks, x=screen_x, y=screen_y)
    else:
        _input.scroll(clicks)
    log.info("scroll", clicks=clicks, intent=intent)
    return ActionResult(True, "ok")


async def _handle_detect(_action_dict, intent):
    if _rescan_full_fn:
        await _rescan_full_fn()
        log.info("full_detection", intent=intent)
        return ActionResult(True, "ok")
    log.warning("full_detection_unavailable")
    return ActionResult(False, "full detection not available")


async def _handle_wait(action_dict, intent):
    ms = _to_int(action_dict.get("ms", DEFAULT_WAIT_MS))
    await asyncio.sleep(ms / 1000)
    log.info("wait", ms=ms, intent=intent)
    return ActionResult(True, "ok")


async def _handle_build(action_dict, intent):
    key = action_dict.get("building_key")
    if not isinstance(key, str) or not key:
        return ActionResult(False, "build: missing building_key")
    menu = str(action_dict.get("menu") or ECON_MENU)
    rejection = build_rejection(key, intent, menu=menu)
    if rejection is not None:
        return ActionResult(False, rejection)
    for step in build_steps(key, intent, menu=menu):
        result = await execute_action(step)
        if not result.success:
            return ActionResult(False, f"build failed at: {step.get('intent', '')}")
    return ActionResult(True, f"built ({intent})")


async def _handle_research(action_dict, intent):
    name = action_dict.get("tech")
    if not isinstance(name, str) or not name:
        return ActionResult(False, "research: missing tech")
    rejection = research_rejection(name)
    if rejection is not None:
        return ActionResult(False, rejection)
    for step in research_steps(name, intent):
        result = await execute_action(step)
        if not result.success:
            return ActionResult(False, f"research failed at: {step.get('intent', '')}")
    _note_pending_research(name, _TECHS[name])
    return ActionResult(True, f"{name} pressed; the HUD spend settles it next turn")


async def _handle_queue_villager(action_dict, intent):
    rejection = villager_queue_rejection()
    if rejection is not None:
        return ActionResult(False, rejection)
    steps = [
        {"type": "press", "key": "h", "intent": f"Select TC ({intent})"},
        {"type": "press", "key": "q", "intent": f"Queue villager ({intent})"},
    ]
    for step in steps:
        result = await execute_action(step)
        if not result.success:
            return ActionResult(False, f"queue_villager failed at: {step.get('intent', '')}")
    _build_gates.villagers_ordered += 1
    log.info("villager_ordered", total=_build_gates.villagers_ordered, intent=intent)
    return ActionResult(True, f"villager queued ({_build_gates.villagers_ordered} ordered)")


_ACTION_HANDLERS = {
    "click": _handle_click, "right_click": _handle_right_click, "press": _handle_press,
    "build": _handle_build, "research": _handle_research, "queue_villager": _handle_queue_villager,
    "drag": _handle_drag, "scroll": _handle_scroll, "detect": _handle_detect, "wait": _handle_wait,
}


async def execute_action(action):
    """Execute a single action from LLM output."""
    if isinstance(action, BaseModel):
        action_dict = cast("dict[str, object]", action.model_dump())
    else:
        validated = validate_action(action)
        if not validated:
            log.warning("invalid_action", action=action)
            return ActionResult(False, "invalid action format")
        action_dict = cast("dict[str, object]", validated.model_dump())
    action_type_raw = action_dict.get("type", "")
    intent_raw = action_dict.get("intent", "")
    action_type = action_type_raw if isinstance(action_type_raw, str) else ""
    intent = intent_raw if isinstance(intent_raw, str) else ""
    handler = _ACTION_HANDLERS.get(action_type)
    if not handler:
        log.warning("unknown_action", action_type=action_type, action=action_dict)
        return ActionResult(False, f"unknown action type '{action_type}'")
    try:
        global _window_offset
        rect = get_game_window_rect()
        if rect:
            _window_offset = (rect[0], rect[1])
        result = await handler(action_dict, intent)
        if rect:
            _input.move_to(rect[0] + 1512, rect[1] + 836)
        await asyncio.sleep(config.action_delay)
        return result
    except KeyError as e:
        log.error("missing_action_param", action=action_dict, missing=str(e))
        return ActionResult(False, f"missing parameter: {e}")
    except Exception as e:
        log.error("action_failed", action=action_dict, error=str(e))
        return ActionResult(False, f"execution error: {e}")


def _as_dict(action):
    if isinstance(action, BaseModel):
        return cast("dict[str, object]", action.model_dump())
    return action


def _moves_camera(action_dict):
    return action_dict.get("type") == "press" and (
        bool(action_dict.get("rescan")) or str(action_dict.get("key", "")).lower() in CAMERA_KEYS
    )


def _uses_raw_coords_only(action_dict):
    return (
        action_dict.get("type") in ("click", "right_click")
        and action_dict.get("x") is not None
        and not action_dict.get("target_id")
        and not action_dict.get("target_class")
        and not action_dict.get("auto_placement")
    )


async def execute_actions(actions):
    """Execute a list of actions sequentially."""
    if not ensure_game_focused():
        log.warning("could_not_focus_before_actions")
        await asyncio.sleep(0.5)
        ensure_game_focused()
    results = []
    camera_moved = False
    for action in actions:
        preview = _as_dict(action)
        if camera_moved and _uses_raw_coords_only(preview):
            log.warning("stale_coords_rejected", intent=preview.get("intent", ""))
            results.append(ActionResult(False, STALE_COORDS_DETAIL))
            continue
        results.append(await execute_action(action))
        camera_moved = camera_moved or _moves_camera(preview)
    return results


__all__ = [
    "ActionResult", "build_menu_steps", "build_rejection", "build_steps",
    "blocked_actions", "BUILD_PLACEMENT_KEYWORDS", "CAMERA_KEYS",
    "CASTLE_PREREQ_COUNT", "clear_detected_entities", "confirmed_buildings",
    "default_build_placement", "ECON_MENU", "execute_action", "execute_actions",
    "FEUDAL_PREREQ_CLASSES", "get_detected_entities", "get_input_injector",
    "get_rescan_fn", "is_pop_capped", "observe_age", "observe_hud",
    "pending_placement_counts", "research_rejection", "research_steps",
    "reset_build_gates", "select_input_backend", "set_detected_entities",
    "set_input_injector", "set_rescan_fn", "set_rescan_full_fn",
    "sighted_buildings", "STALE_COORDS_DETAIL", "validate_actions",
    "villager_queue_rejection", "villagers_ordered",
]
