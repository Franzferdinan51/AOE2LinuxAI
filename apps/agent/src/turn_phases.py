"""Per-turn decisions: context build, response parsing, action execution.

Owns the four pieces the game loop runs once per iteration:

  - `_get_ground_commands`: hardcoded actions that run alongside the LLM call
    (zoom on turn 1). Pure function of the iteration number.
  - `_build_llm_context`: stitches memory + goals + entities into the prompt
    string the executor sees. Mirrors `evaluation.context_builder._build_context`.
  - `record_llm_turn`: parses the LLM's response, strips the
    `[applied: ...]` memory-attribution prefix, snapshots state for reward
    computation, and returns `(actions, game_end_reason | None)`.
  - `_execute_turn_actions`: runs the validated actions through the executor
    and records success/failure feedback into memory, with a hardcoded
    fallback when the LLM returns no actions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import structlog

from .entity_utils import extract_attrs
from .executor import (
    CAMERA_KEYS,
    CASTLE_PREREQ_COUNT,
    FEUDAL_PREREQ_CLASSES,
    blocked_actions,
    build_steps,
    confirmed_buildings,
    execute_actions,
    get_detected_entities,
    get_rescan_fn,
    pending_placement_counts,
    sighted_buildings,
)
from .memory import GameState
from .models import validate_actions
from .villager_roles import infer_jobs, job_counts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .executor import ActionResult
    from .goal_logger import GoalLogger
    from .goals import GoalManager
    from .memory import AgentMemory
    from .providers.base import LLMResult

log = structlog.stdlib.get_logger()


_APPLIED_RE = re.compile(r"^\s*\[applied:\s*([^\]]+)\]", re.IGNORECASE)
_EXECUTOR_OUTAGE_STREAK = 3


def _extract_applied_memories(reasoning, loaded_titles):
    m = _APPLIED_RE.match(reasoning or "")
    if not m:
        return [], [], reasoning
    raw = [t.strip() for t in m.group(1).split(",") if t.strip()]
    known = [t for t in raw if t in loaded_titles]
    unknown = [t for t in raw if t not in loaded_titles]
    cleaned = reasoning[m.end():].lstrip()
    return known, unknown, cleaned


INITIAL_ZOOM_CLICKS = 5


def _get_ground_commands(iteration):
    """Return hardcoded actions injected BEFORE LLM actions each turn."""
    if iteration != 1:
        return []
    return [
        {"type": "scroll", "clicks": INITIAL_ZOOM_CLICKS, "intent": "Zoom in for better object detection"},
        {"type": "press", "key": ",", "intent": "Select scout (ground cmd)"},
        {"type": "press", "key": "g", "intent": "Auto Scout (ground cmd)"},
    ]


def _build_llm_context(memory, goal_manager, entity_summary, detected_entities=None):
    context = memory.get_context_for_llm()
    resource_context = goal_manager.get_resource_context()
    if resource_context:
        context = resource_context + "\n\n" + context
    goal_context = goal_manager.get_context_for_llm()
    if goal_context:
        context = goal_context + "\n\n" + context
    if entity_summary:
        entity_context = (
            "\n## Detected Entities (from YOLO)\n"
            "Use target_class or target_id to interact with these:\n" + entity_summary + "\n"
        )
        entity_context += _villager_jobs_line(detected_entities)
        entity_context += known_buildings_line(detected_entities)
        entity_context += castle_gate_line(memory.game_state.current_age)
        entity_context += blocked_actions_line()
        context = entity_context + "\n" + context
    return context


def _villager_jobs_line(detected_entities):
    if not detected_entities:
        return ""
    counts = job_counts(infer_jobs(detected_entities))
    working = {kind: n for kind, n in counts.items() if n}
    if not working:
        return ""
    breakdown = " ".join(f"{kind}={n}" for kind, n in working.items())
    return f"Villagers by job (approx, from proximity): {breakdown}\n"


def blocked_actions_line():
    blocked = blocked_actions()
    if not blocked:
        return ""
    return "Currently refused: " + " ".join(blocked) + "\n"


def castle_gate_line(age):
    if not age.startswith("Feudal"):
        return ""
    have = sorted(confirmed_buildings() & FEUDAL_PREREQ_CLASSES)
    names = " ".join(have) if have else "none"
    return (
        f"Feudal-Age buildings: {len(have)}/{CASTLE_PREREQ_COUNT} ({names}) — "
        f"the Castle Age needs {CASTLE_PREREQ_COUNT} of "
        f"{', '.join(sorted(FEUDAL_PREREQ_CLASSES))}\n"
    )


def known_buildings_line(detected_entities):
    confirmed = confirmed_buildings()
    pending = pending_placement_counts()
    unverified = sighted_buildings() - confirmed
    if not confirmed and not pending and not unverified:
        return ""
    segments = []
    if confirmed:
        detected = _class_counts(detected_entities or [])
        segments.append(" ".join(f"{c}={max(detected.get(c, 0), 1)}" for c in sorted(confirmed)))
    if pending:
        segments.append("(pending: " + " ".join(f"{c}={n}" for c, n in sorted(pending.items())) + ")")
    if unverified:
        segments.append("(unverified sightings, NOT owned: " + " ".join(sorted(unverified)) + ")")
    return "Known buildings: " + " ".join(segments) + "\n"


def record_llm_turn(response, memory, goal_manager, iteration, goal_logger):
    reasoning = response.get("reasoning", "")
    observations = response.get("observations", {})
    actions = response.get("actions", [])
    errored = bool(response.get("error", False))
    streak = memory.record_llm_outcome(errored=errored)
    if errored and streak == _EXECUTOR_OUTAGE_STREAK:
        log.error(
            "executor_outage",
            iteration=iteration,
            consecutive_failures=streak,
            detail=reasoning[:200],
            hint="every LLM path is failing; the reactive tier alone cannot build a mill",
        )
    loaded = set(memory.memories_loaded)
    known_titles, unknown_titles, reasoning = _extract_applied_memories(reasoning, loaded)
    if known_titles:
        memory.record_memories_applied(known_titles)
        log.info("memories_applied", iteration=iteration, titles=known_titles)
    if unknown_titles:
        log.warning(
            "memories_applied_unknown",
            iteration=iteration,
            titles=unknown_titles,
            loaded=sorted(loaded),
        )
    log.info(
        "llm_response",
        iteration=iteration,
        reasoning=reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
        action_count=len(actions),
    )
    prev_state = GameState(
        resources=dict(memory.game_state.resources),
        population=memory.game_state.population,
        population_cap=memory.game_state.population_cap,
        current_age=memory.game_state.current_age,
    )
    turn = memory.create_turn(reasoning=reasoning, actions=actions, observations=observations)
    goal_manager.evaluate_progress(memory.game_state, iteration)
    reward = goal_manager.compute_turn_reward(prev_state, memory.game_state)
    turn.reward = reward.get("total", 0.0)
    goal_logger.log_progress(iteration, goal_manager.active_goals, reward)
    for goal in goal_manager.completed_goals:
        if goal.created_turn != iteration:
            goal_logger.log_goal_completed(iteration, goal)
    if reward["total"] != 0:
        log.info("turn_reward", iteration=iteration, **reward)
    return actions


def check_game_over(response, memory, iteration):
    observations = response.get("observations", {})
    state = observations.get("game_state", "playing") if observations else "playing"
    if state in ("victory", "defeat"):
        log.info("game_over_detected", result=state, iteration=iteration)
        return str(state)
    return None


_BUILDING_CLASSES = frozenset({
    "town_center", "house", "lumber_camp", "mining_camp", "mill", "market", "dock",
    "farm", "barracks", "archery_range", "stable", "blacksmith", "siege_workshop",
    "monastery", "castle", "university", "gate", "wall", "tower", "wonder", "krepost",
})


@dataclass(frozen=True, slots=True)
class _Expectation:
    kind: Literal["new_building", "selection_change", "none"]
    detail: str


def _expectation_for(action):
    a_type = str(action.get("type", ""))
    intent = str(action.get("intent", "")).lower()
    if a_type == "build" or (
        a_type in ("click", "right_click") and ("build" in intent or "place" in intent)
    ):
        return _Expectation("new_building", "build/place should add a building")
    if a_type == "press":
        key = str(action.get("key", "")).lower()
        if action.get("rescan") or key in CAMERA_KEYS:
            return _Expectation("selection_change", f"press {key} should change the view")
    return _Expectation("none", "")


def _any_entity_expectation(actions):
    return any(_expectation_for(a).kind != "none" for a in actions if isinstance(a, dict))


def _class_counts(entities):
    counts = {}
    for e in entities:
        cls = extract_attrs(e).class_name
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _new_buildings(before, after):
    bc, ac = _class_counts(before), _class_counts(after)
    return sorted(cls for cls in _BUILDING_CLASSES if ac.get(cls, 0) > bc.get(cls, 0))


def _failed_lines(actions, results):
    lines = []
    for action, result in zip(actions, results, strict=True):
        if not result.success:
            a_intent = action.get("intent", "") if isinstance(action, dict) else ""
            a_type = action.get("type", "") if isinstance(action, dict) else ""
            lines.append(f"- FAILED {a_type}: {a_intent} — {result.detail}")
    return lines


def _build_verification(actions, results, before_entities, after_entities):
    lines = _failed_lines(actions, results)
    kinds = {_expectation_for(a).kind for a in actions if isinstance(a, dict)}
    if "new_building" in kinds:
        built = _new_buildings(before_entities, after_entities)
        if built:
            lines.append(f"- CONFIRMED built: {', '.join(built)}")
        else:
            lines.append("- no visible change: build produced no new building")
    counts_unchanged = _class_counts(before_entities) == _class_counts(after_entities)
    if "selection_change" in kinds and counts_unchanged:
        lines.append("- no visible change: view unchanged after camera action")
    return "\n".join(lines)


def _fallback_actions(memory):
    state = memory.game_state
    is_housed = state.population_cap > 0 and state.population >= state.population_cap
    if is_housed:
        return build_steps("q", "Place house to raise pop cap (fallback build)")
    return [
        {"type": "queue_villager", "intent": "Queue villager (fallback)"},
        {"type": "press", "key": ".", "rescan": True, "intent": "Select idle villager (fallback)"},
    ]


async def _execute_turn_actions(actions, iteration, memory, reasoning):
    if actions:
        verify = _any_entity_expectation(actions)
        before_entities = list(get_detected_entities()) if verify else []
        results = await execute_actions(actions)
        success_count = sum(1 for r in results if r.success)
        memory.record_action_results(success_count, len(actions))
        log.info("actions_executed", iteration=iteration, total=len(actions), successful=success_count)
        if verify:
            rescan = get_rescan_fn()
            if rescan is not None:
                await rescan()
            verification = _build_verification(actions, results, before_entities, list(get_detected_entities()))
        else:
            verification = "\n".join(_failed_lines(actions, results))
        if verification:
            memory.set_last_verification(verification)
    else:
        log.warning("no_actions_fallback", iteration=iteration, reasoning=reasoning[:200])
        fallback_actions = validate_actions(_fallback_actions(memory))
        if fallback_actions:
            fb_results = await execute_actions(fallback_actions)
            fb_success = sum(1 for r in fb_results if r.success)
            memory.record_action_results(fb_success, len(fallback_actions))
