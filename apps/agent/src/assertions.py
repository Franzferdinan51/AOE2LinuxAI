"""Assertion DSL for scenario evaluation.

Each function takes (executed_actions, expected_value, context) and returns a
list of failure strings. Empty list = assertion passed.

Property-dict matching uses subset semantics: an action is considered a match
for `{"type": "press", "key": "h"}` if all expected keys are present with
equal values, regardless of extra fields on the action.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

REASONING_PREVIEW_CHARS = 300
ACTION_DISPLAY_KEYS = ("key", "building_key", "target_class", "target_id", "x", "y")


def matches(action: dict, pattern: dict) -> bool:
    """Subset match: every key in pattern must equal the same key in action."""
    return all(action.get(key) == expected for key, expected in pattern.items())


def _format_action(action: dict) -> str:
    """One-line action repr for failure messages."""
    parts = [f"type={action.get('type')!r}"]
    for key in ACTION_DISPLAY_KEYS:
        if key in action:
            parts.append(f"{key}={action[key]!r}")
    return "{" + ", ".join(parts) + "}"


def _format_action_list(actions: Iterable[dict]) -> str:
    items = list(actions)
    if not items:
        return "(no actions)"
    return "\n  - " + "\n  - ".join(_format_action(action) for action in items)


def _preview(reasoning: str) -> str:
    return reasoning[:REASONING_PREVIEW_CHARS] if reasoning else "(empty)"


# ---------------------------------------------------------------------------
# Action-list assertions
# ---------------------------------------------------------------------------


def must_include(actions: list[dict], pattern: dict, **_: object) -> list[str]:
    """Action list contains at least one action matching the pattern (anywhere)."""
    if not isinstance(pattern, dict):
        return [f"must_include expected a dict, got {type(pattern).__name__}"]
    if any(matches(action, pattern) for action in actions):
        return []
    return [
        f"must_include FAILED — no action matched {pattern!r}.\n"
        f"  Actual actions:{_format_action_list(actions)}"
    ]


def must_include_first(actions: list[dict], patterns: list[dict], **_: object) -> list[str]:
    """First N actions match exactly (ordered prefix)."""
    if not isinstance(patterns, list):
        return [f"must_include_first expected a list, got {type(patterns).__name__}"]
    for index, pattern in enumerate(patterns):
        if index >= len(actions):
            return [
                f"must_include_first FAILED — expected {len(patterns)} prefix actions, "
                f"got only {len(actions)}.\n  Actual:{_format_action_list(actions)}"
            ]
        if not matches(actions[index], pattern):
            return [
                f"must_include_first FAILED at index {index} — expected {pattern!r}, "
                f"got {_format_action(actions[index])}.\n"
                f"  Full actions:{_format_action_list(actions)}"
            ]
    return []


def must_not_include(actions: list[dict], pattern: dict, **_: object) -> list[str]:
    """Action list contains NO action matching the pattern."""
    if not isinstance(pattern, dict):
        return [f"must_not_include expected a dict, got {type(pattern).__name__}"]
    for action in actions:
        if matches(action, pattern):
            return [
                f"must_not_include FAILED — found forbidden action {_format_action(action)} "
                f"matching {pattern!r}.\n  Full actions:{_format_action_list(actions)}"
            ]
    return []


def _count_matches(actions: list[dict], spec: dict, default_n: int) -> tuple[int, int, dict]:
    """Return (expected_n, actual_count, pattern) extracted from a count spec."""
    expected_n = int(spec.get("n", default_n))
    pattern = {key: value for key, value in spec.items() if key != "n"}
    actual_count = sum(1 for action in actions if matches(action, pattern))
    return expected_n, actual_count, pattern


def count_at_least(actions: list[dict], spec: dict, **_: object) -> list[str]:
    """At least N actions match the type/pattern."""
    expected_n, actual, pattern = _count_matches(actions, spec, default_n=1)
    if actual < expected_n:
        return [
            f"count_at_least FAILED — expected ≥ {expected_n} actions matching {pattern!r}, "
            f"got {actual}.\n  Full actions:{_format_action_list(actions)}"
        ]
    return []


def count_at_most(actions: list[dict], spec: dict, **_: object) -> list[str]:
    """At most N actions match the type/pattern (n: 0 forbids the type entirely)."""
    expected_n, actual, pattern = _count_matches(actions, spec, default_n=0)
    if actual > expected_n:
        return [
            f"count_at_most FAILED — expected ≤ {expected_n} actions matching {pattern!r}, "
            f"got {actual}.\n  Full actions:{_format_action_list(actions)}"
        ]
    return []


def differs_from_baseline_by(
    actions: list[dict],
    spec: dict,
    *,
    baseline_actions: list[dict] | None = None,
    **_: object,
) -> list[str]:
    """Differential assertion: this variant's action list differs from the baseline.

    The first variant in a scenario IS the baseline; subsequent variants
    compare against it. Spec accepts:
      must_include: pattern        # appears in this variant, NOT in baseline
      must_not_include: pattern    # appears in baseline, NOT in this variant

    Both check directions are independent — supply either or both.
    """
    if baseline_actions is None:
        return [
            "differs_from_baseline_by FAILED — no baseline available. "
            "This assertion only works on a NON-FIRST variant within a scenario "
            "that has multiple variants. The first variant is treated as the baseline."
        ]
    if not isinstance(spec, dict):
        return [f"differs_from_baseline_by expected a dict, got {type(spec).__name__}"]

    failures: list[str] = []

    if "must_include" in spec:
        pattern = spec["must_include"]
        in_variant = any(matches(action, pattern) for action in actions)
        in_baseline = any(matches(action, pattern) for action in baseline_actions)
        if not in_variant or in_baseline:
            failures.append(
                f"differs_from_baseline_by.must_include FAILED — pattern {pattern!r} "
                f"should appear in this variant but NOT baseline. "
                f"Got: variant={in_variant}, baseline={in_baseline}."
            )

    if "must_not_include" in spec:
        pattern = spec["must_not_include"]
        in_variant = any(matches(action, pattern) for action in actions)
        in_baseline = any(matches(action, pattern) for action in baseline_actions)
        if in_variant or not in_baseline:
            failures.append(
                f"differs_from_baseline_by.must_not_include FAILED — pattern {pattern!r} "
                f"should appear in baseline but NOT this variant. "
                f"Got: variant={in_variant}, baseline={in_baseline}."
            )

    return failures


# ---------------------------------------------------------------------------
# Reasoning + memory assertions
# ---------------------------------------------------------------------------

_APPLIED_RE = re.compile(r"\[applied:\s*([^\]]+)\]", re.IGNORECASE)


def _extract_applied_titles(reasoning: str) -> list[str]:
    """Extract titles from any `[applied: t1, t2]` tag in reasoning.

    Was anchored to start-of-string (`re.match`), but the model often emits
    the tag inside a numbered list or after a header (e.g. `**Plan:**\n1. [applied: ...]`).
    Position is incidental — the contract is "model self-reports applied
    memories somewhere in its response" — so we search instead of match.
    Multiple `[applied: ...]` tags are unioned.
    """
    titles: list[str] = []
    for match in _APPLIED_RE.finditer(reasoning or ""):
        titles.extend(t.strip() for t in match.group(1).split(",") if t.strip())
    # dict.fromkeys preserves first-seen insertion order (Py 3.7+) and dedupes.
    return list(dict.fromkeys(titles))


def applied_memories(reasoning: str, expected: list[str], **_: object) -> list[str]:
    """Reasoning's `[applied: ...]` tag names exactly these titles (set-equal)."""
    actual = _extract_applied_titles(reasoning)
    if set(actual) != set(expected):
        return [
            f"applied_memories FAILED — expected {sorted(expected)!r}, "
            f"got {sorted(actual)!r}.\n  Reasoning: {_preview(reasoning)}"
        ]
    return []


def applied_memories_subset(reasoning: str, expected: list[str], **_: object) -> list[str]:
    """Reasoning's `[applied: ...]` tag names AT LEAST these titles (extras allowed)."""
    actual = set(_extract_applied_titles(reasoning))
    missing = set(expected) - actual
    if missing:
        return [
            f"applied_memories_subset FAILED — missing {sorted(missing)!r} from tag.\n"
            f"  Tagged: {sorted(actual)!r}.\n  Reasoning: {_preview(reasoning)}"
        ]
    return []


def reasoning_contains(reasoning: str, expected: str, **_: object) -> list[str]:
    """Reasoning string contains the substring (case-insensitive)."""
    if not isinstance(expected, str):
        return [f"reasoning_contains expected a string, got {type(expected).__name__}"]
    if expected.lower() not in (reasoning or "").lower():
        return [
            f"reasoning_contains FAILED — substring {expected!r} not found.\n"
            f"  Reasoning: {_preview(reasoning)}"
        ]
    return []


def reasoning_not_contains(reasoning: str, forbidden: str, **_: object) -> list[str]:
    """Reasoning must NOT contain a substring (case-insensitive)."""
    if not isinstance(forbidden, str):
        return [f"reasoning_not_contains expected a string, got {type(forbidden).__name__}"]
    if forbidden.lower() in (reasoning or "").lower():
        return [
            f"reasoning_not_contains FAILED — forbidden substring {forbidden!r} was found.\n"
            f"  Reasoning: {_preview(reasoning)}"
        ]
    return []
