"""Convert a saved game.txt log + screenshots/ into a YAML scenario fixture.

For a given turn N from a real game, this CLI scrapes:
  - resources/age from the LATEST strategist_response in the log (the
    strategist runs every ~10 turns, so this is approximate)
  - detected entity classes from ownership_classified lines within the turn

Outputs a fixture under gameplay_agent/scenarios/regression/<name>.yaml. Goals,
real entity coordinates, and assertions still need hand-editing — this CLI
is a starting-point template, not a full export.

Usage:
    python -m gameplay_agent.fixture_builder logs/2026_04_25/game.txt --turn 14
    python -m gameplay_agent.fixture_builder logs/2026_04_25/game.txt --turn 14 \
        --out gameplay_agent/scenarios/regression/exp_0013_turn_14.yaml \
        --name regression_exp_0013_turn_14_housing_stall
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_AGE = "Dark Age"
DEFAULT_POPULATION = "0/0"
PLACEHOLDER_BASE_X = 1500
PLACEHOLDER_BASE_Y = 800
PLACEHOLDER_X_STEP_PER_CLASS = 80
PLACEHOLDER_X_STEP_PER_INSTANCE = 20
PLACEHOLDER_Y_STEP_PER_CLASS = 60
PLACEHOLDER_CONFIDENCE = 0.85

_TURN_RE = re.compile(r"\[info\s*\]\s*iteration_start\s+iteration=(\d+)")
_RESOURCES_RE = re.compile(r"resources=(\{[^}]+\})")


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------


def _parse_kv_line(line: str) -> dict:
    """Parse a single-line 'k1=v1 k2=v2' log line into a dict.

    Values are kept as strings — callers do their own type coercion.  This is
    the lowest-common-denominator: every log line in this repo uses this shape.
    """
    out: dict = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        out[key] = value
    return out


def _parse_resources(resources_str: str) -> dict:
    """Convert a Python-dict-looking resource string (e.g. ``"{food: 200, wood: 0}"``) to JSON.

    Game logs emit Python reprs (``{food: 200, wood: 0}``) which is not valid JSON. Wrap keys
    in quotes and substitute ``True``/``False``/``None`` for JSON compatibility, then
    `json.loads`. Empty input returns an empty dict.
    """
    if not resources_str:
        return {}
    # Quote bare keys (foo:) and (foo : ) to JSON-style keys
    fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', resources_str)
    # Replace Python literals with JSON literals
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return {}


def _scan_turn_window(log_lines: list[str], target_turn: int) -> tuple[list[str], int]:
    """Return lines strictly inside the iteration_start block for `target_turn`.

    Returns the slice of lines belonging to that turn and the start index of the
    next turn (or len(log_lines) if this is the last).
    """
    n = len(log_lines)
    start = next(
        (i for i, line in enumerate(log_lines) if _TURN_RE.match(line) and int(_TURN_RE.match(line).group(1)) == target_turn),
        None,
    )
    if start is None:
        return [], n
    end = next(
        (i for i in range(start + 1, n) if _TURN_RE.match(log_lines[i])),
        n,
    )
    return log_lines[start:end], end


# ---------------------------------------------------------------------------
# Field extraction from a turn window
# ---------------------------------------------------------------------------


def _latest_age(log_lines: list[str]) -> str:
    """Extract the latest age mentioned in the turn window."""
    last = DEFAULT_AGE
    for line in log_lines:
        for age in ("Imperial", "Castle", "Feudal", "Dark"):
            if age in line:
                last = f"{age} Age"
                break
    return last


def _latest_resources(log_lines: list[str]) -> dict:
    """The last ``resources={...}`` field seen in this turn window.

    Resources are reported on every iteration_start line and after every strategist
    response — the LAST occurrence is the freshest snapshot for the target turn.
    """
    matches = [_RESOURCES_RE.search(line) for line in log_lines]
    matches = [m for m in matches if m]
    if not matches:
        return {}
    return _parse_resources(matches[-1].group(1))


def _latest_turn_phase(log_lines: list[str]) -> str:
    """The latest ``turn_phase=`` value seen in the turn window."""
    last = ""
    for line in log_lines:
        kv = _parse_kv_line(line)
        if "turn_phase" in kv:
            last = kv["turn_phase"]
    return last or "unknown"


def _entity_classes_in_turn(log_lines: list[str]) -> dict[str, int]:
    """Count detected entity classes within the turn window.

    Walks `ownership_classified` lines (one per entity) and tallies by class. Used
    to seed fixture entity lists with what the detector actually saw.
    """
    counts: dict[str, int] = defaultdict(int)
    for line in log_lines:
        if "ownership_classified" not in line:
            continue
        kv = _parse_kv_line(line)
        cls = kv.get("class")
        if cls:
            counts[cls] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


def _yaml_escape(value: str) -> str:
    """Quote a string for YAML — single-quoted, with embedded ``'`` doubled."""
    return "'" + value.replace("'", "''") + "'"


def _format_entity_list(class_counts: dict[str, int]) -> str:
    """One entity per line, class-indexed, with placeholder coords."""
    out: list[str] = []
    for class_index, (cls, count) in enumerate(sorted(class_counts.items())):
        for instance in range(count):
            x = (
                PLACEHOLDER_BASE_X
                + class_index * PLACEHOLDER_X_STEP_PER_CLASS
                + instance * PLACEHOLDER_X_STEP_PER_INSTANCE
            )
            y = PLACEHOLDER_BASE_Y + class_index * PLACEHOLDER_Y_STEP_PER_CLASS
            out.append(
                f"  - {{id: e_{cls}_{instance}, class: {_yaml_escape(cls)}, "
                f"center: [{x}, {y}], confidence: {PLACEHOLDER_CONFIDENCE}}}"
            )
    return "\n".join(out) if out else "  []"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_path", type=Path, help="Path to a saved game.txt log")
    parser.add_argument("--turn", type=int, required=True, help="Which iteration/turn to snapshot")
    parser.add_argument("--out", type=Path, default=None, help="Output YAML path (default: stdout)")
    parser.add_argument("--name", type=str, default=None, help="Scenario name (defaults to <log>_<turn>)")
    args = parser.parse_args(argv)

    if not args.log_path.is_file():
        print(f"fixture_builder: log file not found: {args.log_path}", file=sys.stderr)
        return 1

    log_lines = args.log_path.read_text(encoding="utf-8").splitlines()
    window, _ = _scan_turn_window(log_lines, args.turn)

    if not window:
        print(
            f"fixture_builder: turn {args.turn} not found in log (only "
            f"{sum(1 for line in log_lines if _TURN_RE.match(line))} turns present)",
            file=sys.stderr,
        )
        return 1

    age = _latest_age(window)
    resources = _latest_resources(window)
    turn_phase = _latest_turn_phase(window)
    class_counts = _entity_classes_in_turn(window)
    name = args.name or f"{args.log_path.stem}_turn_{args.turn}"

    entity_block = _format_entity_list(class_counts)
    resources_yaml = json.dumps(resources) if resources else "{}"
    scenario_yaml = (
        f"name: {_yaml_escape(name)}\n"
        f"description: {_yaml_escape(f'Auto-generated from {args.log_path} turn {args.turn}')}\n"
        f"starting_state:\n"
        f"  age: {_yaml_escape(age)}\n"
        f"  resources: {resources_yaml}\n"
        f"  population: {_yaml_escape(DEFAULT_POPULATION)}\n"
        f"  turn: {args.turn}\n"
        f"  turn_phase: {_yaml_escape(turn_phase)}\n"
        f"  entities:\n"
        f"{entity_block}\n"
        f"goals:\n"
        f"  # TODO: hand-edit goal list. This CLI only seeds starting_state + entity counts.\n"
        f"  []\n"
        f"assertions:\n"
        f"  # TODO: hand-edit assertions (must_include / count_at_least / reasoning_contains).\n"
        f"  []\n"
        f"variants: []\n"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(scenario_yaml, encoding="utf-8")
        print(f"Wrote fixture: {args.out}")
    else:
        sys.stdout.write(scenario_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
