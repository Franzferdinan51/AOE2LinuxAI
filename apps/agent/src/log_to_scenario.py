"""Convert a saved game.txt log into a YAML scenario fixture.

Companion to ``fixture_builder.py`` — that CLI scrapes ONE turn; this module
walks a whole log and emits a fixture covering several turns. Same output
shape (snake_case scenario under apps/agent/src/scenarios/), but the input is
a multi-turn log and the output is hand-tunable.

Used by the autoresearch reflective loop: when a run fails its goal check,
this CLI turns the per-turn trajectory into a regression fixture the next
run can replay.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


_TURN_RE = re.compile(r"\[info\s*\]\s*iteration_start\s+iteration=(\d+)")
_RESOURCES_RE = re.compile(r"resources=(\{[^}]+\})")


def _parse_kv_line(line: str) -> dict:
    out: dict = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        out[key] = value
    return out


def _parse_resources(resources_str: str) -> dict:
    if not resources_str:
        return {}
    fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', resources_str)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return {}


def _latest_age(log_lines: list[str]) -> str:
    last = "Dark Age"
    for line in log_lines:
        for age in ("Imperial", "Castle", "Feudal", "Dark"):
            if age in line:
                last = f"{age} Age"
                break
    return last


def _latest_resources(log_lines: list[str]) -> dict:
    matches = [_RESOURCES_RE.search(line) for line in log_lines]
    matches = [m for m in matches if m]
    if not matches:
        return {}
    return _parse_resources(matches[-1].group(1))


def _class_counts(log_lines: list[str]) -> dict[str, int]:
    from collections import defaultdict

    counts: dict[str, int] = defaultdict(int)
    for line in log_lines:
        if "ownership_classified" not in line:
            continue
        kv = _parse_kv_line(line)
        cls = kv.get("class")
        if cls:
            counts[cls] += 1
    return dict(counts)


def build_fixture(
    log_path: Path,
    name: str,
    *,
    turn: int | None = None,
) -> str:
    """Build a YAML scenario fixture from a log file.

    If `turn` is provided, only that iteration's window is parsed. Otherwise
    the whole log is summarized (resources/age are taken at the end).
    """
    lines = log_path.read_text(encoding="utf-8").splitlines()

    if turn is not None:
        start = next(
            (
                i
                for i, line in enumerate(lines)
                if _TURN_RE.match(line) and int(_TURN_RE.match(line).group(1)) == turn
            ),
            None,
        )
        if start is None:
            raise ValueError(f"turn {turn} not found in log {log_path}")
        end = next(
            (i for i in range(start + 1, len(lines)) if _TURN_RE.match(lines[i])),
            len(lines),
        )
        window = lines[start:end]
    else:
        window = lines

    age = _latest_age(window)
    resources = _latest_resources(window)
    counts = _class_counts(window)
    return (
        f"name: {name}\n"
        f"description: 'Generated from {log_path.name}'\n"
        f"starting_state:\n"
        f"  age: {age}\n"
        f"  resources: {json.dumps(resources)}\n"
        f"  entities:\n"
    ) + "\n".join(f"    - {{class: {cls}, count: {n}}}" for cls, n in sorted(counts.items())) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--turn", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.log_path.is_file():
        print(f"log_to_scenario: log file not found: {args.log_path}", file=sys.stderr)
        return 1
    text = build_fixture(args.log_path, args.name, turn=args.turn)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
