"""Robust JSON extraction from LLM responses."""

from __future__ import annotations

import json
import re
from typing import cast

import structlog

log = structlog.stdlib.get_logger()


def _loads(text: str) -> object:
    return cast("object", json.loads(text))


def extract_json_object(text: str) -> dict[str, object] | None:
    """Extract a JSON object from LLM response text.

    Tries three strategies in order:
    1. Direct parse of the entire text
    2. Regex match of a ```json``` code block
    3. Bracket matching with string-escape handling

    Returns the parsed dict, or None if extraction fails.
    """
    try:
        result = _loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    code_match = re.search(r"```(?:json)?\s*(\{.+\})\s*```", text, re.DOTALL)
    if code_match:
        try:
            result = _loads(code_match.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return _extract_by_bracket_matching(text)


def _try_loads(text: str) -> object | None:
    try:
        return _loads(text)
    except json.JSONDecodeError:
        return None


def _as_dict_list(parsed: object) -> list[dict[str, object]]:
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def extract_json_array(text: str) -> list[dict[str, object]]:
    """Extract a JSON array of objects from LLM response text."""
    parsed = _try_loads(text)
    if parsed is None:
        code_match = re.search(r"```(?:json)?\s*(\[.+\]|\{.+\})\s*```", text, re.DOTALL)
        if code_match:
            parsed = _try_loads(code_match.group(1))
    if parsed is None:
        parsed = _extract_array_by_bracket_matching(text)
    return _as_dict_list(parsed)


def _extract_array_by_bracket_matching(text: str) -> object | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return _try_loads(text[start : i + 1])
    return None


def _extract_by_bracket_matching(text: str) -> dict[str, object] | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = _loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


__all__ = ["extract_json_array", "extract_json_object"]
