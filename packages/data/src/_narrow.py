"""Helpers for narrowing untyped (`object`) values to concrete python types."""

from __future__ import annotations


def as_int(value: object, default: int = 0) -> int:
    """Coerce a JSON-derived value to int, falling back to `default`."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def as_str(value: object, default: str = "") -> str:
    """Coerce a JSON-derived value to str, falling back to `default`."""
    if isinstance(value, str):
        return value
    return default if value is None else str(value)


def as_dict(value: object) -> dict[str, object]:
    """Return `value` if it's a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}
