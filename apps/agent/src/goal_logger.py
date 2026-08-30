"""Goal-history logger.

Appends one JSONL line per closed goal so regression tests can verify the
agent's strategic trajectory without re-running the LLM.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

_GOAL_LOG_ENV = "AOE2_GOAL_LOG"


def goal_log_path() -> Path | None:
    """Return the configured goal-log path, or None if logging is disabled.

    `AOE2_GOAL_LOG=/path/to/goals.jsonl` enables the logger; unset means no I/O.
    The parent directory is created lazily on first write.
    """
    raw = os.environ.get(_GOAL_LOG_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def log_goal_event(event: str, **fields: object) -> None:
    """Append a single JSONL line describing one goal event.

    `event` is one of ``"open"``, ``"close"``, ``"miss"``. Any additional fields
    (e.g. ``iteration``, ``goal_id``, ``description``) are serialized verbatim.
    """
    path = goal_log_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
