"""DuckDB sink + cold-path reader for the event log."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from core.event_log import (
    SCHEMA_VERSION,
    ActionPayload,
    ActionResultPayload,
    Event,
    EventRow,
    EventSink,
    ForkPayload,
    LlmPromptPayload,
    LlmResponsePayload,
    MetricPayload,
    NullEventSink,
    ObservationPayload,
    Payload,
    TurnStartPayload,
    WorldMutationPayload,
    WorldStateSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import duckdb
    from evaluation.event_broker import EventEnvelope


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    run_id           VARCHAR,
    agent_id         VARCHAR,
    t                INTEGER,
    kind             VARCHAR,
    payload_json     VARCHAR,
    ts               TIMESTAMP,
    schema_version   INTEGER
)
"""

_INSERT_SQL = """
INSERT INTO events (run_id, agent_id, t, kind, payload_json, ts, schema_version)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


class DuckDBEventSink:
    """Append events to a DuckDB connection."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._conn.execute(_CREATE_TABLE_SQL)

    def emit(self, event: Event) -> None:
        self._conn.execute(
            _INSERT_SQL,
            (
                event.run_id,
                event.agent_id,
                event.t,
                event.payload.kind,
                event.payload.model_dump_json(),
                event.ts,
                event.schema_version,
            ),
        )


def stream_cold(db_path, run_id):
    """Read finalized events for `run_id` from DuckDB, in canonical order."""
    import duckdb
    from evaluation.event_broker import EventEnvelope, RunId, Seq
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = cast("list[EventRow]", conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY t, rowid", [run_id]
        ).fetchall())
    typed_run = RunId(run_id)
    for i, row in enumerate(rows, start=1):
        yield EventEnvelope(run_id=typed_run, seq=Seq(i), event=Event.from_row(row))


__all__ = [
    "SCHEMA_VERSION", "ActionPayload", "ActionResultPayload",
    "DuckDBEventSink", "Event", "EventRow", "EventSink", "ForkPayload",
    "LlmPromptPayload", "LlmResponsePayload", "MetricPayload",
    "NullEventSink", "ObservationPayload", "Payload",
    "TurnStartPayload", "WorldMutationPayload", "WorldStateSnapshot",
    "stream_cold",
]
