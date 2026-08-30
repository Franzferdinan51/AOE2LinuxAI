"""DuckDB persister — one of N event-broker consumers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
from evaluation.event_broker import RunId, RunMeta, Seq
from evaluation.event_log import DuckDBEventSink

if TYPE_CHECKING:
    from pathlib import Path
    from evaluation.event_broker import EventBroker
    from evaluation.event_log import Event


async def persist_to_duckdb(broker, run_id, db_path):
    """Drain every published event for run_id into a DuckDB file."""
    with duckdb.connect(str(db_path)) as conn:
        sink = DuckDBEventSink(conn)
        async for envelope in broker.stream(run_id, from_seq=Seq(0)):
            sink.emit(envelope.event)


async def persist_to_duckdb_via_sink(broker, run_id, sink):
    """Drain run_id's broker stream into a caller-owned DuckDB sink."""
    async for envelope in broker.stream(run_id, from_seq=Seq(0)):
        sink.emit(envelope.event)


@dataclass(slots=True)
class MultiRunBrokerSink:
    """EventSink that routes per-run events onto a shared broker."""

    broker: EventBroker
    db_sink: DuckDBEventSink
    loop: asyncio.AbstractEventLoop
    label: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    _persisters: dict[RunId, asyncio.Task[None]] = field(default_factory=dict)
    _pending_publishes: set[asyncio.Task[Seq]] = field(default_factory=set)

    def emit(self, event: Event) -> None:
        self.loop.call_soon_threadsafe(self._handle_emit, event)

    def _handle_emit(self, event: Event) -> None:
        rid = RunId(event.run_id)
        if rid not in self._persisters:
            self.broker.open_run(rid, RunMeta(label=self.label, started_at=self.started_at))
            self._persisters[rid] = asyncio.create_task(
                persist_to_duckdb_via_sink(self.broker, rid, self.db_sink)
            )
        publish_task = asyncio.create_task(self.broker.publish(rid, event))
        self._pending_publishes.add(publish_task)
        publish_task.add_done_callback(self._pending_publishes.discard)

    async def close_all(self) -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if self._pending_publishes:
            await asyncio.gather(*self._pending_publishes)
        for rid in self._persisters:
            self.broker.close_run(rid)
        if self._persisters:
            await asyncio.gather(*self._persisters.values())
        for rid in self._persisters:
            self.broker.reap(rid)


__all__ = ["MultiRunBrokerSink", "persist_to_duckdb", "persist_to_duckdb_via_sink"]
