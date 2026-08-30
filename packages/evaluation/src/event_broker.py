"""Event broker — live pub/sub source for the SSE architecture."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, NewType, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from evaluation.event_log import Event

RunId = NewType("RunId", str)
Seq = NewType("Seq", int)


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    run_id: RunId
    seq: Seq
    event: Event


@dataclass(frozen=True, slots=True)
class RunMeta:
    label: str | None = None
    started_at: str | None = None


@dataclass(frozen=True, slots=True)
class LiveRun:
    run_id: RunId
    label: str | None
    started_at: str | None
    n_events: int


@dataclass(frozen=True, slots=True)
class BrokerOverflowError(Exception):
    run_id: RunId
    requested_seq: Seq
    available_from: Seq


@dataclass(frozen=True, slots=True)
class BrokerMetricsSnapshot:
    events_published: int
    events_streamed: int
    streams_dropped: int
    runs_open: int

    def to_dict(self) -> dict[str, int]:
        return {
            "events_published": self.events_published,
            "events_streamed": self.events_streamed,
            "streams_dropped": self.streams_dropped,
            "runs_open": self.runs_open,
        }


class EventBroker(Protocol):
    """Pub/sub with replay-from-offset semantics."""

    def open_run(self, run_id: RunId, meta: RunMeta | None = None) -> None: ...
    def close_run(self, run_id: RunId) -> None: ...
    def reap(self, run_id: RunId) -> None: ...
    def is_open(self, run_id: RunId) -> bool: ...
    async def is_open_remote(self, run_id: RunId) -> bool: ...
    async def live_runs(self) -> Sequence[LiveRun]: ...
    async def publish(self, run_id: RunId, event: Event) -> Seq: ...
    def stream(self, run_id: RunId, from_seq: Seq = Seq(0)) -> AsyncIterator[EventEnvelope]: ...


_DEFAULT_MAX_BUFFER_SIZE = 10_000


class InProcessEventBroker:
    """Single-process, asyncio-native broker."""

    def __init__(self, max_buffer_size: int = _DEFAULT_MAX_BUFFER_SIZE) -> None:
        self._buffers: dict[RunId, deque[EventEnvelope]] = defaultdict(
            lambda: deque(maxlen=max_buffer_size)
        )
        self._head_seq: dict[RunId, int] = defaultdict(lambda: 1)
        self._seq: dict[RunId, count[int]] = defaultdict(lambda: count(1))
        self._open: set[RunId] = set()
        self._waiters: dict[RunId, list[asyncio.Event]] = defaultdict(list)
        self._closed: dict[RunId, asyncio.Event] = defaultdict(asyncio.Event)
        self._max_buffer_size = max_buffer_size
        self._metrics_events_published = 0
        self._metrics_events_streamed = 0
        self._metrics_streams_dropped = 0
        self._lock = asyncio.Lock()

    def open_run(self, run_id: RunId, meta: RunMeta | None = None) -> None:
        if run_id in self._open:
            raise RuntimeError(f"run {run_id!r} already open")
        self._open.add(run_id)
        self._closed[run_id].clear()

    def close_run(self, run_id: RunId) -> None:
        self._open.discard(run_id)
        closed_event = self._closed[run_id]
        if not closed_event.is_set():
            closed_event.set()

    def reap(self, run_id: RunId) -> None:
        if run_id in self._open:
            raise ValueError(f"cannot reap open run {run_id!r}")
        self._buffers.pop(run_id, None)
        self._head_seq.pop(run_id, None)
        self._seq.pop(run_id, None)
        self._waiters.pop(run_id, None)
        self._closed.pop(run_id, None)

    def is_open(self, run_id: RunId) -> bool:
        return run_id in self._open

    async def is_open_remote(self, run_id: RunId) -> bool:
        return self.is_open(run_id)

    async def live_runs(self) -> Sequence[LiveRun]:
        return tuple(
            LiveRun(
                run_id=rid,
                label=None,
                started_at=None,
                n_events=next(self._seq[rid]) - 1,
            )
            for rid in sorted(self._open)
        )

    async def publish(self, run_id: RunId, event: Event) -> Seq:
        if run_id not in self._open:
            raise RuntimeError(f"cannot publish to closed run {run_id!r}")
        async with self._lock:
            seq = Seq(next(self._seq[run_id]))
            envelope = EventEnvelope(run_id=run_id, seq=seq, event=event)
            buffer = self._buffers[run_id]
            if len(buffer) == self._max_buffer_size:
                self._head_seq[run_id] += 1
                self._metrics_streams_dropped += 1
            buffer.append(envelope)
            self._metrics_events_published += 1
            for waiter in self._waiters[run_id]:
                waiter.set()
            self._waiters[run_id] = []
        return seq

    def stream(self, run_id: RunId, from_seq: Seq = Seq(0)) -> AsyncIterator[EventEnvelope]:
        broker = self

        async def _iter():
            next_seq = from_seq + 1 if from_seq > 0 else 1
            head = broker._head_seq.get(run_id, 1)
            if next_seq < head:
                raise BrokerOverflowError(run_id=run_id, requested_seq=Seq(next_seq), available_from=Seq(head))
            emitted: set[Seq] = set()
            while True:
                if run_id in broker._buffers:
                    idx = next_seq - head
                    buf = broker._buffers[run_id]
                    if 0 <= idx < len(buf):
                        env = buf[idx]
                        broker._metrics_events_streamed += 1
                        yield env
                        next_seq = int(env.seq) + 1
                        emitted.add(env.seq)
                        continue
                if run_id not in broker._open:
                    while True:
                        if run_id in broker._buffers:
                            idx = next_seq - head
                            buf = broker._buffers[run_id]
                            if 0 <= idx < len(buf):
                                env = buf[idx]
                                broker._metrics_events_streamed += 1
                                yield env
                                next_seq = int(env.seq) + 1
                                continue
                        break
                    return
                waiter = asyncio.Event()
                broker._waiters[run_id].append(waiter)
                try:
                    await asyncio.wait_for(waiter.wait(), timeout=0.5)
                except TimeoutError:
                    broker._waiters[run_id].remove(waiter)

        return _iter()

    def metrics(self) -> BrokerMetricsSnapshot:
        return BrokerMetricsSnapshot(
            events_published=self._metrics_events_published,
            events_streamed=self._metrics_events_streamed,
            streams_dropped=self._metrics_streams_dropped,
            runs_open=len(self._open),
        )


@dataclass(frozen=True, slots=True)
class BrokerEventSink:
    """EventSink adapter that publishes onto a broker's event loop."""

    broker: EventBroker
    run_id: RunId
    loop: asyncio.AbstractEventLoop

    def emit(self, event: Event) -> None:
        self.loop.call_soon_threadsafe(asyncio.create_task, self.broker.publish(self.run_id, event))


__all__ = [
    "BrokerEventSink", "BrokerMetricsSnapshot", "BrokerOverflowError",
    "EventBroker", "EventEnvelope", "InProcessEventBroker",
    "LiveRun", "RunId", "RunMeta", "Seq",
]
