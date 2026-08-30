"""Redis Streams broker — Phase C of the log-first SSE architecture."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, cast

from evaluation.event_broker import (
    BrokerMetricsSnapshot, BrokerOverflowError, EventEnvelope,
    LiveRun, RunId, RunMeta, Seq,
)
from evaluation.event_log import Event, EventRow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from redis.asyncio import Redis
    from redis.typing import EncodableT, FieldT

_log = logging.getLogger(__name__)

_DEFAULT_MAX_STREAM_LEN: Final = 10_000
_DEFAULT_OPEN_TTL_SECONDS: Final = 60 * 60 * 6
_DEFAULT_XREAD_BLOCK_MS: Final = 100
_KEY_PREFIX_DEFAULT: Final = "arena"


@dataclass(frozen=True, slots=True)
class _Keys:
    events: bytes
    seq: bytes
    open: bytes


def _keys_for(prefix: str, run_id: RunId) -> _Keys:
    base = f"{prefix}:run:{run_id}"
    return _Keys(
        events=f"{base}:events".encode(),
        seq=f"{base}:seq".encode(),
        open=f"{base}:open".encode(),
    )


def _run_id_from_open_key(prefix: str, key: bytes) -> RunId:
    text = key.decode()
    head = f"{prefix}:run:"
    tail = ":open"
    return RunId(text[len(head) : -len(tail)])


def _encode_sentinel(meta: RunMeta | None) -> bytes:
    if meta is None:
        return b"1"
    return json.dumps({"label": meta.label, "started_at": meta.started_at}).encode()


def _decode_sentinel(raw: bytes | None) -> RunMeta:
    if raw is None:
        return RunMeta()
    try:
        data = cast("object", json.loads(raw))
    except (ValueError, TypeError):
        return RunMeta()
    if not isinstance(data, dict):
        return RunMeta()
    label = data.get("label")
    started_at = data.get("started_at")
    return RunMeta(
        label=label if isinstance(label, str) else None,
        started_at=started_at if isinstance(started_at, str) else None,
    )


_F_RUN_ID: Final = b"r"
_F_AGENT_ID: Final = b"a"
_F_T: Final = b"t"
_F_KIND: Final = b"k"
_F_PAYLOAD: Final = b"p"
_F_TS: Final = b"ts"
_F_SCHEMA_VERSION: Final = b"v"


def _event_to_fields(event: Event) -> dict[FieldT, EncodableT]:
    fields: dict[FieldT, EncodableT] = {
        _F_RUN_ID: event.run_id, _F_AGENT_ID: event.agent_id,
        _F_T: event.t, _F_KIND: event.payload.kind,
        _F_PAYLOAD: event.payload.model_dump_json(),
        _F_TS: event.ts.isoformat(), _F_SCHEMA_VERSION: event.schema_version,
    }
    return fields


def _fields_to_event(fields: dict[bytes, bytes]) -> Event:
    row: EventRow = (
        fields[_F_RUN_ID].decode(),
        fields[_F_AGENT_ID].decode(),
        int(fields[_F_T]),
        fields[_F_KIND].decode(),
        fields[_F_PAYLOAD].decode(),
        datetime.fromisoformat(fields[_F_TS].decode()),
        int(fields[_F_SCHEMA_VERSION]),
    )
    return Event.from_row(row)


def _seq_to_stream_id(seq: Seq) -> bytes:
    return f"{int(seq)}-0".encode()


def _stream_id_to_seq(stream_id: bytes | str) -> Seq:
    raw = stream_id.decode() if isinstance(stream_id, bytes) else stream_id
    return Seq(int(raw.split("-", 1)[0]))


class RedisStreamsBroker:
    """EventBroker impl backed by one Redis stream per run."""

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str = _KEY_PREFIX_DEFAULT,
        max_stream_len: int = _DEFAULT_MAX_STREAM_LEN,
        open_ttl_seconds: int = _DEFAULT_OPEN_TTL_SECONDS,
        xread_block_ms: int = _DEFAULT_XREAD_BLOCK_MS,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._max_stream_len = max_stream_len
        self._open_ttl_seconds = open_ttl_seconds
        self._xread_block_ms = xread_block_ms
        self._open_locally: set[RunId] = set()
        self._pending_admin: list[Callable[[], Awaitable[object]]] = []
        self._metrics_events_published = 0
        self._metrics_events_streamed = 0
        self._metrics_streams_dropped = 0

    def _enqueue_admin(self, build_coro: Callable[[], Awaitable[object]]) -> None:
        self._pending_admin.append(build_coro)

    async def flush(self) -> None:
        if not self._pending_admin:
            return
        ops = self._pending_admin
        self._pending_admin = []
        for build in ops:
            await build()

    def open_run(self, run_id: RunId, meta: RunMeta | None = None) -> None:
        if run_id in self._open_locally:
            raise ValueError(f"run {run_id!r} is already open")
        self._open_locally.add(run_id)
        keys = _keys_for(self._key_prefix, run_id)
        sentinel = _encode_sentinel(meta)
        self._enqueue_admin(lambda: self._client.set(keys.open, sentinel, ex=self._open_ttl_seconds))

    def close_run(self, run_id: RunId) -> None:
        self._open_locally.discard(run_id)
        keys = _keys_for(self._key_prefix, run_id)
        self._enqueue_admin(lambda: self._client.delete(keys.open))

    def reap(self, run_id: RunId) -> None:
        if run_id in self._open_locally:
            raise ValueError(f"cannot reap open run {run_id!r}; close it first")
        keys = _keys_for(self._key_prefix, run_id)
        self._enqueue_admin(lambda: self._client.delete(keys.events, keys.seq, keys.open))

    def is_open(self, run_id: RunId) -> bool:
        return run_id in self._open_locally

    async def is_open_remote(self, run_id: RunId) -> bool:
        await self.flush()
        keys = _keys_for(self._key_prefix, run_id)
        count = cast("int", await self._client.exists(keys.open))
        return count > 0

    async def live_runs(self) -> list[LiveRun]:
        await self.flush()
        pattern = f"{self._key_prefix}:run:*:open"
        runs: list[LiveRun] = []
        async for key in self._client.scan_iter(match=pattern):
            key_bytes = cast("bytes", key)
            run_id = _run_id_from_open_key(self._key_prefix, key_bytes)
            keys = _keys_for(self._key_prefix, run_id)
            raw = cast("bytes | None", await self._client.get(keys.open))
            if raw is None:
                continue
            meta = _decode_sentinel(raw)
            n_events = cast("int", await self._client.xlen(keys.events))
            runs.append(
                LiveRun(run_id=run_id, label=meta.label, started_at=meta.started_at, n_events=n_events)
            )
        return runs

    async def publish(self, run_id: RunId, event: Event) -> Seq:
        if run_id not in self._open_locally:
            raise RuntimeError(f"run {run_id!r} is not open")
        if event.run_id != run_id:
            raise ValueError(f"event.run_id {event.run_id!r} does not match broker run_id {run_id!r}")
        await self.flush()
        keys = _keys_for(self._key_prefix, run_id)
        seq_int = cast("int", await self._client.incr(keys.seq))
        seq = Seq(seq_int)
        await self._client.xadd(
            keys.events,
            _event_to_fields(event),
            id=_seq_to_stream_id(seq),
            maxlen=self._max_stream_len,
            approximate=True,
        )
        self._metrics_events_published += 1
        return seq

    async def stream(self, run_id: RunId, from_seq: Seq = Seq(0)) -> AsyncIterator[EventEnvelope]:
        try:
            keys = _keys_for(self._key_prefix, run_id)
            cursor = max(0, int(from_seq) - 1)
            while True:
                await self.flush()
                await self._check_overflow(keys, run_id, cursor)
                entries = cast(
                    "list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]] | None",
                    await self._client.xread(
                        streams={keys.events: _seq_to_stream_id(Seq(cursor))},
                        block=self._xread_block_ms,
                        count=None,
                    ),
                )
                yielded_any = False
                for envelope in self._envelopes_from_xread_reply(entries, run_id):
                    cursor = int(envelope.seq)
                    self._metrics_events_streamed += 1
                    yielded_any = True
                    yield envelope
                if yielded_any:
                    continue
                if not await self.is_open_remote(run_id):
                    return
        except (asyncio.CancelledError, GeneratorExit):
            await self._client.connection_pool.disconnect(inuse_connections=False)
            raise

    async def _check_overflow(self, keys: _Keys, run_id: RunId, cursor: int) -> None:
        try:
            info = cast("dict[bytes | str, object]", await self._client.xinfo_stream(keys.events))
        except Exception as exc:
            _log.debug("xinfo_stream(%s) failed: %s", run_id, exc)
            return
        first_entry = info.get("first-entry") or info.get(b"first-entry")
        if first_entry is None:
            return
        first_id = cast("tuple[bytes, object]", first_entry)[0]
        head_seq = int(_stream_id_to_seq(first_id))
        if cursor + 1 < head_seq:
            self._metrics_streams_dropped += 1
            raise BrokerOverflowError(
                run_id=run_id,
                requested_seq=Seq(cursor + 1),
                available_from=Seq(head_seq),
            )

    def _envelopes_from_xread_reply(self, entries, run_id):
        if not entries:
            return []
        _, items = entries[0]
        return [
            EventEnvelope(
                run_id=run_id,
                seq=_stream_id_to_seq(item_id),
                event=_fields_to_event(fields),
            )
            for item_id, fields in items
        ]

    async def metrics(self) -> BrokerMetricsSnapshot:
        return BrokerMetricsSnapshot(
            events_published=self._metrics_events_published,
            events_streamed=self._metrics_events_streamed,
            streams_dropped=self._metrics_streams_dropped,
            runs_open=len(self._open_locally),
        )


__all__ = ["RedisStreamsBroker"]
