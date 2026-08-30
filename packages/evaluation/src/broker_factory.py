"""Broker selection — the single point where `ARENA_BROKER_BACKEND` is read.

`make_broker()` decides between the in-process broker (default, no
external deps) and `RedisStreamsBroker` (Phase C, cross-process). Every
callsite that needs a broker constructs it through this factory so the
env-var read happens in exactly one place.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

from evaluation.event_broker import InProcessEventBroker

if TYPE_CHECKING:
    from evaluation.event_broker import EventBroker


_BACKEND_ENV: Final = "ARENA_BROKER_BACKEND"
_REDIS_URL_ENV: Final = "REDIS_URL"
_REDIS_PASSWORD_ENV: Final = "REDIS_PASSWORD"


def _resolved_redis_url() -> str:
    explicit = os.environ.get(_REDIS_URL_ENV)
    if explicit:
        return explicit
    password = os.environ.get(_REDIS_PASSWORD_ENV)
    if password:
        return f"redis://:{password}@localhost:6379/0"
    return "redis://localhost:6379/0"


def make_broker() -> EventBroker:
    backend = os.environ.get(_BACKEND_ENV, "inprocess").strip().lower()
    if backend == "inprocess":
        return InProcessEventBroker()
    if backend == "redis":
        from evaluation.redis_broker import RedisStreamsBroker
        from redis.asyncio import Redis
        return RedisStreamsBroker(Redis.from_url(_resolved_redis_url()))
    raise ValueError(f"unknown {_BACKEND_ENV}={backend!r}; expected 'inprocess' or 'redis'")


__all__ = ["make_broker"]
