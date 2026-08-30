"""What the three clocks hand each other: one frame, and the pipe it travels."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

from ..resource_ocr import ResourceReadings
from ..turn_timing import elapsed_ms


@dataclass(frozen=True, slots=True)
class Perception:
    """One frame, shared by reference across the three loops."""

    screenshot: bytes = b""
    width: int = 0
    height: int = 0
    entities: tuple[object, ...] = ()
    entity_summary: str = ""
    hud_readings: ResourceReadings = field(default_factory=ResourceReadings)
    alarm: bool = False
    tick: int = 0
    captured_at: float = field(default_factory=time.monotonic)

    @property
    def age_ms(self) -> float:
        return elapsed_ms(self.captured_at)


class FramePipe:
    """The channel from the perceive loop to the other two."""

    __slots__ = ("_arrived", "_frame", "_urgent")

    def __init__(self) -> None:
        self._frame: Perception | None = None
        self._arrived = asyncio.Event()
        self._urgent = asyncio.Event()

    def put(self, frame: Perception) -> None:
        self._frame = frame
        self._arrived.set()

    def latest(self) -> Perception | None:
        return self._frame

    async def after(self, captured_at: float) -> Perception:
        while True:
            self._arrived.clear()
            frame = self._frame
            if frame is not None and frame.captured_at > captured_at:
                return frame
            await self._arrived.wait()

    def request_now(self) -> None:
        self._urgent.set()

    async def wait_for_due(self, interval: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._urgent.wait(), timeout=interval)
        self._urgent.clear()


__all__ = ["FramePipe", "Perception"]
