"""What the three clocks share, and only that."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from ..turn_timing import LatencyRecorder
from .snapshot import FramePipe

if TYPE_CHECKING:
    from ..goal_logger import GoalLogger
    from ..goals import GoalManager
    from ..memory import AgentMemory
    from .source import Actuator, FrameSource

log = structlog.stdlib.get_logger()


@dataclass(frozen=True, slots=True)
class LoopContext:
    """The act, perceive and deliberate loops, and what sits between them."""

    memory: AgentMemory
    goal_manager: GoalManager
    goal_logger: GoalLogger
    source: FrameSource
    actuator: Actuator

    frames: FramePipe = field(default_factory=FramePipe)
    latency: LatencyRecorder = field(default_factory=LatencyRecorder)
    input_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    time_budget: float | None = None
    max_iterations: int | None = None

    def request_stop(self, reason: str) -> None:
        if self.stop.is_set():
            log.debug("stop_already_requested", ignored=reason, reason=self.memory.game_end_reason)
            return
        self.memory.game_end_reason = reason
        self.stop.set()
        log.info("stop_requested", reason=reason)

    @property
    def stopping(self) -> bool:
        return self.stop.is_set()


__all__ = ["LoopContext"]
