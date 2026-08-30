"""Screen-capture abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class ScreenCapturer(Protocol):
    def capture_screenshot(
        self, monitor: int = 1, quality: int | None = None
    ) -> tuple[bytes, int, int]:
        ...


@dataclass(frozen=True, slots=True)
class ScreenSize:
    width: int
    height: int
