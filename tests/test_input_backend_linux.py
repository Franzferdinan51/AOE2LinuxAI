"""Linux input backend selection."""

from __future__ import annotations

import pytest

from gameplay_agent.io import (
    InputInjector,
    SilentInputInjector,
    WaylandInputInjector,
    X11InputInjector,
    select_input_backend,
)


def test_select_input_backend_returns_silent_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOE2_INPUT_BACKEND", "silent")
    backend = select_input_backend()
    assert isinstance(backend, SilentInputInjector)
    assert isinstance(backend, InputInjector)


def test_select_input_backend_returns_x11_explicit() -> None:
    backend = select_input_backend(override="x11")
    assert isinstance(backend, X11InputInjector)


def test_select_input_backend_returns_wayland_explicit() -> None:
    backend = select_input_backend(override="wayland")
    assert isinstance(backend, WaylandInputInjector)


def test_silent_injector_records_clicks() -> None:
    backend = SilentInputInjector()
    backend.click(1, 2)
    backend.press("h")
    assert backend.records[0]["method"] == "click"
    assert backend.records[1]["method"] == "press"
