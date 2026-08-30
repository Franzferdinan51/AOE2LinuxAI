"""Linux-native input-injection backends (xdotool / ydotool / silent)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Final

from .input import InputInjector

DEFAULT_AUTO_PREFERENCE: Final[tuple[str, ...]] = ("wayland", "x11")


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        argv,
        check=False,
        env=env,
        capture_output=True,
        start_new_session=True,
    )


class X11InputInjector:
    def __init__(self, xdotool_path: str | None = None) -> None:
        self._xdotool = xdotool_path or _which("xdotool") or "xdotool"
        self._env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}

    def click(self, x: int, y: int) -> None:
        _run([self._xdotool, "mousemove", "--sync", str(int(x)), str(int(y))], env=self._env)
        _run([self._xdotool, "click", "--clearmodifiers", "1"], env=self._env)

    def right_click(self, x: int, y: int) -> None:
        _run([self._xdotool, "mousemove", "--sync", str(int(x)), str(int(y))], env=self._env)
        _run([self._xdotool, "click", "--clearmodifiers", "3"], env=self._env)

    def move_to(self, x: int, y: int) -> None:
        _run([self._xdotool, "mousemove", "--sync", str(int(x)), str(int(y))], env=self._env)

    def press(self, key: str) -> None:
        _run([self._xdotool, "key", "--clearmodifiers", key], env=self._env)

    def hotkey(self, modifiers: list[str], key: str) -> None:
        if not modifiers:
            self.press(key)
            return
        chord = "+".join([*modifiers, key])
        _run([self._xdotool, "key", "--clearmodifiers", chord], env=self._env)

    def drag(self, start_xy: tuple[int, int], end_xy: tuple[int, int], duration: float) -> None:
        sx, sy = start_xy
        ex, ey = end_xy
        _run([self._xdotool, "mousemove", "--sync", str(int(sx)), str(int(sy))], env=self._env)
        steps = max(int(duration * 60), 1)
        for step in range(1, steps + 1):
            ratio = step / steps
            ix = int(int(sx) + (int(ex) - int(sx)) * ratio)
            iy = int(int(sy) + (int(ey) - int(sy)) * ratio)
            _run([self._xdotool, "mousemove", "--sync", str(ix), str(iy)], env=self._env)
            time.sleep(duration / steps if duration > 0 else 0)
        _run([self._xdotool, "mousedown", "--clearmodifiers", "1"], env=self._env)
        time.sleep(0.05)
        _run([self._xdotool, "mouseup", "--clearmodifiers", "1"], env=self._env)

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> None:
        argv = [self._xdotool]
        if x is not None and y is not None:
            _run([self._xdotool, "mousemove", "--sync", str(int(x)), str(int(y))], env=self._env)
            argv.extend(["click", "--clearmodifiers", "--repeat", str(abs(int(clicks)))])
        else:
            argv.extend(["click", "--clearmodifiers", "--repeat", str(abs(int(clicks)))])
        direction = "5" if clicks < 0 else "4"
        argv.extend([direction])
        _run(argv, env=self._env)


class WaylandInputInjector:
    def __init__(self, ydotool_path: str | None = None) -> None:
        self._ydotool = ydotool_path or _which("ydotool") or "ydotool"

    def _move(self, x: int, y: int) -> None:
        _run([self._ydotool, "mousemove", "--absolute", "--", str(int(x)), str(int(y))])

    def _click_button(self, code: int) -> None:
        _run([self._ydotool, "click", str(code)])

    def click(self, x: int, y: int) -> None:
        self._move(x, y)
        self._click_button(0xC0)

    def right_click(self, x: int, y: int) -> None:
        self._move(x, y)
        self._click_button(0xC2)

    def move_to(self, x: int, y: int) -> None:
        self._move(x, y)

    def press(self, key: str) -> None:
        _run([self._ydotool, "key", key])

    def hotkey(self, modifiers: list[str], key: str) -> None:
        if not modifiers:
            self.press(key)
            return
        _run([self._ydotool, "key", *(modifiers), key])

    def drag(self, start_xy: tuple[int, int], end_xy: tuple[int, int], duration: float) -> None:
        sx, sy = start_xy
        ex, ey = end_xy
        self._move(sx, sy)
        steps = max(int(duration * 60), 1)
        for step in range(1, steps + 1):
            ratio = step / steps
            ix = int(int(sx) + (int(ex) - int(sx)) * ratio)
            iy = int(int(sy) + (int(ey) - int(sy)) * ratio)
            self._move(ix, iy)
            time.sleep(duration / steps if duration > 0 else 0)
        _run([self._ydotool, "mousedown", "--", "0x40"])
        time.sleep(0.05)
        self._move(int(ex), int(ey))
        _run([self._ydotool, "mouseup", "--", "0x80"])

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> None:
        if x is not None and y is not None:
            self._move(x, y)
        direction = "down" if clicks < 0 else "up"
        for _ in range(abs(int(clicks))):
            _run([self._ydotool, "wheel", direction])


class SilentInputInjector:
    def __init__(self, log_path: str | None = None) -> None:
        import threading

        self._lock = threading.Lock()
        self.records: list[dict[str, object]] = []
        self._log_path = log_path

    def _record(self, method: str, **args: object) -> None:
        import json

        entry = {"method": method, **args}
        with self._lock:
            self.records.append(entry)
        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry) + "\n")

    def click(self, x: int, y: int) -> None:
        self._record("click", x=x, y=y)

    def right_click(self, x: int, y: int) -> None:
        self._record("right_click", x=x, y=y)

    def move_to(self, x: int, y: int) -> None:
        self._record("move_to", x=x, y=y)

    def press(self, key: str) -> None:
        self._record("press", key=key)

    def hotkey(self, modifiers: list[str], key: str) -> None:
        self._record("hotkey", modifiers=modifiers, key=key)

    def drag(self, start_xy: tuple[int, int], end_xy: tuple[int, int], duration: float) -> None:
        self._record("drag", start=list(start_xy), end=list(end_xy), duration=duration)

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> None:
        self._record("scroll", clicks=clicks, x=x, y=y)


def select_input_backend(
    *,
    override: str | None = None,
    xdotool_path: str | None = None,
    ydotool_path: str | None = None,
) -> InputInjector:
    choice = (override or os.environ.get("AOE2_INPUT_BACKEND") or "auto").strip().lower()
    if choice == "silent":
        return SilentInputInjector()
    if choice == "pyautogui":
        from .pyautogui_input import PyautoguiInjector

        return PyautoguiInjector(failsafe=False, pause=0.02)
    if choice == "x11":
        return X11InputInjector(xdotool_path=xdotool_path)
    if choice == "wayland":
        return WaylandInputInjector(ydotool_path=ydotool_path)
    session = (os.environ.get("XDG_SESSION_TYPE") or "").strip().lower()
    if session == "wayland":
        if _which(ydotool_path or "ydotool") is not None:
            return WaylandInputInjector(ydotool_path=ydotool_path)
        if _which(xdotool_path or "xdotool") is not None and os.environ.get("DISPLAY"):
            return X11InputInjector(xdotool_path=xdotool_path)
    elif session in ("x11", ""):
        if _which(xdotool_path or "xdotool") is not None and os.environ.get("DISPLAY"):
            return X11InputInjector(xdotool_path=xdotool_path)
        if _which(ydotool_path or "ydotool") is not None:
            return WaylandInputInjector(ydotool_path=ydotool_path)
    return SilentInputInjector()


__all__ = [
    "SilentInputInjector",
    "WaylandInputInjector",
    "X11InputInjector",
    "select_input_backend",
]
