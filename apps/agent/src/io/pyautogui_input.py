"""pyautogui InputInjector (Windows extra)."""

from __future__ import annotations


class PyautoguiInjector:
    def __init__(self, failsafe: bool = False, pause: float = 0.0) -> None:
        import pyautogui

        pyautogui.FAILSAFE = failsafe
        pyautogui.PAUSE = pause
        self._pyautogui = pyautogui

    def click(self, x: int, y: int) -> None:
        self._pyautogui.click(int(x), int(y))

    def right_click(self, x: int, y: int) -> None:
        self._pyautogui.rightClick(int(x), int(y))

    def move_to(self, x: int, y: int) -> None:
        self._pyautogui.moveTo(int(x), int(y))

    def press(self, key: str) -> None:
        self._pyautogui.press(key)

    def hotkey(self, modifiers: list[str], key: str) -> None:
        if modifiers:
            self._pyautogui.hotkey(*modifiers, key)
        else:
            self._pyautogui.press(key)

    def drag(self, start_xy: tuple[int, int], end_xy: tuple[int, int], duration: float) -> None:
        sx, sy = start_xy
        ex, ey = end_xy
        self._pyautogui.moveTo(int(sx), int(sy))
        self._pyautogui.drag(int(ex) - int(sx), int(ey) - int(sy), duration=duration)

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> None:
        if x is not None and y is not None:
            self._pyautogui.scroll(int(clicks), x=int(x), y=int(y))
        else:
            self._pyautogui.scroll(int(clicks))
