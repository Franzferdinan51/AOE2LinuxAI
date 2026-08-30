"""Centralised configuration for the agent.

Built once at import time from environment variables; every other module
imports the ``config`` singleton rather than reading ``os.environ`` directly.
This makes overrides (tests, evaluation harness) a single mutation away.

The Linux port flips ``AOE2_INPUT_BACKEND`` through the values
``auto|x11|wayland|silent|pyautogui`` — see ``io/input.py`` for the dispatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .env_file import load_env_file

load_env_file()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    """All runtime configuration in one place.

    Defaults match the Linux + LM Studio reference run; the Windows port flips
    ``input_backend`` and the wire settings via env vars.
    """

    # ---- LLM wiring -----------------------------------------------------
    llm_wire: str = field(default_factory=lambda: _env_str("AOE2_LLM_WIRE", "lmstudio"))
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env_str("AOE2_MODEL", "qwen2.5-7b-instruct"))
    anthropic_api_key: str = field(default_factory=lambda: _env_str("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(
        default_factory=lambda: _env_str("AOE2_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    )
    lmstudio_host: str = field(
        default_factory=lambda: _env_str("AOE2_LMSTUDIO_HOST", "http://localhost:1234")
    )

    # ---- Agent behaviour -------------------------------------------------
    screenshot_quality: int = field(default_factory=lambda: _env_int("AOE2_SCREENSHOT_QUALITY", 70))
    # The interval (in turns) at which a full detection pass is forced.
    full_sahi_interval: int = field(default_factory=lambda: _env_int("AOE2_FULL_SAHI_INTERVAL", 5))
    adaptive_sahi: bool = field(default_factory=lambda: _env_bool("AOE2_ADAPTIVE_SAHI", True))
    save_screenshots: bool = field(default_factory=lambda: _env_bool("AOE2_SAVE_SCREENSHOTS", False))
    # Off-screen turn playback so regression hunts can run in CI.
    offline_mode: bool = field(default_factory=lambda: _env_bool("AOE2_OFFLINE", False))
    # Backend probe order for Linux input. `auto` tries xdotool, then ydotool,
    # then pyautogui; `silent` accepts input commands but never sends them.
    input_backend: str = field(default_factory=lambda: _env_str("AOE2_INPUT_BACKEND", "auto"))
    # xdotool: search window by substring; override for non-default titles.
    window_title_match: str = field(
        default_factory=lambda: _env_str("AOE2_WINDOW_TITLE_MATCH", "Age of Empires")
    )

    # ---- Detection -------------------------------------------------------
    detection_host: str = field(default_factory=lambda: _env_str("AOE2_DETECTION_HOST", ""))
    detection_model: str = field(
        default_factory=lambda: _env_str("AOE2_DETECTION_MODEL", "aoe2_yolo_v5")
    )
    detection_imgsz: int = field(default_factory=lambda: _env_int("AOE2_DETECTION_IMGSZ", 1280))
    rescan_cache: bool = field(default_factory=lambda: _env_bool("AOE2_RESCAN_CACHE", True))


config = Config()


def reload_config(**overrides: object) -> Config:
    """Return a fresh Config with `overrides` applied.

    Tests call this to pin ``input_backend`` and turn intervals without
    polluting environment variables.
    """
    from dataclasses import replace

    base = config
    return replace(base, **overrides)


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from `start` until a ``pyproject.toml`` is found.

    The agent runs from ``apps/agent/src``, so this finds the workspace root
    in three hops regardless of how the user invoked it.
    """
    base = start or Path(__file__).resolve().parent
    for directory in (base, *base.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return base


__all__ = ["Config", "config", "reload_config", "find_repo_root"]
