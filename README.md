# AOE2LinuxAI

Linux + LM Studio port of [dobosmarton/aoe2-agent](https://github.com/dobosmarton/aoe2-agent).

An LLM agent that plays Age of Empires II: Definitive Edition. Perception is YOLO + local OCR (text-only LLMs). Input on Linux is `xdotool` (X11) or `ydotool` (Wayland). Local models run through LM Studio.

## Quick start

```bash
# system tools
sudo pacman -S xdotool ydotool   # Arch; Debian: apt install xdotool

# Python 3.11+
uv sync
cp .env.example .env

# LM Studio: start the local server, then:
export AOE2_LLM_WIRE=lmstudio
export AOE2_MODEL=qwen2.5-7b-instruct   # whatever lms ls shows
just agent-lmstudio
```

## Layout

- `apps/agent/src/io/` — `InputInjector` protocol, X11/Wayland/silent/pyautogui backends
- `apps/agent/src/providers/wire_lmstudio.py` — OpenAI-compatible LM Studio wire (`http://localhost:1234/v1`)
- `tests/test_lmstudio_wire.py`, `tests/test_input_backend_linux.py` — Linux/LM Studio gates

Branch `linux-lmstudio` is the working branch. Full upstream monorepo (YOLO weights, arena stack) lives locally at `~/Work/AOE2LinuxAI` and will keep landing here in follow-up commits.

Upstream is MIT; this fork is MIT.
