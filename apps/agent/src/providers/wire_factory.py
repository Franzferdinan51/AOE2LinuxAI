"""The single point where an LLM wire is chosen by name."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

if TYPE_CHECKING:
    from ..config import WireName
    from .base import ChatWire
    from .text_wire import TextCompleter

ZEN_BASE_URL: Final = "https://opencode.ai/zen/v1"


def make_wire(
    name: WireName,
    model: str,
    api_key: str = "",
    base_url: str | None = None,
    max_retries: int = 3,
) -> ChatWire:
    match name:
        case "anthropic":
            from .wire_anthropic import AnthropicWire

            return AnthropicWire(model=model, api_key=api_key or None, max_retries=max_retries)

        case "openai" | "zen":
            from .wire_openai import OpenAIWire

            return OpenAIWire(
                model=model,
                api_key=api_key or None,
                base_url=_openai_endpoint(name, base_url),
                max_retries=max_retries,
            )

        case "lmstudio":
            from .wire_lmstudio import LMStudioWire

            return LMStudioWire(
                model=model,
                api_key=api_key or None,
                base_url=_openai_endpoint(name, base_url),
                max_retries=max_retries,
            )

        case _ as unreachable:
            assert_never(unreachable)


def make_text_completer(
    name: WireName,
    model: str,
    api_key: str = "",
    base_url: str | None = None,
) -> TextCompleter:
    match name:
        case "anthropic":
            from .text_wire import AnthropicTextCompleter

            return AnthropicTextCompleter(model=model, api_key=api_key or None)

        case "openai" | "zen":
            from .text_wire import OpenAITextCompleter

            return OpenAITextCompleter(
                model=model,
                api_key=api_key or None,
                base_url=_openai_endpoint(name, base_url),
            )

        case "lmstudio":
            from .text_wire import OpenAITextCompleter
            from .wire_lmstudio import LMSTUDIO_DEFAULT_BASE_URL

            return OpenAITextCompleter(
                model=model,
                api_key=api_key or "lm-studio",
                base_url=_openai_endpoint(name, base_url) or LMSTUDIO_DEFAULT_BASE_URL,
            )

        case _ as unreachable:
            assert_never(unreachable)


def _openai_endpoint(name: WireName, override: str | None) -> str | None:
    if name == "lmstudio":
        return override or None
    return override or (ZEN_BASE_URL if name == "zen" else None)


__all__ = ["ZEN_BASE_URL", "make_text_completer", "make_wire"]
