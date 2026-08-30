"""LM Studio wire is registered and talks OpenAI-shaped Chat Completions."""

from __future__ import annotations

from gameplay_agent.config import WireName
from gameplay_agent.providers.wire_factory import make_wire
from gameplay_agent.providers.wire_lmstudio import LMSTUDIO_DEFAULT_BASE_URL, LMStudioWire


def test_lmstudio_is_registered_in_make_wire() -> None:
    wire = make_wire("lmstudio", model="qwen2.5-7b-instruct")
    assert type(wire).__name__ == "LMStudioWire"
    assert isinstance(wire, LMStudioWire)
    assert wire.model == "qwen2.5-7b-instruct"


def test_lmstudio_default_base_url_is_localhost_1234_v1() -> None:
    wire = make_wire("lmstudio", model="m", api_key="ignored")
    assert LMSTUDIO_DEFAULT_BASE_URL == "http://localhost:1234/v1"
    assert str(wire.client.base_url).rstrip("/") == "http://localhost:1234/v1"


def test_lmstudio_empty_api_key_is_substituted_with_dummy() -> None:
    wire = make_wire("lmstudio", model="m", api_key="")
    assert wire.client.api_key == "lm-studio"


def test_every_wire_implements_protocol_methods() -> None:
    names: list[WireName] = ["anthropic", "openai", "zen", "lmstudio"]
    for name in names:
        wire = make_wire(name, model="m", api_key="k")
        for method in ("tool_turn", "parse_structured", "is_api_error", "is_schema_too_large"):
            assert callable(getattr(wire, method))
