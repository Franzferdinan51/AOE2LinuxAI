"""Anthropic Messages API implementation of `ChatWire`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import anthropic

from .base import (
    AssistantTurn, ChatRequest, ModelRefusedError, SystemBlock, TokenUsage,
    ToolCall, ToolOutcome, ToolResultsTurn, ToolTurnResult, UserTurn, tool_outcome_json,
)

if TYPE_CHECKING:
    from anthropic.types import OutputConfigParam, Usage
    from .base import ModelT, Turn


def _ephemeral() -> dict[str, str]:
    return {"type": "ephemeral"}


def _temperature(value: float | None) -> float | anthropic.NotGiven:
    return anthropic.NOT_GIVEN if value is None else value


class _HasUsage(Protocol):
    usage: Usage


class AnthropicWire:
    """`ChatWire` over `anthropic.AsyncAnthropic`."""

    def __init__(self, model: str, api_key: str | None = None, max_retries: int = 3) -> None:
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self.endpoint = str(self.client.base_url)

    @staticmethod
    def _render_system(blocks: tuple[SystemBlock, ...]) -> list[dict[str, object]]:
        rendered: list[dict[str, object]] = []
        for block in blocks:
            entry: dict[str, object] = {"type": "text", "text": block.text}
            if block.cacheable:
                entry["cache_control"] = _ephemeral()
            rendered.append(entry)
        return rendered

    @staticmethod
    def _render_outcome(outcome: ToolOutcome) -> dict[str, object]:
        return {
            "type": "tool_result",
            "tool_use_id": outcome.tool_call_id,
            "content": tool_outcome_json(outcome),
        }

    @classmethod
    def _render_turns(cls, turns: tuple[Turn, ...]) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        for turn in turns:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": [{"type": "text", "text": turn.text}]})
            elif isinstance(turn, AssistantTurn):
                blocks: list[dict[str, object]] = []
                if turn.text:
                    blocks.append({"type": "text", "text": turn.text})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in turn.tool_calls
                )
                messages.append({"role": "assistant", "content": blocks})
            elif isinstance(turn, ToolResultsTurn):
                results = [cls._render_outcome(outcome) for outcome in turn.outcomes]
                messages.append({"role": "user", "content": results})
        return cls._with_moving_cache_breakpoint(messages)

    @staticmethod
    def _with_moving_cache_breakpoint(messages: list[dict[str, object]]) -> list[dict[str, object]]:
        if not messages:
            return messages
        last = messages[-1]
        content = last["content"]
        if not isinstance(content, list) or not content:
            return messages
        tail = cast("object", content[-1])
        if not isinstance(tail, dict):
            return messages
        marked = cast("dict[str, object]", {**tail, "cache_control": _ephemeral()})
        head = cast("list[object]", content[:-1])
        return [*messages[:-1], {**last, "content": [*head, marked]}]

    @staticmethod
    def _usage_of(response: _HasUsage) -> TokenUsage:
        usage = response.usage
        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        )

    async def tool_turn(self, request: ChatRequest, tools: list[dict[str, object]]) -> ToolTurnResult:
        output_config: OutputConfigParam = {"effort": request.effort}
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=request.max_tokens,
            temperature=_temperature(request.temperature),
            system=self._render_system(request.system),
            messages=self._render_turns(request.turns),
            tools=tools,
            output_config=output_config,
        )
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                reasoning_parts.append(block.text.strip())
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=cast("dict[str, object]", block.input),
                    )
                )
        wants_tools = response.stop_reason == "tool_use"
        return ToolTurnResult(
            text=" ".join(reasoning_parts),
            tool_calls=tuple(calls) if wants_tools else (),
            usage=self._usage_of(response),
        )

    async def parse_structured(self, request: ChatRequest, schema: type[ModelT]) -> tuple[ModelT, TokenUsage]:
        output_config: OutputConfigParam = {"effort": request.effort}
        response = await self.client.messages.parse(
            model=self.model,
            max_tokens=request.max_tokens,
            temperature=_temperature(request.temperature),
            system=self._render_system(request.system),
            messages=self._render_turns(request.turns),
            output_format=schema,
            output_config=output_config,
        )
        if response.stop_reason == "refusal":
            raise ModelRefusedError(f"{self.model} refused the request")
        return cast("ModelT", response.parsed_output), self._usage_of(response)

    def warm_up(self) -> None:
        _ = self.client.messages

    def is_api_error(self, exc: Exception) -> bool:
        return isinstance(exc, anthropic.APIError)

    def is_schema_too_large(self, exc: Exception) -> bool:
        return isinstance(exc, anthropic.BadRequestError)


__all__ = ["AnthropicWire"]
