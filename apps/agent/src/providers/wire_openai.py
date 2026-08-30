"""OpenAI-compatible Chat Completions implementation of `ChatWire`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import openai
import structlog

from .action_tools import to_openai_tools
from .base import (
    AssistantTurn, ChatRequest, ModelRefusedError, SystemBlock, TokenUsage,
    ToolCall, ToolOutcome, ToolResultsTurn, ToolTurnResult, UserTurn, tool_outcome_json,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from .base import ModelT, Turn

log = structlog.stdlib.get_logger()


def _temperature(value: float | None) -> float | openai.NotGiven:
    return openai.NOT_GIVEN if value is None else value


class OpenAIWire:
    """`ChatWire` over `openai.AsyncOpenAI`, pointed at any compatible endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)
        self.endpoint = str(self.client.base_url)

    @staticmethod
    def _render_system(blocks: tuple[SystemBlock, ...]) -> list[dict[str, object]]:
        text = "\n\n".join(block.text for block in blocks if block.text)
        return [{"role": "system", "content": text}] if text else []

    @staticmethod
    def _render_tool_call(call: ToolCall) -> dict[str, object]:
        return {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }

    @staticmethod
    def _render_outcome(outcome: ToolOutcome) -> dict[str, object]:
        return {
            "role": "tool",
            "tool_call_id": outcome.tool_call_id,
            "content": tool_outcome_json(outcome),
        }

    @classmethod
    def _render_turns(cls, turns: tuple[Turn, ...]) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        for turn in turns:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantTurn):
                message: dict[str, object] = {"role": "assistant", "content": turn.text or None}
                if turn.tool_calls:
                    message["tool_calls"] = [cls._render_tool_call(c) for c in turn.tool_calls]
                messages.append(message)
            elif isinstance(turn, ToolResultsTurn):
                messages.extend(cls._render_outcome(outcome) for outcome in turn.outcomes)
        return messages

    @classmethod
    def _render_messages(cls, request: ChatRequest) -> list[dict[str, object]]:
        return cls._render_system(request.system) + cls._render_turns(request.turns)

    @staticmethod
    def _usage_of(response: ChatCompletion) -> TokenUsage:
        usage = response.usage
        if usage is None:
            return TokenUsage()
        details = usage.prompt_tokens_details
        cached = (details.cached_tokens or 0) if details is not None else 0
        return TokenUsage(
            input_tokens=max(usage.prompt_tokens - cached, 0),
            output_tokens=usage.completion_tokens,
            cache_read_tokens=cached,
            cache_write_tokens=0,
        )

    @classmethod
    def _parse_tool_calls(cls, message: ChatCompletionMessage) -> tuple[ToolCall, ...]:
        return tuple(
            ToolCall(
                id=item.id,
                name=item.function.name,
                arguments=cls._decode(item.function.name, item.function.arguments),
            )
            for item in (message.tool_calls or [])
            if item.type == "function"
        )

    @staticmethod
    def _decode(tool: str, arguments: str) -> dict[str, object]:
        try:
            decoded: object = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            log.warning("openai_tool_arguments_unparsable", tool=tool, raw=arguments[:200])
            return {}
        return decoded if isinstance(decoded, dict) else {}

    async def tool_turn(self, request: ChatRequest, tools: list[dict[str, object]]) -> ToolTurnResult:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self._render_messages(request),
            tools=to_openai_tools(tools),
            temperature=_temperature(request.temperature),
            reasoning_effort="none",
            max_completion_tokens=request.max_tokens,
        )
        choice = response.choices[0]
        calls = self._parse_tool_calls(choice.message)
        return ToolTurnResult(
            text=(choice.message.content or "").strip(),
            tool_calls=calls if choice.finish_reason == "tool_calls" else (),
            usage=self._usage_of(response),
        )

    async def parse_structured(self, request: ChatRequest, schema: type[ModelT]) -> tuple[ModelT, TokenUsage]:
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=self._render_messages(request),
            response_format=schema,
            temperature=_temperature(request.temperature),
            reasoning_effort=request.effort,
            max_completion_tokens=request.max_tokens,
        )
        message = response.choices[0].message
        if message.refusal:
            raise ModelRefusedError(f"{self.model} refused the request: {message.refusal}")
        if message.parsed is None:
            raise ValueError(f"{self.model} returned no parsable {schema.__name__}")
        return message.parsed, self._usage_of(response)

    def warm_up(self) -> None:
        _ = self.client.chat

    def is_api_error(self, exc: Exception) -> bool:
        return isinstance(exc, openai.APIError)

    def is_schema_too_large(self, exc: Exception) -> bool:
        return False


__all__ = ["OpenAIWire"]
