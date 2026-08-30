"""Provider-neutral LLM types: the seam between game logic and wire format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypedDict, TypeVar

from ..config import EffortLevel as EffortLevel
from ..config import WireName as WireName

if TYPE_CHECKING:
    from pydantic import BaseModel


class LLMResult(TypedDict, total=False):
    """Payload from `ExecutorProvider.get_actions`; every key is optional."""

    reasoning: str
    actions: list[dict[str, object]]
    observations: dict[str, object] | None
    actions_already_executed: bool
    success_count: int
    error: bool


ModelT = TypeVar("ModelT", bound="BaseModel")


class ModelRefusedError(RuntimeError):
    """The model declined to answer."""


def text_of_blocks(content: list[dict[str, object]]) -> str:
    return "\n\n".join(
        str(block.get("text", "")) for block in content if block.get("type") == "text"
    )


@dataclass(frozen=True, slots=True)
class SystemBlock:
    """One system-prompt segment plus whether it is worth caching."""

    text: str
    cacheable: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model's request to run one tool; handlers narrow `arguments` themselves."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of running one `ToolCall`, before wire-specific framing."""

    tool_call_id: str
    success: bool
    detail: str
    entities: tuple[dict[str, object], ...] = ()


def tool_outcome_json(outcome: ToolOutcome) -> str:
    payload: dict[str, object] = {"success": outcome.success, "detail": outcome.detail}
    if outcome.entities:
        payload["entities"] = list(outcome.entities)
    return json.dumps(payload)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Per-call token counts, normalised across vendors."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class UserTurn:
    text: str


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolResultsTurn:
    outcomes: tuple[ToolOutcome, ...]


Turn = UserTurn | AssistantTurn | ToolResultsTurn


@dataclass(frozen=True, slots=True)
class ChatRequest:
    system: tuple[SystemBlock, ...]
    turns: tuple[Turn, ...]
    max_tokens: int
    temperature: float | None
    effort: EffortLevel = "low"


@dataclass(frozen=True, slots=True)
class ToolTurnResult:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def wants_more_tools(self) -> bool:
        return bool(self.tool_calls)


class ChatWire(Protocol):
    """Vendor-specific transport for one chat model."""

    model: str
    endpoint: str

    async def tool_turn(
        self,
        request: ChatRequest,
        tools: list[dict[str, object]],
    ) -> ToolTurnResult: ...

    async def parse_structured(
        self,
        request: ChatRequest,
        schema: type[ModelT],
    ) -> tuple[ModelT, TokenUsage]: ...

    def warm_up(self) -> None: ...

    def is_api_error(self, exc: Exception) -> bool: ...

    def is_schema_too_large(self, exc: Exception) -> bool: ...


__all__ = [
    "AssistantTurn", "ChatRequest", "ChatWire", "EffortLevel", "LLMResult",
    "ModelRefusedError", "ModelT", "SystemBlock", "TokenUsage", "ToolCall",
    "ToolOutcome", "ToolResultsTurn", "ToolTurnResult", "Turn", "UserTurn",
    "WireName", "text_of_blocks", "tool_outcome_json",
]
