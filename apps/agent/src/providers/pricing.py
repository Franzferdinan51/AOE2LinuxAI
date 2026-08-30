"""Per-model token pricing — the single source for every cost figure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import structlog

if TYPE_CHECKING:
    from .base import TokenUsage

log = structlog.stdlib.get_logger()

_PER_MILLION: Final = 1_000_000

_CACHE_READ_MULTIPLIER: Final = 0.1
_CACHE_WRITE_MULTIPLIER: Final = 1.25
_RATE_DECIMALS: Final = 4


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Dollars per million tokens for one model."""

    input: float
    output: float
    cache_read: float
    cache_write: float


def _standard(input_rate: float, output_rate: float) -> ModelPrice:
    return ModelPrice(
        input=input_rate,
        output=output_rate,
        cache_read=round(input_rate * _CACHE_READ_MULTIPLIER, _RATE_DECIMALS),
        cache_write=round(input_rate * _CACHE_WRITE_MULTIPLIER, _RATE_DECIMALS),
    )


_PRICES: Final[dict[str, ModelPrice]] = {
    "claude-opus-5": _standard(5.00, 25.00),
    "claude-opus-4-7": _standard(5.00, 25.00),
    "claude-sonnet-5": _standard(3.00, 15.00),
    "claude-sonnet-4-6": _standard(3.00, 15.00),
    "claude-haiku-4-5": _standard(1.00, 5.00),
    "claude-haiku-4-5-20251001": _standard(1.00, 5.00),
    "gpt-5.6-luna": _standard(0.20, 1.20),
    "gpt-5.6-terra": _standard(2.00, 12.00),
    "kimi-k2.7-code": _standard(0.95, 4.00),
}

_UNKNOWN = ModelPrice(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0)


def price_for(model: str) -> ModelPrice:
    price = _PRICES.get(model)
    if price is None:
        log.warning("pricing_unknown_model", model=model)
        return _UNKNOWN
    return price


def cost_usd(model: str, usage: TokenUsage) -> float:
    price = price_for(model)
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    ) / _PER_MILLION


__all__ = ["ModelPrice", "cost_usd", "price_for"]
