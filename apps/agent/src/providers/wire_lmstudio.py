"""LM Studio Chat Completions implementation of ChatWire."""

from __future__ import annotations

from .wire_openai import OpenAIWire

LMSTUDIO_DEFAULT_BASE_URL: str = "http://localhost:1234/v1"
LMSTUDIO_DUMMY_API_KEY: str = "lm-studio"


class LMStudioWire(OpenAIWire):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key or LMSTUDIO_DUMMY_API_KEY,
            base_url=base_url or LMSTUDIO_DEFAULT_BASE_URL,
            max_retries=max_retries,
        )


def list_loaded_models(base_url: str | None = None) -> list[dict[str, object]]:
    import json
    from urllib.request import Request, urlopen

    endpoint = (base_url or LMSTUDIO_DEFAULT_BASE_URL).rstrip("/") + "/models"
    request = Request(endpoint, headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:
        payload: dict[str, object] = json.loads(response.read().decode("utf-8"))
    data = payload.get("data", ())
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


__all__ = ["LMSTUDIO_DEFAULT_BASE_URL", "LMStudioWire", "list_loaded_models"]
