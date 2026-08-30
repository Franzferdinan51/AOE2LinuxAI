"""LM Studio Chat Completions implementation of `ChatWire`.

LM Studio runs an OpenAI-compatible server on `http://localhost:1234/v1` by
default; this adapter inherits every transport detail from `OpenAIWire` and
only changes the defaults (no auth header, a localhost endpoint). The wire is
selected by `AOE2_LLM_WIRE=lmstudio`; the model id follows the loaded LM Studio
model and is set with `AOE2_MODEL=<loaded-model-id>` (`lms ls` lists them).

LM Studio ignores the `Authorization` header, but the OpenAI SDK insists on a
non-empty `api_key`. `lmstudio` is a string the server treats as opaque, and
also the documented "no-auth" sentinel from `lms server --help`.
"""

from __future__ import annotations

from .wire_openai import OpenAIWire

LMSTUDIO_DEFAULT_BASE_URL: str = "http://localhost:1234/v1"
LMSTUDIO_DUMMY_API_KEY: str = "lm-studio"


class LMStudioWire(OpenAIWire):
    """`ChatWire` over LM Studio's local OpenAI-compatible endpoint."""

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
    """Return the LM Studio server's currently-loaded models.

    LM Studio exposes an OpenAI-compatible `GET /v1/models`. `requests` is
    imported lazily so this helper is cheap to call from tooling without pulling
    in the OpenAI SDK on the user's behalf.
    """
    import json
    from urllib.request import Request, urlopen

    endpoint = (base_url or LMSTUDIO_DEFAULT_BASE_URL).rstrip("/") + "/models"
    request = Request(endpoint, headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:  # noqa: S310 — endpoint is user-supplied
        payload: dict[str, object] = json.loads(response.read().decode("utf-8"))
    data = payload.get("data", ())
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


__all__ = ["LMSTUDIO_DEFAULT_BASE_URL", "LMStudioWire", "list_loaded_models"]
