from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from search_agent import tuning
from search_agent.settings import AppSettings


def build_openai_provider(settings: AppSettings) -> OpenAIProvider:
    # Use explicit AsyncOpenAI client so we can set a hard timeout.
    # Without this, a stalled provider response hangs forever (observed: 603 s).
    # max_retries=0 because the agent loop handles retries at the iteration level.
    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=float(tuning.LLM_REQUEST_TIMEOUT),
        max_retries=0,
    )
    return OpenAIProvider(openai_client=client)


def build_openai_model(settings: AppSettings) -> OpenAIChatModel:
    return OpenAIChatModel(
        settings.llm_model,
        provider=build_openai_provider(settings),
    )


def _is_reasoning_model(model_name: str) -> bool:
    """Return True for models that use mandatory reasoning (o1/o3/o4/gpt-oss family).

    These models:
    - Do NOT accept a temperature parameter (raises 400)
    - Require PromptedOutput for structured responses (JSON schema / tool calls
      are silently ignored or malformed on some providers)
    """
    name = model_name.lower()
    return any(x in name for x in ("gpt-oss", "/o1", "/o3", "/o4", "-o1", "-o3", "-o4"))


def build_model_settings(
    settings: AppSettings,
    *,
    max_tokens: int,
    temperature: float,
) -> OpenAIChatModelSettings:
    extra_body: dict[str, object] = {}
    model_lower = settings.llm_model.lower()

    # qwen models generate <think> tokens before the answer which consume part
    # of max_tokens; disabling reasoning keeps outputs compact and fast.
    # Other models (gpt-oss, llama, etc.) either don't support this field or
    # require reasoning — skip it for non-qwen models.
    if "qwen" in model_lower:
        extra_body["reasoning"] = {"effort": "none"}

    if settings.llm_provider:
        extra_body["provider"] = {
            "order": [settings.llm_provider.lower()],
            "allow_fallbacks": False,
        }

    kwargs: dict[str, object] = {
        "max_tokens": min(max_tokens, settings.llm_max_tokens),
        "extra_headers": {"HTTP-Referer": settings.llm_http_referer},
        "extra_body": extra_body,
    }
    # Reasoning models (gpt-oss, o1, o3, o4) reject the temperature parameter —
    # omit it entirely so the API doesn't return a 400 error.
    if not _is_reasoning_model(settings.llm_model):
        kwargs["temperature"] = temperature

    return OpenAIChatModelSettings(**kwargs)  # type: ignore[arg-type]
