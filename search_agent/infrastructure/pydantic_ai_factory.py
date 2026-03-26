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


def build_model_settings(
    settings: AppSettings,
    *,
    max_tokens: int,
    temperature: float,
) -> OpenAIChatModelSettings:
    extra_body: dict[str, object] = {"reasoning": {"effort": "none"}}
    if settings.llm_provider:
        extra_body["provider"] = {
            "order": [settings.llm_provider.lower()],
            "allow_fallbacks": False,
        }
    return OpenAIChatModelSettings(
        max_tokens=min(max_tokens, settings.llm_max_tokens),
        temperature=temperature,
        extra_headers={"HTTP-Referer": settings.llm_http_referer},
        extra_body=extra_body,
    )
