from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("LLM_API_KEY"))
    llm_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("LLM_BASE_URL"),
    )
    llm_model: str = Field(
        default="qwen/qwen3.5-35b-a3b",
        validation_alias=AliasChoices("LLM_MODEL"),
    )
    llm_provider: str | None = Field(default=None, validation_alias=AliasChoices("LLM_PROVIDER"))
    llm_max_tokens: int = Field(default=8192, validation_alias=AliasChoices("LLM_MAX_TOKENS"))
    llm_http_referer: str = Field(
        default="https://github.com/local/search-agent",
        validation_alias=AliasChoices("LLM_HTTP_REFERER"),
    )

    search_provider: str = Field(
        default="searxng",
        validation_alias=AliasChoices("SEARCH_PROVIDER"),
    )
    ddgs_region: str = Field(default="wt-wt", validation_alias=AliasChoices("DDGS_REGION"))
    ddgs_safesearch: str = Field(
        default="moderate",
        validation_alias=AliasChoices("DDGS_SAFESEARCH"),
    )
    ddgs_timeout: int = Field(default=15, validation_alias=AliasChoices("DDGS_TIMEOUT"))

    logfire_token: str | None = Field(default=None, validation_alias=AliasChoices("LOGFIRE_TOKEN"))
    logfire_service_name: str = Field(
        default="self-hosted-search-agent",
        validation_alias=AliasChoices("LOGFIRE_SERVICE_NAME"),
    )
    logfire_environment: str = Field(
        default="development",
        validation_alias=AliasChoices("LOGFIRE_ENVIRONMENT"),
    )
    logfire_send_to_logfire: str = Field(
        default="if-token-present",
        validation_alias=AliasChoices("LOGFIRE_SEND_TO_LOGFIRE"),
    )
    logfire_local: bool = Field(default=True, validation_alias=AliasChoices("LOGFIRE_LOCAL"))
    logfire_console: bool = Field(default=False, validation_alias=AliasChoices("LOGFIRE_CONSOLE"))
    logfire_include_content: bool = Field(
        default=False,
        validation_alias=AliasChoices("LOGFIRE_INCLUDE_CONTENT"),
    )

    def resolved_send_to_logfire(self) -> bool | str:
        value = (self.logfire_send_to_logfire or "").strip().lower()
        if value in {"", "if-token-present"}:
            return "if-token-present"
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
