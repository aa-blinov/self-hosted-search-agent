from __future__ import annotations

import json
import re
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Env: only deployment‑sensitive options (keys, URLs, provider, profiles). Tuning: ``search_agent.tuning``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str | None = None
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "qwen/qwen3.5-35b-a3b"
    llm_provider: str | None = None
    llm_max_tokens: int = 8192
    llm_http_referer: str = "https://github.com/local/search-agent"

    search_provider: str = "brave"
    search_provider_override: str | None = None
    brave_api_key: str | None = None
    brave_base_url: str = "https://api.search.brave.com"
    brave_country: str = "US"
    brave_safesearch: str = "moderate"
    brave_goggles: str | None = None
    search_backend_fallback_delay_sec: float = 1.0
    brave_search_fallback: bool = True
    ddgs_region: str = "wt-wt"
    ddgs_safesearch: str = "moderate"
    ddgs_timeout: int = 15

    default_profile: str = "web"
    agent_receipts_dir: str | None = None

    logfire_token: str | None = None

    def resolved_brave_goggles(self) -> list[str]:
        """Parse ``brave_goggles``: JSON array, or semicolon/newline-separated URLs / inline rules."""
        raw = (self.brave_goggles or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return []
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
            return []
        return [p.strip() for p in re.split(r"[;\n]", raw) if p.strip()]

    def resolved_search_provider(self) -> str:
        o = (self.search_provider_override or "").strip().lower()
        if o:
            return o
        return (self.search_provider or "brave").strip().lower()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
