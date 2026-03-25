from __future__ import annotations

import json
import re
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Env vars: UPPER_SNAKE of each field name (e.g. ``extract_max_chars`` → ``EXTRACT_MAX_CHARS``)."""

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
    # When set, overrides ``search_provider`` (CLI ``-S`` sets this in the environment).
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

    shallow_fetch_timeout: int = 8
    extract_max_chars: int = 3000
    crawl4ai_timeout: int = 25
    crawl4ai_delay_before_html: float = 1.0
    crawl4ai_prefer_raw: bool = False
    fetch_try_http_first: bool = False
    fetch_http_min_chars: int = 1200
    fetch_shallow_concurrency: int = 8
    fetch_deep_concurrency: int = 2

    agent_max_claim_iterations: int = 3
    agent_max_query_variants: int = 6
    agent_max_refine_variants: int = 12
    agent_fetch_top_n: int = 4
    agent_passage_top_k: int = 8
    serp_gate_min_urls: int = 15
    serp_gate_max_urls: int = 30
    agent_snippet_fallback_docs: int = 2
    shallow_fetch_short_limit: int = 8
    shallow_fetch_targeted_limit: int = 12
    shallow_fetch_iterative_limit: int = 15
    deep_fetch_short_limit: int = 2
    deep_fetch_targeted_limit: int = 3
    deep_fetch_iterative_limit: int = 4
    cheap_passage_limit: int = 12

    eval_case_delay_sec: float = 2.0

    default_profile: str = "web"
    agent_receipts_dir: str | None = None

    logfire_token: str | None = None
    logfire_service_name: str = "self-hosted-search-agent"
    logfire_environment: str = "development"
    logfire_send_to_logfire: str = "if-token-present"
    logfire_local: bool = True
    logfire_console: bool = False
    logfire_include_content: bool = False

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
