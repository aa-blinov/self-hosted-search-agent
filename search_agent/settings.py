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

    extract_max_chars: int | None = None
    compose_answer_max_tokens: int | None = None
    rag_analysis_max_tokens: int | None = None
    claim_decompose_max_tokens: int | None = None
    verify_claim_max_tokens: int | None = None
    time_normalize_max_tokens: int | None = None

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

    def resolved_extract_max_chars(self) -> int:
        """Env ``EXTRACT_MAX_CHARS`` overrides :data:`search_agent.tuning.EXTRACT_MAX_CHARS` when set and positive."""
        v = self.extract_max_chars
        if v is not None and v > 0:
            return int(v)
        from search_agent import tuning

        return tuning.EXTRACT_MAX_CHARS

    def resolved_compose_answer_max_tokens(self) -> int:
        """Env ``COMPOSE_ANSWER_MAX_TOKENS`` overrides :data:`search_agent.tuning.COMPOSE_ANSWER_MAX_TOKENS` when set and positive."""
        v = self.compose_answer_max_tokens
        if v is not None and v > 0:
            return int(v)
        from search_agent import tuning

        return tuning.COMPOSE_ANSWER_MAX_TOKENS

    def resolved_rag_analysis_max_tokens(self) -> int:
        """Env ``RAG_ANALYSIS_MAX_TOKENS`` overrides :data:`search_agent.tuning.RAG_ANALYSIS_MAX_TOKENS` when set and positive."""
        v = self.rag_analysis_max_tokens
        if v is not None and v > 0:
            return int(v)
        from search_agent import tuning

        return tuning.RAG_ANALYSIS_MAX_TOKENS

    def resolved_claim_decompose_max_tokens(self) -> int:
        """Env ``CLAIM_DECOMPOSE_MAX_TOKENS`` overrides :data:`search_agent.tuning.CLAIM_DECOMPOSE_MAX_TOKENS` when set and positive."""
        v = self.claim_decompose_max_tokens
        if v is not None and v > 0:
            return int(v)
        from search_agent import tuning

        return tuning.CLAIM_DECOMPOSE_MAX_TOKENS

    def resolved_verify_claim_max_tokens(self) -> int:
        """Env ``VERIFY_CLAIM_MAX_TOKENS`` overrides :data:`search_agent.tuning.VERIFY_CLAIM_MAX_TOKENS` when set and positive."""
        v = self.verify_claim_max_tokens
        if v is not None and v > 0:
            return int(v)
        from search_agent import tuning

        return tuning.VERIFY_CLAIM_MAX_TOKENS

    def resolved_time_normalize_max_tokens(self) -> int:
        """Env ``TIME_NORMALIZE_MAX_TOKENS`` overrides :data:`search_agent.tuning.TIME_NORMALIZE_MAX_TOKENS` when set and positive."""
        v = self.time_normalize_max_tokens
        if v is not None and v > 0:
            return int(v)
        from search_agent import tuning

        return tuning.TIME_NORMALIZE_MAX_TOKENS


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
