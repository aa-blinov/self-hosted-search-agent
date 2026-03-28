"""Compatibility facade for LLM-backed answer generation and paper analysis."""

from __future__ import annotations

from datetime import datetime

from search_agent.infrastructure.llm_tasks import build_task_runner
from search_agent.settings import get_settings

_SETTINGS = get_settings()
MODEL = _SETTINGS.llm_model
MAX_TOKENS = _SETTINGS.llm_max_tokens
_PROVIDER = (_SETTINGS.llm_provider or "").strip()


def _extra() -> dict:
    body: dict = {"reasoning": {"effort": "none"}}
    if _PROVIDER:
        body["provider"] = {
            "order": [_PROVIDER.lower()],
            "allow_fallbacks": False,
        }
    return body


def answer_with_sources(query: str, sources: list[dict], client=None, log=None) -> str:
    del client
    today = datetime.now().strftime("%d %B %Y, %A")
    return build_task_runner().answer_with_sources(query, sources, today=today, log=log)


def analyze_rag_papers(papers: list[dict], client=None, log=None) -> str:
    del client
    return build_task_runner().analyze_rag_papers(papers, log=log)
