from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from search_agent.infrastructure.pydantic_ai_factory import build_model_settings, build_openai_model
from search_agent.infrastructure.telemetry import configure_logfire
from search_agent.settings import AppSettings, get_settings

_SYSTEM_PROMPT_TEMPLATE = """\
Ты — поисковый ассистент. Поиск уже выполнен: ниже передан контекст из реальных источников.
Сегодня: {today}.

Твоя задача — извлечь из контекста конкретные факты и дать прямой ответ на вопрос.

Правила:
1. Используй только факты из контекста.
2. Если источники противоречат друг другу, укажи это явно.
3. Если данных недостаточно, так и напиши.
4. Каждый фактический тезис должен иметь inline-ссылку [1], [2] и т.д.
5. В конце добавь список источников в формате:
[1] Название — URL
[2] Название — URL
"""

RAG_ANALYSIS_PROMPT = """\
You are a research analyst reviewing papers on RAG and hallucination reduction.
For each paper abstract below, describe in 2-3 sentences:
1. The core technique proposed.
2. Whether it could improve a search assistant pipeline (Brave Search -> LLM).
Use only the provided abstracts.
"""


class PydanticAITaskRunner:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        configure_logfire(settings)
        self._model = build_openai_model(settings)
        self._grounded_answer_agent = Agent(
            self._model,
            output_type=str,
            retries=1,
            instrument=True,
        )
        self._research_analysis_agent = Agent(
            self._model,
            output_type=str,
            retries=1,
            instrument=True,
            system_prompt=RAG_ANALYSIS_PROMPT,
        )

    def answer_with_sources(self, query: str, sources: list[dict], *, today: str) -> str:
        if not sources:
            return "No sources retrieved. Cannot answer without context."

        prompt = (
            _SYSTEM_PROMPT_TEMPLATE.format(today=today)
            + "\n\n"
            + _build_context_block(query, sources)
        )
        result = self._grounded_answer_agent.run_sync(
            prompt,
            model_settings=build_model_settings(
                self._settings,
                max_tokens=1600,
                temperature=0.1,
            ),
        )
        return result.output.strip()

    def analyze_rag_papers(self, papers: list[dict]) -> str:
        if not papers:
            return "No papers retrieved."

        prompt = "\n\n---\n\n".join(
            [
                f"[{i}] {paper['title']}\n"
                f"Authors: {', '.join(paper.get('authors', []))}\n"
                f"URL: {paper['url']}\n"
                f"Abstract: {paper['abstract']}"
                for i, paper in enumerate(papers, 1)
            ]
        )
        result = self._research_analysis_agent.run_sync(
            prompt,
            model_settings=build_model_settings(
                self._settings,
                max_tokens=1400,
                temperature=0.2,
            ),
        )
        return result.output.strip()


def _build_context_block(query: str, sources: list[dict]) -> str:
    blocks = []
    for i, src in enumerate(sources, 1):
        text = src.get("snippet") or src.get("abstract") or ""
        blocks.append(f"[{i}] {src.get('title', '')}\nURL: {src.get('url', '')}\n{text}")
    context = "\n\n---\n\n".join(blocks)
    return (
        f"Контекст из найденных источников:\n\n{context}\n\n"
        f"---\n\n"
        f"Вопрос: {query}\n\n"
        "Извлеки конкретные данные из контекста выше и ответь напрямую."
    )


@lru_cache(maxsize=1)
def build_task_runner() -> PydanticAITaskRunner:
    return PydanticAITaskRunner(get_settings())
