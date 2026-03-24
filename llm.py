"""
LLM client via OpenRouter.

Provider: Alibaba only (allow_fallbacks=False).
Thinking: disabled via reasoning.effort="none".
"""
import os
from datetime import datetime

from openai import OpenAI

MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-35b-a3b")

# Passed to every completion call via extra_body
_EXTRA = {
    "provider": {
        "order": ["alibaba"],
        "allow_fallbacks": False,  # Alibaba natively serves this model
    },
    "reasoning": {
        "effort": "none",          # disable Qwen3 thinking/chain-of-thought
    },
}

_SYSTEM_PROMPT_TEMPLATE = """\
Ты — поисковый ассистент. Поиск уже выполнен: ниже передан контекст из реальных источников.
Сегодня: {today}.

Твоя задача — извлечь из контекста конкретные факты и дать прямой ответ на вопрос.

Правила работы с контекстом:
1. ИЗВЛЕКАЙ данные из текста и ОТВЕЧАЙ ими напрямую.
   Плохо: «На сайте gismeteo можно найти прогноз погоды.»
   Хорошо: «В среду +12°C, облачно, ветер 5 м/с [1].»
2. Если источники противоречат друг другу — укажи это явно и приведи оба значения.
3. Если в контексте есть только частичные данные — изложи то, что есть, и честно скажи, \
чего не хватает. Не перенаправляй пользователя по ссылкам.
4. Никогда не добавляй факты из своих внутренних знаний — только из контекста.
5. Если контекст совсем не содержит нужных данных — ответь: \
«В найденных источниках недостаточно информации для ответа.»

Оформление:
— Каждое фактическое утверждение должно содержать inline-ссылку: [1], [2] и т.д.
— В конце ответа — нумерованный список источников:
  [1] Название — URL
  [2] Название — URL
— Не выдумывай URL. Используй только те, что переданы в контексте.
"""


def _build_system_prompt() -> str:
    today = datetime.now().strftime("%d %B %Y, %A")  # "24 March 2026, Tuesday"
    return _SYSTEM_PROMPT_TEMPLATE.format(today=today)

RAG_ANALYSIS_PROMPT = """\
You are a research analyst reviewing papers on RAG and hallucination reduction.
For each paper abstract below, describe in 2-3 sentences:
1. The core technique proposed.
2. Whether it could improve a search assistant pipeline (SearXNG → LLM).
Use only the provided abstracts.
"""


def _extra() -> dict:
    """Return extra_body dict. Built fresh each call so env overrides work."""
    return _EXTRA


def _build_context(sources: list[dict]) -> str:
    blocks = []
    for i, src in enumerate(sources, 1):
        text = src.get("snippet") or src.get("abstract") or ""
        blocks.append(f"[{i}] {src.get('title', '')}\nURL: {src.get('url', '')}\n{text}")
    return "\n\n---\n\n".join(blocks)


def answer_with_sources(query: str, sources: list[dict], client: OpenAI) -> str:
    """Non-streaming fallback (used in --query CLI mode)."""
    if not sources:
        return "No sources retrieved. Cannot answer without context."

    context = _build_context(sources)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": (
                f"Контекст из найденных источников:\n\n{context}\n\n"
                f"---\n\n"
                f"Вопрос: {query}\n\n"
                f"Извлеки конкретные данные из контекста выше и ответь напрямую."
            )},
        ],
        temperature=0.1,
        max_tokens=1024,
        extra_body=_extra(),
    )
    return response.choices[0].message.content or ""


def answer_streaming(
    query: str,
    sources: list[dict],
    client: OpenAI,
    console,                  # rich.console.Console
    panel_width: int = 100,
) -> str:
    """
    Stream the answer token-by-token into a live Rich Panel.

    Returns the full generated text for logging/post-processing.
    """
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel

    if not sources:
        console.print("[yellow]No sources retrieved. Cannot answer without context.[/]")
        return ""

    context = _build_context(sources)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": (
                f"Контекст из найденных источников:\n\n{context}\n\n"
                f"---\n\n"
                f"Вопрос: {query}\n\n"
                f"Извлеки конкретные данные из контекста выше и ответь напрямую."
            )},
        ],
        temperature=0.1,
        max_tokens=1024,
        stream=True,
        extra_body=_extra(),
    )

    full_text = ""

    def _panel(text: str):
        return Panel(
            Markdown(text) if text.strip() else "[dim]…[/dim]",
            title="[bold green]Answer[/]",
            border_style="green",
            width=panel_width,
        )

    with Live(
        _panel(""),
        console=console,
        refresh_per_second=12,   # max 12 re-renders/sec — не тормозит на 35B
        vertical_overflow="visible",
    ) as live:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_text += delta
                live.update(_panel(full_text))

    return full_text


def analyze_rag_papers(papers: list[dict], client: OpenAI) -> str:
    if not papers:
        return "No papers retrieved."

    blocks = []
    for i, p in enumerate(papers, 1):
        blocks.append(
            f"[{i}] {p['title']}\nAuthors: {', '.join(p.get('authors', []))}\n"
            f"URL: {p['url']}\nAbstract: {p['abstract']}"
        )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAG_ANALYSIS_PROMPT},
            {"role": "user", "content": "\n\n---\n\n".join(blocks)},
        ],
        temperature=0.2,
        max_tokens=1200,
        extra_body=_extra(),
    )
    return response.choices[0].message.content or ""
