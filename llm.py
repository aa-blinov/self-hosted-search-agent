"""
LLM client — OpenRouter / LM Studio / любой OpenAI-совместимый сервер.

Thinking-режим Qwen3.5:
  OpenRouter  → reasoning.effort="none" в extra_body отключает thinking.
  LM Studio   → модель кладёт thinking в delta.reasoning_content, ответ в delta.content.
                Нужно достаточно токенов (LLM_MAX_TOKENS) чтобы thinking завершился
                и модель записала content. При max_tokens=1024 модель обрывается внутри
                reasoning_content и content остаётся пустым.
"""
import os
from datetime import datetime

from openai import OpenAI

MODEL       = os.getenv("LLM_MODEL",      "qwen/qwen3.5-35b-a3b")
MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "8192"))   # 1024 было мало для thinking

# Опциональный провайдер (например "alibaba"). Если пустой — не передаём в запрос.
_PROVIDER = os.getenv("LLM_PROVIDER", "").strip()


def _extra() -> dict:
    """Return extra_body dict. Built fresh each call so env overrides work at runtime."""
    body: dict = {
        "reasoning": {"effort": "none"},  # OpenRouter: отключает Qwen3 thinking
    }
    if _PROVIDER:
        body["provider"] = {
            "order": [_PROVIDER.lower()],
            "allow_fallbacks": False,
        }
    return body


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
— В конце ответа — список источников. КАЖДЫЙ источник на отдельной строке, формат строго:

[1] Название — URL
[2] Название — URL
[3] Название — URL

— Не выдумывай URL. Используй только те, что переданы в контексте.
"""


def _build_system_prompt() -> str:
    today = datetime.now().strftime("%d %B %Y, %A")
    return _SYSTEM_PROMPT_TEMPLATE.format(today=today)


RAG_ANALYSIS_PROMPT = """\
You are a research analyst reviewing papers on RAG and hallucination reduction.
For each paper abstract below, describe in 2-3 sentences:
1. The core technique proposed.
2. Whether it could improve a search assistant pipeline (SearXNG → LLM).
Use only the provided abstracts.
"""


def _build_context(sources: list[dict]) -> str:
    blocks = []
    for i, src in enumerate(sources, 1):
        text = src.get("snippet") or src.get("abstract") or ""
        blocks.append(f"[{i}] {src.get('title', '')}\nURL: {src.get('url', '')}\n{text}")
    return "\n\n---\n\n".join(blocks)


def _user_message(query: str, context: str) -> str:
    return (
        f"Контекст из найденных источников:\n\n{context}\n\n"
        f"---\n\n"
        f"Вопрос: {query}\n\n"
        f"Извлеки конкретные данные из контекста выше и ответь напрямую."
    )


def answer_with_sources(query: str, sources: list[dict], client: OpenAI) -> str:
    """Non-streaming fallback (используется в --query режиме)."""
    if not sources:
        return "No sources retrieved. Cannot answer without context."

    context = _build_context(sources)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user",   "content": _user_message(query, context)},
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        extra_body=_extra(),
    )
    return response.choices[0].message.content or ""


def answer_streaming(
    query: str,
    sources: list[dict],
    client: OpenAI,
    console,
    panel_width: int = 100,
) -> str:
    """
    Stream the answer token-by-token into a live Rich Panel.

    LM Studio / Qwen3.5 thinking mode:
      - delta.reasoning_content  → thinking tokens  (показываем как "⠦ Thinking…")
      - delta.content            → финальный ответ  (показываем в Answer панели)

    OpenRouter с reasoning.effort=none:
      - delta.reasoning_content  → пусто
      - delta.content            → сразу финальный ответ
    """
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.text import Text

    if not sources:
        console.print("[yellow]No sources retrieved. Cannot answer without context.[/]")
        return ""

    context = _build_context(sources)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user",   "content": _user_message(query, context)},
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        stream=True,
        extra_body=_extra(),
    )

    full_text    = ""
    thinking_len = 0

    def _answer_panel(text: str) -> Panel:
        return Panel(
            Markdown(text) if text.strip() else "[dim]…[/dim]",
            title="[bold green]Answer[/]",
            border_style="green",
            width=panel_width,
        )

    def _thinking_panel(n_chars: int) -> Panel:
        return Panel(
            Text(f"⠦ Thinking…  ({n_chars} chars)", style="dim"),
            title="[dim]Thinking[/]",
            border_style="dim",
            width=panel_width,
        )

    # Стартуем с пустым Answer — thinking panel появится только если придёт
    # reasoning_content (LM Studio с включённым thinking).
    with Live(
        _answer_panel(""),
        console=console,
        refresh_per_second=12,
        vertical_overflow="visible",
    ) as live:
        for chunk in stream:
            delta_obj = chunk.choices[0].delta

            thinking_delta = getattr(delta_obj, "reasoning_content", "") or ""
            content_delta  = delta_obj.content or ""

            if thinking_delta and not full_text:
                # thinking идёт, ответ ещё не начался
                thinking_len += len(thinking_delta)
                live.update(_thinking_panel(thinking_len))

            if content_delta:
                full_text += content_delta
                live.update(_answer_panel(full_text))

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
            {"role": "user",   "content": "\n\n---\n\n".join(blocks)},
        ],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
        extra_body=_extra(),
    )
    return response.choices[0].message.content or ""
