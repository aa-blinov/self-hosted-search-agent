"""
EvidenceLoop — iterative search with quality assessment and query reformulation.

Based on WebDetective (arXiv 2510.05137).

Query reformulation uses SearXNG's own `suggestions` field — no LLM call needed.
If SearXNG returns no suggestions, falls back to LLM reformulation.
"""

import os
from datetime import datetime
from typing import Callable

from openai import OpenAI

MAX_ITERATIONS = int(os.getenv("EVIDENCE_LOOP_ITERATIONS", "2"))
MIN_CONTEXT_CHARS = int(os.getenv("EVIDENCE_LOOP_MIN_CHARS", "800"))

_REFORMULATE_SYSTEM = (
    "You are a search query optimizer. "
    "Generate alternative search queries to find more relevant information."
)

# Только конкретные относительные ссылки, которые можно точно перевести в дату.
# НЕ включаем "недавно"/"recently" — слишком размыто, LLM пинит к сегодняшней дате.
_TIME_RELATIVE = {
    # русские
    "сегодня", "завтра", "вчера", "послезавтра",
    "на этой неделе", "эта неделя",
    "на следующей неделе", "следующей неделе", "следующая неделя",
    "на прошлой неделе", "прошлой неделе",
    "в этом месяце", "этот месяц",
    "в следующем месяце", "следующий месяц",
    "в прошлом месяце", "прошлый месяц",
    "сейчас", "прямо сейчас",
    # english
    "today", "tomorrow", "yesterday",
    "this week", "next week", "last week",
    "this month", "next month", "last month",
    "right now",
}


def _has_time_ref(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _TIME_RELATIVE)


def _normalize_query(query: str, client: OpenAI, log=None) -> str:
    """
    Заменяет относительные временны́е ссылки на конкретные даты через LLM.
    Пример: "погода в астане на этой неделе"
         → "погода в астане 24-30 марта 2026"
    """
    if not _has_time_ref(query):
        return query

    today = datetime.now()
    today_str = today.strftime("%d %B %Y, %A")  # "24 March 2026, Tuesday"

    prompt = (
        f"Сегодня: {today_str}.\n\n"
        "Перепиши поисковый запрос, заменив ТОЛЬКО точные относительные временны́е "
        "ссылки на конкретные даты.\n"
        "Правила:\n"
        "- «сегодня» → конкретная дата (например, 24 марта 2026)\n"
        "- «завтра» / «вчера» → конкретная дата\n"
        "- «на этой неделе» → диапазон дат (24–30 марта 2026)\n"
        "- «в следующем месяце» → название месяца и год\n"
        "- «недавно», «последнее время», «в последнее время» — НЕ трогай, "
        "это не конкретная дата\n"
        "Верни ТОЛЬКО переписанный запрос, без пояснений.\n\n"
        f"Запрос: {query}"
    )

    try:
        from llm import MODEL, _extra
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
            extra_body=_extra(),
        )
        normalized = (resp.choices[0].message.content or "").strip()
        if normalized and normalized != query:
            if log:
                log(f"  [dim]↳ query normalized: [italic]{normalized}[/italic][/dim]")
            return normalized
    except Exception as e:
        if log:
            log(f"  [dim yellow]⚠ normalize error: {e}[/dim yellow]")

    return query


def _assess_quality(sources: list[dict], query: str) -> bool:
    total_text = " ".join(s.get("snippet", "") for s in sources)
    if len(total_text) < MIN_CONTEXT_CHARS:
        return False
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    if keywords:
        text_lower = total_text.lower()
        if sum(1 for kw in keywords if kw in text_lower) == 0:
            return False
    return True


def _llm_reformulate(original_query: str, attempt: int, client: OpenAI) -> list[str]:
    """Fallback: ask LLM for alternative queries when SearXNG has no suggestions."""
    prompt = (
        f"The search query '{original_query}' returned insufficient results "
        f"(attempt {attempt}).\n"
        "Generate 2 alternative search queries. "
        "Return ONLY the queries, one per line, no numbering."
    )
    try:
        from llm import MODEL, _extra
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _REFORMULATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=80,
            extra_body=_extra(),
        )
        raw = resp.choices[0].message.content or ""
        return [q.strip() for q in raw.strip().splitlines() if q.strip()][:2]
    except Exception as e:
        print(f"[evidence loop LLM reformulate error] {e}")
        return []


def evidence_loop(
    query: str,
    search_fn: Callable,   # (query: str, profile, log) -> tuple[list[dict], list[str]]
    profile,               # SearchProfile
    client: OpenAI,
    log=None,
) -> tuple[list[dict], list[str]]:
    """
    Iteratively search until context quality is sufficient.

    Returns:
        (sources, search_log)
    """
    _noop = lambda msg: None
    log = log or _noop

    seen_urls: set[str] = set()
    all_sources: list[dict] = []
    search_log: list[str] = []

    def _add(new_sources: list[dict]) -> None:
        for s in new_sources:
            if s.get("url") not in seen_urls:
                seen_urls.add(s["url"])
                all_sources.append(s)

    log(f"\n[bold]▸ EvidenceLoop[/bold]  [dim]max {MAX_ITERATIONS} iter · "
        f"min {MIN_CONTEXT_CHARS} chars[/dim]")

    # Нормализуем относительные даты → конкретные до первого поиска
    query = _normalize_query(query, client, log=log)

    for attempt in range(MAX_ITERATIONS):
        current_query = query if attempt == 0 else None

        if attempt == 0:
            log(f"\n[bold]  Iteration 1/{MAX_ITERATIONS}[/bold]")
            sources, suggestions = search_fn(query, profile, log=log)
            _add(sources)
            search_log.append(query)
        else:
            total = sum(len(s.get("snippet", "")) for s in all_sources)
            if _assess_quality(all_sources, query):
                log(f"\n  [green]✓ quality OK[/green]  [dim]{total} chars total[/dim]")
                break

            log(f"\n  [yellow]✗ context weak[/yellow]  "
                f"[dim]{total} chars — reformulating…[/dim]")

            alt_queries = suggestions[:2] if suggestions else _llm_reformulate(query, attempt, client)
            src_label = "SearXNG suggestions" if suggestions else "LLM"
            log(f"  [dim]↻ {src_label}: {alt_queries}[/dim]")

            log(f"\n[bold]  Iteration {attempt + 1}/{MAX_ITERATIONS}[/bold]")
            for alt in alt_queries:
                new_sources, new_suggestions = search_fn(alt, profile, log=log)
                _add(new_sources)
                search_log.append(alt)
                suggestions = new_suggestions or suggestions

    total = sum(len(s.get("snippet", "")) for s in all_sources)
    log(f"\n  [dim]total context: {total} chars across {len(all_sources)} sources[/dim]\n")

    return all_sources, search_log
