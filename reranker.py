"""
LLM-based reranker on SearXNG snippets.

Идея: перед тем как запускать crawl4ai (медленно), отдаём LLM только
title + snippet (~300 chars каждый) и просим упорядочить по релевантности.
Crawl4ai запускается только для топ-N отранжированных результатов.

Это "listwise" reranking — LLM видит все кандидаты сразу и возвращает
упорядоченный список индексов. Ответ: ~10-20 токенов.

Ref: "Lost in the Middle" (Liu et al., 2023) — релевантные документы
должны быть в начале контекста, reranking это обеспечивает.
"""

import os

from openai import OpenAI

SCORE_THRESHOLD = float(os.getenv("SEARXNG_SCORE_THRESHOLD", "0.5"))

_RERANK_SYSTEM = (
    "You are a search result ranker. "
    "Given a query and snippets, return the snippet numbers ordered by relevance "
    "to the query, most relevant first. "
    "Return ONLY comma-separated numbers. Example: 3,1,5,2,4"
)


def filter_by_score(
    results: list[dict],
    log=None,
) -> list[dict]:
    """
    Отфильтровать результаты с SearXNG score < SCORE_THRESHOLD.
    Если score не задан (None / 0) — оставляем (движок не возвращает score).
    """
    log = log or (lambda m: None)

    if SCORE_THRESHOLD <= 0:
        return results

    filtered = []
    dropped  = []
    for r in results:
        score = r.get("score")
        # score может быть None (нет данных) или 0.0 (движок не считает)
        if score is not None and score > 0 and score < SCORE_THRESHOLD:
            dropped.append(r)
        else:
            filtered.append(r)

    if dropped:
        log(
            f"  [dim]⊘ score filter (<{SCORE_THRESHOLD}): "
            f"removed {len(dropped)} — "
            + ", ".join(r.get("title", r.get("url", ""))[:40] for r in dropped)
            + "[/dim]"
        )

    return filtered


def llm_rerank(
    query: str,
    results: list[dict],
    client: OpenAI,
    top_n: int = 0,
    log=None,
) -> list[dict]:
    """
    Переупорядочить результаты по релевантности через LLM (snippet-only).

    Returns ALL results in ranked order (no truncation).
    Truncation is handled downstream by the fetch budget.
    top_n is kept for logging only.
    """
    log = log or (lambda m: None)

    if len(results) <= 1:
        return results

    # Собираем блоки title + snippet (не более 300 chars на сниппет)
    lines = []
    for i, r in enumerate(results, 1):
        title   = r.get("title", "").strip()
        snippet = (r.get("snippet") or r.get("content") or "").strip()[:300]
        lines.append(f"[{i}] {title}\n{snippet}")

    prompt = (
        f'Query: "{query}"\n\n'
        + "\n\n".join(lines)
        + "\n\nReturn numbers ordered by relevance, most relevant first:"
    )

    try:
        from llm import MODEL, _extra
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=40,
            extra_body=_extra(),
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log(f"  [yellow]⚠ rerank LLM error: {e} — using original order[/yellow]")
        return results

    # Парсим "3,1,5,2,4" → [2, 0, 4, 1, 3] (0-based)
    indices: list[int] = []
    seen: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(results) and idx not in seen:
                indices.append(idx)
                seen.add(idx)
        except ValueError:
            pass

    # Дополняем пропущенными (на случай если LLM вернул неполный список)
    for i in range(len(results)):
        if i not in seen:
            indices.append(i)

    reranked = [results[i] for i in indices]

    if raw:
        log(
            f"  [dim]↕ rerank: [{raw}] "
            f"→ {len(reranked)} ranked: "
            + ", ".join(f"[{indices[i]+1}]" for i in range(len(reranked)))
            + "[/dim]"
        )

    return reranked
