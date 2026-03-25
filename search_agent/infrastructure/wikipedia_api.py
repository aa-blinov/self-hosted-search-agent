"""
Wikipedia article text via official `REST v1` summary API (no HTML scraping).

Falls back to generic HTTP fetch in :mod:`search_agent.infrastructure.extractor` when
the URL is not a ``/wiki/`` article or the API returns 404 / empty extract.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote, unquote, urlparse

import requests
from rich.markup import escape as _rich_escape

from search_agent import tuning
from search_agent.infrastructure.log_preview import preview_snippet

_REST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 "
        "(search-agent; +https://github.com/local/search-agent)"
    ),
    "Accept": "application/json;q=0.9,text/plain;q=0.8,*/*;q=0.1",
}


def parse_wikipedia_article_url(url: str) -> tuple[str, str] | None:
    """
    Parse ``https://{lang}.wikipedia.org/wiki/Title`` (including ``{lang}.m.`` mobile host).
    ``lang`` may contain hyphens (e.g. ``zh-min-nan``). Returns ``(lang, title_plain)`` or ``None``.
    """
    try:
        u = urlparse(url.strip())
    except Exception:
        return None
    host = (u.netloc or "").lower()
    if "@" in host:
        return None
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if not host.endswith(".wikipedia.org"):
        return None
    left = host[: -len(".wikipedia.org")].rstrip(".")
    if not left:
        return None
    if left.endswith(".m"):
        left = left[: -2].rstrip(".")
    lang = left
    if not lang or "." in lang:
        return None
    path = u.path or ""
    if not path.startswith("/wiki/"):
        return None
    raw = path[len("/wiki/") :].rstrip("/")
    if not raw:
        return None
    title = unquote(raw.replace("_", " "))
    return lang, title


def _rest_get_summary_json(lang: str, title: str, *, timeout: float) -> dict[str, Any] | None:
    title_seg = quote(title.replace(" ", "_"), safe="")
    api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title_seg}"
    attempts = max(1, tuning.SHALLOW_FETCH_HTTP_ATTEMPTS)
    backoff = tuning.SHALLOW_FETCH_RETRY_BACKOFF_SEC
    for attempt in range(attempts):
        try:
            r = requests.get(api_url, headers=_REST_HEADERS, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            return None
        except (requests.RequestException, ValueError, TypeError):
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            return None
    return None


def wikipedia_extract_plaintext(url: str, *, timeout: float) -> str | None:
    """Return lead/summary text for a Wikipedia article URL, or ``None``."""
    parsed = parse_wikipedia_article_url(url)
    if parsed is None:
        return None
    lang, title = parsed
    data = _rest_get_summary_json(lang, title, timeout=timeout)
    if not data:
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    return extract


def wikipedia_shallow_payload(url: str, *, max_chars: int, timeout: float, log) -> dict | None:
    """
    Build the same structure as :func:`extractor.shallow_fetch` from REST summary, or ``None``.
    """
    parsed = parse_wikipedia_article_url(url)
    if parsed is None:
        return None
    lang, title = parsed
    data = _rest_get_summary_json(lang, title, timeout=timeout)
    if not data:
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    page_title = (data.get("title") or title).strip() or title
    content = extract[:max_chars]
    first_para = content.split("\n")[0].strip()
    first_paragraphs = [first_para] if first_para else [content[:400]]
    canonical = url
    urls = data.get("content_urls") or {}
    if isinstance(urls, dict):
        desktop = urls.get("desktop") or {}
        if isinstance(desktop, dict):
            page = desktop.get("page")
            if isinstance(page, str) and page.startswith("http"):
                canonical = page

    log(
        "    [dim]  [green]extract[/green] wikipedia REST · REST v1 page/summary (lead extract) · "
        f"[cyan]{len(content)}[/] chars · [italic]{_rich_escape(preview_snippet(content))}[/][/dim]"
    )
    return {
        "final_url": canonical,
        "title": page_title,
        "meta_description": (data.get("description") or None) if isinstance(data.get("description"), str) else None,
        "headings": [],
        "first_paragraphs": first_paragraphs,
        "author": None,
        "published_at": None,
        "schema_org": {},
        "content": content,
    }
