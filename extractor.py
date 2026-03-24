"""
HTML → clean Markdown via crawl4ai (Playwright-based).

Handles JS-rendered pages (gismeteo, yandex weather, etc.).
No GPU needed.

First run: uv run crawl4ai-setup   (installs Chromium)
"""

import asyncio
import contextlib
import io
import json
import os
import re
from html import unescape

import requests as _requests

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# --------------------------------------------------------------------------- #
#  Reddit — официальный JSON API, без браузера                                #
# --------------------------------------------------------------------------- #

_REDDIT_RE = re.compile(r"reddit\.com/r/\w+/comments/\w+")
_REDDIT_UA = "Mozilla/5.0 (compatible; search-agent/1.0; +https://github.com)"


def _is_reddit(url: str) -> bool:
    return bool(_REDDIT_RE.search(url))


def _extract_reddit(url: str, max_comments: int = 12) -> str:
    """
    Fetches Reddit post + top comments via the public JSON API.
    Converts  https://www.reddit.com/r/sub/comments/id/slug/
           →  https://www.reddit.com/r/sub/comments/id/slug.json?limit=N&sort=top
    No browser, no auth, ~0.5s.
    """
    json_url = re.sub(r"/?(\?.*)?$", ".json?limit=25&sort=top", url)
    resp = _requests.get(
        json_url,
        headers={"User-Agent": _REDDIT_UA},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    parts: list[str] = []

    # ── пост ──────────────────────────────────────────────────────────────── #
    try:
        post = data[0]["data"]["children"][0]["data"]
        title      = post.get("title", "")
        selftext   = (post.get("selftext") or "").strip()
        score      = post.get("score", 0)
        subreddit  = post.get("subreddit", "")
        parts.append(f"# {title}")
        parts.append(f"*r/{subreddit} · {score} upvotes*\n")
        if selftext and selftext not in ("[deleted]", "[removed]"):
            parts.append(selftext[:1500])
    except (IndexError, KeyError):
        pass

    # ── комментарии ───────────────────────────────────────────────────────── #
    try:
        comments = data[1]["data"]["children"]
        parts.append("\n## Top Comments\n")
        count = 0
        for child in comments:
            if child.get("kind") != "t1":
                continue
            c     = child["data"]
            body  = (c.get("body") or "").strip()
            score = c.get("score", 0)
            author = c.get("author", "")
            if body and body not in ("[deleted]", "[removed]"):
                parts.append(f"**{author}** ({score} pts): {body[:600]}\n")
                count += 1
                if count >= max_comments:
                    break
    except (IndexError, KeyError):
        pass

    return "\n".join(parts)

SHALLOW_FETCH_TIMEOUT = int(os.getenv("SHALLOW_FETCH_TIMEOUT", "8"))
_HTTP_UA = "Mozilla/5.0 (compatible; search-agent/1.0; +https://github.com)"
EXTRACT_MAX_CHARS = int(os.getenv("EXTRACT_MAX_CHARS", "3000"))
CRAWL4AI_TIMEOUT = int(os.getenv("CRAWL4AI_TIMEOUT", "25"))


def _clean_html_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_first(pattern: str, html: str, flags: int = 0) -> str:
    match = re.search(pattern, html, flags)
    if not match:
        return ""
    return _clean_html_text(match.group(1))


def _extract_all(pattern: str, html: str, limit: int = 5, flags: int = 0) -> list[str]:
    items: list[str] = []
    for match in re.finditer(pattern, html, flags):
        text = _clean_html_text(match.group(1))
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _extract_meta_content(html: str, keys: list[str]) -> str:
    for key in keys:
        pattern = (
            r'<meta[^>]+(?:name|property)=["\']%s["\'][^>]+content=["\']([^"\']+)["\'][^>]*>'
            % re.escape(key)
        )
        value = _extract_first(pattern, html, flags=re.IGNORECASE)
        if value:
            return value
        reverse_pattern = (
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\']%s["\'][^>]*>'
            % re.escape(key)
        )
        value = _extract_first(reverse_pattern, html, flags=re.IGNORECASE)
        if value:
            return value
    return ""


def _extract_schema_org(html: str) -> dict:
    collected: dict[str, object] = {}
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        queue = data if isinstance(data, list) else [data]
        while queue:
            item = queue.pop(0)
            if not isinstance(item, dict):
                continue
            graph_items = item.get("@graph")
            if isinstance(graph_items, list):
                queue.extend(graph_items)
            if "@type" in item and "@type" not in collected:
                collected["@type"] = item["@type"]
            if "headline" in item and "headline" not in collected:
                collected["headline"] = item["headline"]
            if "datePublished" in item and "datePublished" not in collected:
                collected["datePublished"] = item["datePublished"]
            if "dateModified" in item and "dateModified" not in collected:
                collected["dateModified"] = item["dateModified"]
            author = item.get("author")
            if isinstance(author, dict) and author.get("name") and "author" not in collected:
                collected["author"] = author["name"]
            elif isinstance(author, list) and author and "author" not in collected:
                names = [
                    sub.get("name")
                    for sub in author
                    if isinstance(sub, dict) and sub.get("name")
                ]
                if names:
                    collected["author"] = names[:3]
    return collected


def shallow_fetch(url: str, log=None) -> dict:
    """Fetch lightweight page signals without browser rendering."""
    log = log or (lambda msg: None)

    short = url[:72] + "..." if len(url) > 72 else url
    log(f"    [dim]~ shallow[/dim] [dim]{short}[/dim]")

    if _is_reddit(url):
        try:
            text = _extract_reddit(url)
        except Exception as exc:
            log(f"    [yellow]  x shallow reddit error: {exc}[/yellow]")
            return {}
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0].lstrip("# ").strip() if lines else url
        body_lines = [line for line in lines[1:] if not line.startswith("*r/")]
        paragraphs = body_lines[:3]
        summary = " ".join(paragraphs[:2])[:EXTRACT_MAX_CHARS]
        return {
            "final_url": url,
            "title": title,
            "meta_description": None,
            "headings": [],
            "first_paragraphs": paragraphs,
            "author": None,
            "published_at": None,
            "schema_org": {},
            "content": summary,
        }

    try:
        response = _requests.get(
            url,
            headers={"User-Agent": _HTTP_UA},
            timeout=SHALLOW_FETCH_TIMEOUT,
        )
        response.raise_for_status()
        html = response.text[:250000]
    except Exception as exc:
        log(f"    [yellow]  x shallow fetch error: {exc}[/yellow]")
        return {}

    title = _extract_first(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    meta_description = _extract_meta_content(html, ["description", "og:description", "twitter:description"]) or None
    headings = _extract_all(r"<h[12][^>]*>(.*?)</h[12]>", html, limit=6, flags=re.IGNORECASE | re.DOTALL)
    paragraphs = [
        text
        for text in _extract_all(r"<p[^>]*>(.*?)</p>", html, limit=8, flags=re.IGNORECASE | re.DOTALL)
        if len(text) >= 40
    ][:3]
    schema_org = _extract_schema_org(html)
    author = (
        _extract_meta_content(html, ["author", "article:author", "parsely-author"])
        or str(schema_org.get("author", ""))
        or None
    )
    published_at = (
        _extract_meta_content(
            html,
            ["article:published_time", "og:article:published_time", "datePublished", "pubdate", "dc.date"],
        )
        or str(schema_org.get("datePublished", ""))
        or None
    )

    summary_parts = [title]
    if meta_description:
        summary_parts.append(meta_description)
    summary_parts.extend(headings[:3])
    summary_parts.extend(paragraphs[:2])
    summary = re.sub(r"\s+", " ", " ".join(part for part in summary_parts if part)).strip()[:EXTRACT_MAX_CHARS]

    return {
        "final_url": response.url,
        "title": title or url,
        "meta_description": meta_description,
        "headings": headings,
        "first_paragraphs": paragraphs,
        "author": author,
        "published_at": published_at,
        "schema_org": schema_org,
        "content": summary,
    }

_RUN_CONFIG = CrawlerRunConfig(
    page_timeout=CRAWL4AI_TIMEOUT * 1000,
    wait_until="domcontentloaded",     # networkidle вызывает таймаут на антибот-сайтах
    delay_before_return_html=2.0,      # даём JS время отрисовать данные
    magic=True,                        # stealth: рандомизирует UA, скрывает автоматизацию
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.35,            # 0.45 было слишком агрессивно — резал погоду
            threshold_type="fixed",
        ),
    ),
    excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
    remove_overlay_elements=True,
    word_count_threshold=5,            # было 8 — yandex.kz терял короткие блоки
)

# Persistent crawler — создаётся один раз, не пересоздаётся на каждый URL.
_crawler: AsyncWebCrawler | None = None


async def _ensure_crawler() -> AsyncWebCrawler:
    global _crawler
    if _crawler is None:
        _crawler = AsyncWebCrawler()
        await _crawler.start()
    return _crawler


async def _extract_async(url: str) -> str:
    crawler = await _ensure_crawler()
    # asyncio.wait_for — hard Python-level timeout поверх page_timeout Playwright.
    # page_timeout иногда не срабатывает (magic=True, антибот-редиректы).
    result = await asyncio.wait_for(
        crawler.arun(url=url, config=_RUN_CONFIG),
        timeout=CRAWL4AI_TIMEOUT,
    )
    if not result.success:
        return ""
    # crawl4ai 0.8+ — fit_markdown переехал в result.markdown.fit_markdown
    md = result.markdown
    if not md:
        return ""
    # fit_markdown — очищенный от шума, raw_markdown — fallback если фильтр срезал всё
    text = md.fit_markdown or md.raw_markdown or ""
    return text[:EXTRACT_MAX_CHARS].strip()


def fetch_and_extract(url: str, log=None) -> str:
    """Fetch URL and return clean Markdown.

    Reddit  → public JSON API  (fast, no browser)
    Others  → crawl4ai Playwright
    """
    log = log or (lambda msg: None)

    short = url[:72] + "…" if len(url) > 72 else url
    log(f"    [dim]↓ fetch[/dim]  [dim]{short}[/dim]")

    # ── Reddit: JSON API, без браузера ────────────────────────────────────── #
    if _is_reddit(url):
        log("    [dim]⚙ reddit JSON API…[/dim]")
        try:
            text = _extract_reddit(url)
            if text:
                log(f"    [green]  ✓ {len(text)} chars (reddit)[/green]")
            else:
                log("    [yellow]  ✗ empty response[/yellow]")
            return text[:EXTRACT_MAX_CHARS]
        except Exception as e:
            log(f"    [red]  ✗ reddit error: {e}[/red]")
            return ""

    log(f"    [dim]⚙ crawl4ai extracting…[/dim]")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            text = loop.run_until_complete(_extract_async(url))
        if text:
            log(f"    [green]  ✓ {len(text)} chars[/green]")
        else:
            log("    [yellow]  ✗ empty response[/yellow]")
        return text
    except asyncio.TimeoutError:
        log(f"    [yellow]  ✗ timeout ({CRAWL4AI_TIMEOUT}s)[/yellow]")
        return ""
    except Exception as e:
        log(f"    [red]  ✗ crawl4ai error: {e}[/red]")
        return ""


def _kill_playwright_node() -> None:
    """
    Явно убивает дочерний Node.js playwright-driver процесс (Windows).

    Проблема: crawler.stop() закрывает браузер, но сам Node.js driver-процесс
    остаётся живым и пытается отправить событие browserContextClosed в уже
    закрытый pipe → EPIPE crash в stderr.
    Решение: найти node.exe дочерние процессы текущего PID и kill /F.
    """
    import subprocess
    current_pid = os.getpid()
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             f"ParentProcessId={current_pid} and Name='node.exe'",
             "get", "ProcessId"],
            capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.splitlines():
            pid_str = line.strip()
            if pid_str.isdigit():
                subprocess.run(
                    ["taskkill", "/F", "/PID", pid_str],
                    capture_output=True, timeout=2,
                )
    except Exception:
        pass


def shutdown() -> None:
    """
    Gracefully stop the persistent Playwright browser on exit.

    Sequence:
    1. crawler.stop()        — закрывает браузер (CDP команда)
    2. _kill_playwright_node — явно убиваем node.exe дочерний процесс
    3. cancel pending tasks  — чистим asyncio очередь
    """
    global _crawler
    if _crawler is None:
        return

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            _crawler = None
            return

        # 1. Останавливаем браузер
        loop.run_until_complete(_crawler.stop())

        # 2. Убиваем Node.js playwright-driver до того как os._exit закроет pipe
        _kill_playwright_node()

        # 3. Отменяем оставшиеся pending задачи
        pending = asyncio.all_tasks(loop)
        if pending:
            for task in pending:
                task.cancel()
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )

    except Exception:
        pass
    finally:
        _crawler = None


def get_extractor_name() -> str:
    return "crawl4ai"
