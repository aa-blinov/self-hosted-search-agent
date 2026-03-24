"""
HTML → clean Markdown via crawl4ai (Playwright-based).

Handles JS-rendered pages (gismeteo, yandex weather, etc.).
No GPU needed.

First run: uv run crawl4ai-setup   (installs Chromium)
"""

import asyncio
import os
import re

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

EXTRACT_MAX_CHARS = int(os.getenv("EXTRACT_MAX_CHARS", "3000"))
CRAWL4AI_TIMEOUT = int(os.getenv("CRAWL4AI_TIMEOUT", "25"))

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


def shutdown() -> None:
    """
    Gracefully stop the persistent Playwright browser on exit.

    Sequence:
    1. crawler.stop()  — закрывает браузер, отправляет EOF в Node.js driver stdin
    2. asyncio.sleep   — даём Node.js driver-процессу время завершиться (~400ms)
    3. cancel tasks    — чистим pending asyncio задачи
    После возврата caller вызывает os._exit(0), минуя Python GC/__del__.

    Без шага 2: os._exit закрывает pipe к Node.js раньше чем тот успевает выйти
    → Node.js падает с "EPIPE: broken pipe" в stderr.
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

        # 2. Ждём пока Node.js playwright-driver получит EOF и завершится
        loop.run_until_complete(asyncio.sleep(0.4))

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
