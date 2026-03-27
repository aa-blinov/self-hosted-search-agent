from __future__ import annotations

"""
HTTP fetch: main text via **trafilatura** (fast, no browser) when possible.

**Deep / JS-heavy pages**: crawl4ai (Playwright) when HTTP text is too thin.

First run: uv run crawl4ai-setup   (installs Chromium)
"""

import asyncio
import concurrent.futures
import contextlib
import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests as _requests
from rich.markup import escape as _rich_escape
from trafilatura import extract as _trafilatura_extract
from trafilatura.metadata import extract_metadata as _trafilatura_extract_metadata

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from search_agent import tuning
from search_agent.infrastructure.log_preview import preview_snippet as _preview_snippet
from search_agent.infrastructure.source_handlers import (
    dispatch_article_plaintext,
    dispatch_shallow_fetch,
    extract_reddit_text,
    is_reddit_post_url,
)
from search_agent.settings import get_settings


def _extract_cap(url: str = "", intent: str = "factual") -> int:
    """Return char limit for this URL.

    Authority domains get a higher limit *only* for synthesis queries — factual
    queries keep the default cap so verify_claim receives compact passage sets and
    stays fast.
    """
    base = get_settings().resolved_extract_max_chars()
    if url and intent == "synthesis":
        try:
            host = (urlparse(url).netloc or "").lower().split("@")[-1].split(":")[0]
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            host = ""
        if any(domain in host for domain in tuning.AUTHORITY_DOMAINS):
            return max(base, tuning.EXTRACT_MAX_CHARS_AUTHORITY)
    return base


def _url_shallow_browser_first(url: str) -> bool:
    """Hosts where plain HTTP GET often fails (TLS EOF); use Playwright for shallow fetch."""
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    host = host.split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    for suf in tuning.SHALLOW_BROWSER_FIRST_HOST_SUFFIXES:
        suf = suf.lower().strip()
        if not suf:
            continue
        if host == suf.lstrip("."):
            return True
        if host.endswith(suf):
            return True
    return False


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in (text or "").split("\n")[:8]:
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()[:400]
    return fallback


def fetch_browser_extract_only(url: str, log=None) -> str:
    """Playwright/crawl4ai only (no ``requests``). For hosts that break TLS on scripted HTTP."""
    log = log or (lambda msg: None)
    loop = _extract_event_loop()
    capture = io.StringIO()
    timeout = float(tuning.CRAWL4AI_BROWSER_ONLY_TIMEOUT)
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        try:
            coro = asyncio.wait_for(_extract_async(url), timeout=timeout)
            return loop.run_until_complete(coro)
        except asyncio.TimeoutError:
            log(f"    [yellow]  x browser-only crawl timeout ({timeout:.0f}s)[/yellow]")
            return ""
        except Exception as exc:
            log(f"    [yellow]  x browser-only crawl error: {exc}[/yellow]")
            return ""


def _shallow_payload_from_plain_text(text: str, *, final_url: str, title_hint: str, max_chars: int) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    title = _title_from_markdown(text, title_hint)
    first_paragraphs: list[str] = []
    for block in text.split("\n\n"):
        b = block.strip()
        if len(b) >= 40:
            first_paragraphs.append(b)
        if len(first_paragraphs) >= 3:
            break
    return {
        "final_url": final_url,
        "title": title,
        "meta_description": None,
        "headings": [],
        "first_paragraphs": first_paragraphs,
        "author": None,
        "published_at": None,
        "schema_org": {},
        "content": text[:max_chars],
    }


def _shallow_browser_extract(url: str, log=None) -> dict:
    """Shallow fetch via crawl4ai only (used for vc.ru and similar)."""
    log = log or (lambda msg: None)
    max_chars = _extract_cap(url)
    short = url[:72] + "..." if len(url) > 72 else url
    log(f"    [dim]~ shallow[/dim] [dim]{short}[/dim]")
    log("    [dim]  [cyan]browser-only[/cyan] shallow · skip HTTP (TLS/bot issues)[/dim]")
    text = fetch_browser_extract_only(url, log=log)
    if not text:
        return {}
    payload = _shallow_payload_from_plain_text(
        text,
        final_url=url,
        title_hint=url,
        max_chars=max_chars,
    )
    main = (payload.get("content") or "").strip()
    if main:
        log(
            f"    [dim]  [green]extract[/green] crawl4ai · browser session · "
            f"[cyan]{len(main)}[/] chars · "
            f"[italic]{_rich_escape(_preview_snippet(main))}[/][/dim]"
        )
    return payload


_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
    "Upgrade-Insecure-Requests": "1",
}


def _http_get_shallow(url: str, *, timeout: float) -> tuple[_requests.Response | None, Exception | None]:
    """
    GET with small retry budget for rate limits and flaky TLS/network.
    Does not retry permanent client errors (403, 404, …).
    """
    attempts = max(1, tuning.SHALLOW_FETCH_HTTP_ATTEMPTS)
    backoff = tuning.SHALLOW_FETCH_RETRY_BACKOFF_SEC
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            response.raise_for_status()
            return response, None
        except _requests.HTTPError as exc:
            last_exc = exc
            code = exc.response.status_code if exc.response is not None else 0
            if code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            return None, exc
        except _requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            return None, exc
    return None, last_exc


def _normalize_html_text(text: str) -> str:
    return " ".join(unescape(text or "").split())


@dataclass(slots=True)
class _HTMLSignals:
    title: str = ""
    meta_tags: list[dict[str, str]] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    schema_json_blocks: list[str] = field(default_factory=list)


class _HTMLSignalsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.signals = _HTMLSignals()
        self._capture_tag: str | None = None
        self._capture_depth = 0
        self._capture_parts: list[str] = []
        self._capture_schema = False
        self._schema_depth = 0
        self._schema_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        attr_map = {(key or "").lower(): (value or "") for key, value in attrs}

        if self._capture_tag is not None:
            self._capture_depth += 1
        if self._capture_schema:
            self._schema_depth += 1

        if normalized_tag == "meta":
            self.signals.meta_tags.append(attr_map)

        if normalized_tag in {"title", "h1", "h2", "p"} and self._capture_tag is None:
            self._capture_tag = normalized_tag
            self._capture_depth = 1
            self._capture_parts = []

        if (
            normalized_tag == "script"
            and not self._capture_schema
            and attr_map.get("type", "").strip().lower() == "application/ld+json"
        ):
            self._capture_schema = True
            self._schema_depth = 1
            self._schema_parts = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._capture_tag is not None and data:
            self._capture_parts.append(data)
        if self._capture_schema and data:
            self._schema_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()

        if self._capture_schema:
            self._schema_depth -= 1
            if normalized_tag == "script" and self._schema_depth <= 0:
                raw = "".join(self._schema_parts).strip()
                if raw:
                    self.signals.schema_json_blocks.append(raw)
                self._capture_schema = False
                self._schema_depth = 0
                self._schema_parts = []

        if self._capture_tag is not None:
            self._capture_depth -= 1
            if self._capture_depth <= 0:
                text = _normalize_html_text("".join(self._capture_parts))
                if text:
                    if self._capture_tag == "title" and not self.signals.title:
                        self.signals.title = text
                    elif self._capture_tag in {"h1", "h2"} and text not in self.signals.headings:
                        self.signals.headings.append(text)
                    elif self._capture_tag == "p" and text not in self.signals.paragraphs:
                        self.signals.paragraphs.append(text)
                self._capture_tag = None
                self._capture_depth = 0
                self._capture_parts = []


def _collect_html_signals(html: str) -> _HTMLSignals:
    parser = _HTMLSignalsParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        return _HTMLSignals()
    return parser.signals


def _extract_meta_content(signals: _HTMLSignals, keys: list[str]) -> str:
    expected = {key.lower() for key in keys}
    for meta in signals.meta_tags:
        marker = (meta.get("name") or meta.get("property") or "").strip().lower()
        content = _normalize_html_text(meta.get("content", ""))
        if marker in expected and content:
            return content
    return ""


def _extract_schema_org(signals: _HTMLSignals) -> dict:
    collected: dict[str, object] = {}
    for raw in signals.schema_json_blocks:
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


def _trafilatura_main_text(html: str, final_url: str) -> str | None:
    """Return main article/body text, or None if too thin or extraction failed."""
    text = _trafilatura_extract(
        html,
        url=final_url,
        fast=True,
        include_comments=False,
        include_tables=True,
    )
    if not text:
        return None
    text = text.strip()
    if len(text) < tuning.TRAIFILATURA_MIN_MAIN_CHARS:
        return None
    return text


def _legacy_shallow_payload(html: str, response, max_chars: int) -> dict:
    signals = _collect_html_signals(html)
    title = signals.title
    meta_description = _extract_meta_content(signals, ["description", "og:description", "twitter:description"]) or None
    headings = signals.headings[:6]
    paragraphs = [
        text
        for text in signals.paragraphs[:8]
        if len(text) >= 40
    ][:3]
    schema_org = _extract_schema_org(signals)
    author = (
        _extract_meta_content(signals, ["author", "article:author", "parsely-author"])
        or str(schema_org.get("author", ""))
        or None
    )
    published_at = (
        _extract_meta_content(
            signals,
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
    summary = " ".join(part for part in summary_parts if part).strip()[:max_chars]

    return {
        "final_url": response.url,
        "title": title or response.url,
        "meta_description": meta_description,
        "headings": headings,
        "first_paragraphs": paragraphs,
        "author": author,
        "published_at": published_at,
        "schema_org": schema_org,
        "content": summary,
    }


def shallow_fetch(url: str, log=None, intent: str = "factual") -> dict:
    """Fetch lightweight page signals without browser rendering."""
    log = log or (lambda msg: None)
    max_chars = _extract_cap(url, intent)

    short = url[:72] + "..." if len(url) > 72 else url
    log(f"    [dim]~ shallow[/dim] [dim]{short}[/dim]")

    routed = dispatch_shallow_fetch(
        url,
        max_chars=max_chars,
        timeout=tuning.SHALLOW_FETCH_TIMEOUT,
        log=log,
    )
    if routed is not None:
        return routed

    if _url_shallow_browser_first(url):
        return _shallow_browser_extract(url, log=log)

    response, err = _http_get_shallow(url, timeout=tuning.SHALLOW_FETCH_TIMEOUT)
    if response is None:
        log(f"    [yellow]  x shallow fetch error: {err}[/yellow]")
        return {}
    try:
        # Try strict UTF-8 decode on the raw bytes first.  Most modern sites
        # (including Russian ones that mis-declare windows-1251 or send no charset)
        # actually serve UTF-8.  If the bytes are valid UTF-8 we use that; otherwise
        # we fall back to requests' charset detection (which handles genuine 1-byte
        # encodings like windows-1251, iso-8859-1, etc.).
        raw_bytes = response.content[:250000]
        try:
            html = raw_bytes.decode("utf-8")
        except (UnicodeDecodeError, LookupError):
            if response.encoding:
                declared = response.encoding.lower().replace("-", "").replace("_", "")
                apparent = (response.apparent_encoding or "").lower().replace("-", "").replace("_", "")
                if declared not in ("utf8", "utf16", "utf32") and apparent in ("utf8",):
                    response.encoding = "utf-8"
            html = response.text[:250000]
    except Exception as exc:
        log(f"    [yellow]  x shallow fetch error: {exc}[/yellow]")
        return {}

    main_text = _trafilatura_main_text(html, response.url)
    if main_text:
        log(
            f"    [dim]  [green]extract[/green] trafilatura · main article text · "
            f"[cyan]{len(main_text)}[/] chars · "
            f"[italic]{_rich_escape(_preview_snippet(main_text))}[/][/dim]"
        )
        meta = _trafilatura_extract_metadata(html, default_url=response.url)
        title = ""
        meta_description = None
        author = None
        published_at = None
        if meta is not None:
            title = (getattr(meta, "title", None) or "").strip()
            meta_description = (getattr(meta, "description", None) or "").strip() or None
            raw_author = getattr(meta, "author", None)
            if isinstance(raw_author, list) and raw_author:
                author = str(raw_author[0])
            elif isinstance(raw_author, str) and raw_author.strip():
                author = raw_author.strip()
            raw_date = getattr(meta, "date", None)
            if raw_date is not None:
                published_at = str(raw_date)
        if not title:
            title = _collect_html_signals(html).title or response.url

        first_paragraphs: list[str] = []
        for block in main_text.split("\n\n"):
            b = block.strip()
            if len(b) >= 40:
                first_paragraphs.append(b)
            if len(first_paragraphs) >= 3:
                break

        return {
            "final_url": response.url,
            "title": title,
            "meta_description": meta_description,
            "headings": [],
            "first_paragraphs": first_paragraphs,
            "author": author,
            "published_at": published_at,
            "schema_org": {},
            "content": main_text[:max_chars],
        }

    payload = _legacy_shallow_payload(html, response, max_chars)
    content = (payload.get("content") or "").strip()
    log(
        f"    [dim]  [yellow]extract[/yellow] legacy HTML · title/meta/h1/p summary · "
        f"[cyan]{len(content)}[/] chars · "
        f"[italic]{_rich_escape(_preview_snippet(content))}[/][/dim]"
    )
    return payload


def shallow_fetch_many(
    urls: list[str],
    log=None,
    page_cache: dict[str, dict] | None = None,
    page_cache_lock: threading.Lock | None = None,
    intent: str = "factual",
) -> list[dict]:
    """Parallel HTTP shallow fetches; order matches ``urls``.

    If *page_cache* (+ *page_cache_lock*) are supplied, already-fetched URLs are
    returned from the cache without an HTTP round-trip.  Cache is populated with
    successful (non-empty) results after each fetch.
    """
    if not urls:
        return []
    log = log or (lambda msg: None)

    results: list[dict | None] = [None] * len(urls)
    to_fetch_indices: list[int] = []
    to_fetch_urls: list[str] = []

    if page_cache is not None and page_cache_lock is not None:
        with page_cache_lock:
            for i, url in enumerate(urls):
                if url in page_cache:
                    short = url[:72] + "..." if len(url) > 72 else url
                    log(f"    [dim]~ shallow[/dim] [dim]{short}[/dim] [dim green](page cache hit)[/dim green]")
                    results[i] = page_cache[url]
                else:
                    to_fetch_indices.append(i)
                    to_fetch_urls.append(url)
    else:
        to_fetch_indices = list(range(len(urls)))
        to_fetch_urls = list(urls)

    if to_fetch_urls:
        workers = min(tuning.FETCH_SHALLOW_CONCURRENCY, len(to_fetch_urls))

        def one(u: str) -> dict:
            return shallow_fetch(u, log=log, intent=intent)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fetched = list(ex.map(one, to_fetch_urls))

        for idx, payload in zip(to_fetch_indices, fetched):
            results[idx] = payload

        if page_cache is not None and page_cache_lock is not None:
            with page_cache_lock:
                for url, payload in zip(to_fetch_urls, fetched):
                    if payload:  # don't cache failed fetches
                        page_cache[url] = payload

    return [r if r is not None else {} for r in results]


def _http_article_text(url: str) -> str:
    """HTTP GET + trafilatura main text; fallback to legacy HTML heuristics (no browser)."""
    specialized = dispatch_article_plaintext(url, timeout=tuning.SHALLOW_FETCH_TIMEOUT)
    if specialized:
        return specialized[: _extract_cap(url)]

    response, _err = _http_get_shallow(url, timeout=tuning.SHALLOW_FETCH_TIMEOUT)
    if response is None:
        return ""
    try:
        raw_bytes = response.content[:500000]
        try:
            html = raw_bytes.decode("utf-8")
        except (UnicodeDecodeError, LookupError):
            if response.encoding:
                declared = response.encoding.lower().replace("-", "").replace("_", "")
                apparent = (response.apparent_encoding or "").lower().replace("-", "").replace("_", "")
                if declared not in ("utf8", "utf16", "utf32") and apparent in ("utf8",):
                    response.encoding = "utf-8"
            html = response.text[:500000]
    except Exception:
        return ""

    cap = _extract_cap(url)
    main = _trafilatura_main_text(html, response.url)
    if main:
        return main[:cap]

    signals = _collect_html_signals(html)
    title = signals.title
    meta_description = _extract_meta_content(signals, ["description", "og:description", "twitter:description"]) or ""
    headings = signals.headings[:8]
    paragraphs = [
        text
        for text in signals.paragraphs[:24]
        if len(text) >= 40
    ]
    summary_parts: list[str] = []
    for part in (title, meta_description):
        if part:
            summary_parts.append(part)
    summary_parts.extend(headings[:4])
    summary_parts.extend(paragraphs[:14])
    text = " ".join(summary_parts).strip()
    return text[:cap]


def _build_crawler_config() -> CrawlerRunConfig:
    return CrawlerRunConfig(
        page_timeout=tuning.CRAWL4AI_TIMEOUT * 1000,
        wait_until="domcontentloaded",
        delay_before_return_html=tuning.CRAWL4AI_DELAY_BEFORE_HTML,
        magic=True,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.35,
                threshold_type="fixed",
            ),
        ),
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
        remove_overlay_elements=True,
        word_count_threshold=5,
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
    result = await asyncio.wait_for(
        crawler.arun(url=url, config=_build_crawler_config()),
        timeout=tuning.CRAWL4AI_TIMEOUT,
    )
    if not result.success:
        return ""
    md = result.markdown
    if not md:
        return ""
    if tuning.CRAWL4AI_PREFER_RAW:
        text = md.raw_markdown or md.fit_markdown or ""
    else:
        text = md.fit_markdown or md.raw_markdown or ""
    return text[: _extract_cap(url)].strip()


async def _fetch_url_content(url: str, log=None) -> str:
    log = log or (lambda msg: None)
    min_http = max(200, tuning.FETCH_HTTP_MIN_CHARS)
    if is_reddit_post_url(url):
        try:
            text = await asyncio.wait_for(asyncio.to_thread(extract_reddit_text, url), timeout=20.0)
            return (text or "")[: _extract_cap(url)].strip()
        except Exception:
            return ""
    try:
        text = await asyncio.to_thread(_http_article_text, url)
        text = (text or "").strip()
        if len(text) >= min_http:
            log("    [dim][http+trafilatura/legacy][/dim]")
            return text[: _extract_cap(url)].strip()
    except Exception:
        pass
    return await _extract_async(url)


def _extract_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


async def _fetch_deep_batch(urls: list[str], log) -> list[str]:
    sem = asyncio.Semaphore(min(tuning.FETCH_DEEP_CONCURRENCY, len(urls)))

    async def one(u: str) -> str:
        async with sem:
            short = u[:72] + "..." if len(u) > 72 else u
            log(f"    [dim]fetch[/dim]  [dim]{short}[/dim]")
            if is_reddit_post_url(u):
                log("    [dim][reddit JSON API][/dim]")
            else:
                log("    [dim][crawl4ai][/dim]")
            try:
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                    text = await _fetch_url_content(u, log=log)
                if text:
                    log(f"    [green]  ok {len(text)} chars[/green]")
                else:
                    log("    [yellow]  err empty response[/yellow]")
                return text
            except asyncio.TimeoutError:
                log(f"    [yellow]  err timeout ({tuning.CRAWL4AI_TIMEOUT}s)[/yellow]")
                return ""
            except Exception as e:
                log(f"    [red]  err crawl4ai error: {e}[/red]")
                return ""

    return await asyncio.gather(*[one(u) for u in urls])


def fetch_and_extract_many(urls: list[str], log=None) -> list[str]:
    """Deep fetch many URLs (parallel crawl4ai tasks with semaphore). Order preserved."""
    if not urls:
        return []
    log = log or (lambda msg: None)
    loop = _extract_event_loop()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        return loop.run_until_complete(_fetch_deep_batch(urls, log))


def fetch_and_extract(url: str, log=None) -> str:
    """Fetch URL and return clean Markdown.

    Reddit  → public JSON API  (fast, no browser)
    Others  → HTTP (trafilatura + legacy) if enough text, else crawl4ai Playwright
    """
    return fetch_and_extract_many([url], log=log)[0]


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
    return "trafilatura+crawl4ai"
