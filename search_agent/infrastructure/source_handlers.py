"""
Unified routing of URL → fetch strategy for shallow payloads and article plaintext.

Handlers are tried **in order**. First handler whose :meth:`supports` is true owns the URL;
:meth:`fetch_shallow` may return ``None`` to fall through to the next handler (e.g. Wikipedia
REST miss → generic HTML). Generic HTTP + trafilatura stays in :mod:`extractor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable
from urllib.parse import urlparse, urlunparse

import requests as _requests

from search_agent.infrastructure.scholarly_sources import (
    arxiv_shallow_payload,
    crossref_shallow_payload,
    github_shallow_payload,
    parse_arxiv_id_from_url,
    parse_doi_from_url,
    parse_github_repo_url,
    parse_semanticscholar_paper_id,
    scholarly_plaintext,
    semantic_scholar_shallow_payload,
)
from search_agent.infrastructure.wikipedia_api import (
    parse_wikipedia_article_url,
    wikipedia_extract_plaintext,
    wikipedia_shallow_payload,
)
from search_agent.settings import get_settings

LogFn = Callable[[str], None]


# --------------------------------------------------------------------------- #
#  Reddit — public JSON API                                                    #
# --------------------------------------------------------------------------- #

_REDDIT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def is_reddit_post_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.netloc or "").lower().split("@")[-1].split(":")[0]
    if host.startswith(("www.", "m.")):
        host = host.split(".", 1)[1]
    if host != "reddit.com":
        return False
    parts = [part for part in parsed.path.split("/") if part]
    return len(parts) >= 4 and parts[0] == "r" and parts[2] == "comments" and bool(parts[3])


def _reddit_json_url(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    if not path.endswith(".json"):
        path += ".json"
    return urlunparse(parsed._replace(path=path, query="limit=25&sort=top", fragment=""))


def extract_reddit_text(url: str, max_comments: int = 12) -> str:
    """Fetch Reddit post + top comments via ``.json`` API."""
    json_url = _reddit_json_url(url)
    resp = _requests.get(json_url, headers=_REDDIT_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    parts: list[str] = []

    try:
        post = data[0]["data"]["children"][0]["data"]
        title = post.get("title", "")
        selftext = (post.get("selftext") or "").strip()
        score = post.get("score", 0)
        subreddit = post.get("subreddit", "")
        parts.append(f"# {title}")
        parts.append(f"*r/{subreddit} · {score} upvotes*\n")
        if selftext and selftext not in ("[deleted]", "[removed]"):
            parts.append(selftext[:1500])
    except (IndexError, KeyError):
        pass

    try:
        comments = data[1]["data"]["children"]
        parts.append("\n## Top Comments\n")
        count = 0
        for child in comments:
            if child.get("kind") != "t1":
                continue
            c = child["data"]
            body = (c.get("body") or "").strip()
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


@runtime_checkable
class ShallowSourceHandler(Protocol):
    """Selects fetch strategy from URL before generic HTTP scraping."""

    id: str

    def supports(self, url: str) -> bool:
        ...

    def fetch_shallow(
        self,
        url: str,
        *,
        max_chars: int,
        timeout: float,
        log: LogFn,
    ) -> dict | None:
        """Return a shallow payload dict, ``{}`` if handled but empty, or ``None`` to try next."""


@runtime_checkable
class ArticlePlaintextHandler(Protocol):
    """Optional fast path for article body text (used before trafilatura in HTTP pipeline)."""

    id: str

    def supports_plaintext(self, url: str) -> bool:
        ...

    def fetch_plaintext(self, url: str, *, timeout: float) -> str | None:
        """Non-empty string to use as body; ``None`` to fall through."""


@dataclass(frozen=True, slots=True)
class RedditSourceHandler:
    """Reddit thread JSON API."""

    id: str = "reddit"

    def supports(self, url: str) -> bool:
        return is_reddit_post_url(url)

    def fetch_shallow(
        self,
        url: str,
        *,
        max_chars: int,
        timeout: float,
        log: LogFn,
    ) -> dict | None:
        if not self.supports(url):
            return None
        try:
            text = extract_reddit_text(url)
        except Exception as exc:
            log(f"    [yellow]  x shallow reddit error: {exc}[/yellow]")
            return {}
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0].lstrip("# ").strip() if lines else url
        body_lines = [line for line in lines[1:] if not line.startswith("*r/")]
        paragraphs = body_lines[:3]
        summary = " ".join(paragraphs[:2])[:max_chars]
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


@dataclass(frozen=True, slots=True)
class WikipediaSourceHandler:
    """MediaWiki REST summary API — lead text, no HTML scrape."""

    id: str = "wikipedia"

    def supports(self, url: str) -> bool:
        return parse_wikipedia_article_url(url) is not None

    def fetch_shallow(
        self,
        url: str,
        *,
        max_chars: int,
        timeout: float,
        log: LogFn,
    ) -> dict | None:
        if not self.supports(url):
            return None
        return wikipedia_shallow_payload(url, max_chars=max_chars, timeout=timeout, log=log)

    def supports_plaintext(self, url: str) -> bool:
        return self.supports(url)

    def fetch_plaintext(self, url: str, *, timeout: float) -> str | None:
        if not self.supports_plaintext(url):
            return None
        return wikipedia_extract_plaintext(url, timeout=timeout)


@dataclass(frozen=True, slots=True)
class ArxivSourceHandler:
    id: str = "arxiv"

    def supports(self, url: str) -> bool:
        return parse_arxiv_id_from_url(url) is not None

    def fetch_shallow(
        self,
        url: str,
        *,
        max_chars: int,
        timeout: float,
        log: LogFn,
    ) -> dict | None:
        if not self.supports(url):
            return None
        return arxiv_shallow_payload(url, max_chars=max_chars, timeout=timeout, log=log)


@dataclass(frozen=True, slots=True)
class CrossrefSourceHandler:
    id: str = "crossref"

    def supports(self, url: str) -> bool:
        return parse_doi_from_url(url) is not None

    def fetch_shallow(
        self,
        url: str,
        *,
        max_chars: int,
        timeout: float,
        log: LogFn,
    ) -> dict | None:
        if not self.supports(url):
            return None
        return crossref_shallow_payload(url, max_chars=max_chars, timeout=timeout, log=log)


@dataclass(frozen=True, slots=True)
class SemanticScholarSourceHandler:
    id: str = "semantic_scholar"

    def supports(self, url: str) -> bool:
        u = url.lower()
        return "semanticscholar.org" in u and parse_semanticscholar_paper_id(url) is not None

    def fetch_shallow(
        self,
        url: str,
        *,
        max_chars: int,
        timeout: float,
        log: LogFn,
    ) -> dict | None:
        if not self.supports(url):
            return None
        return semantic_scholar_shallow_payload(url, max_chars=max_chars, timeout=timeout, log=log)


@dataclass(frozen=True, slots=True)
class GitHubSourceHandler:
    id: str = "github"

    def supports(self, url: str) -> bool:
        return parse_github_repo_url(url) is not None

    def fetch_shallow(
        self,
        url: str,
        *,
        max_chars: int,
        timeout: float,
        log: LogFn,
    ) -> dict | None:
        if not self.supports(url):
            return None
        token = (get_settings().github_token or "").strip() or None
        return github_shallow_payload(
            url,
            max_chars=max_chars,
            timeout=timeout,
            log=log,
            github_token=token,
        )


@dataclass(frozen=True, slots=True)
class ScholarlyApiPlaintextHandler:
    """arXiv / Crossref / Semantic Scholar / GitHub — same order as shallow APIs."""

    id: str = "scholarly_api"

    def supports_plaintext(self, url: str) -> bool:
        u = url.lower()
        return (
            parse_arxiv_id_from_url(url) is not None
            or parse_doi_from_url(url) is not None
            or ("semanticscholar.org" in u and parse_semanticscholar_paper_id(url) is not None)
            or parse_github_repo_url(url) is not None
        )

    def fetch_plaintext(self, url: str, *, timeout: float) -> str | None:
        if not self.supports_plaintext(url):
            return None
        token = (get_settings().github_token or "").strip() or None
        return scholarly_plaintext(url, timeout=timeout, github_token=token)


SHALLOW_SOURCE_HANDLERS: tuple[ShallowSourceHandler, ...] = (
    RedditSourceHandler(),
    # Wikipedia URLs go through the normal trafilatura path — the REST summary API
    # returned only the short lead section (~500–900 chars) which is less content
    # than trafilatura extracts from the full page (up to EXTRACT_MAX_CHARS=4000).
    ArxivSourceHandler(),
    CrossrefSourceHandler(),
    SemanticScholarSourceHandler(),
    GitHubSourceHandler(),
)

ARTICLE_PLAINTEXT_HANDLERS: tuple[ArticlePlaintextHandler, ...] = (
    ScholarlyApiPlaintextHandler(),
)


def dispatch_shallow_fetch(
    url: str,
    *,
    max_chars: int,
    timeout: float,
    log: LogFn,
) -> dict | None:
    """
    Run registered shallow handlers in order.
    Returns a dict when a handler returns non-``None``; ``None`` → use generic HTTP in extractor.
    """
    for handler in SHALLOW_SOURCE_HANDLERS:
        if not handler.supports(url):
            continue
        result = handler.fetch_shallow(url, max_chars=max_chars, timeout=timeout, log=log)
        if result is not None:
            return result
    return None


def dispatch_article_plaintext(url: str, *, timeout: float) -> str | None:
    """Return article text from a specialized handler, or ``None`` for generic HTTP extraction."""
    for handler in ARTICLE_PLAINTEXT_HANDLERS:
        if not handler.supports_plaintext(url):
            continue
        text = handler.fetch_plaintext(url, timeout=timeout)
        if text:
            return text
    return None
