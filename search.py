"""
SearXNG search helpers.

Two layers are exposed:
  1. `search_searxng()` returns a structured SERP snapshot.
  2. `search_web()` keeps the legacy flow (score filter -> rerank -> fetch).

The new agent pipeline uses `search_searxng()` directly so source gating and
claim-level verification can happen before deep crawling.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from agent_types import SearchSnapshot, SerpResult

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
SEARCH_BACKEND_FALLBACK_DELAY_SEC = float(os.getenv("SEARCH_BACKEND_FALLBACK_DELAY_SEC", "1.0"))

SearchResult = dict  # {title: str, url: str, snippet: str, score: float}

_TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "gclid", "fbclid", "ref", "ref_src", "ref_url", "source",
}


def _canonicalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_QUERY_KEYS
        ]
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/") or "/"
        return urlunparse(
            (
                parsed.scheme.lower() or "https",
                netloc,
                path,
                "",
                urlencode(query),
                "",
            )
        )
    except Exception:
        return url


def _parse_iso_datetime(value: str | None) -> str | None:
    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC).isoformat()
        except ValueError:
            continue
    return None


def _extract_published_at(raw_result: dict) -> str | None:
    for key in ("publishedDate", "published_at", "published", "date", "pubdate"):
        parsed = _parse_iso_datetime(raw_result.get(key))
        if parsed:
            return parsed
    return None


def _build_routed_query(query: str, profile) -> str:
    prefixes = [prefix.strip() for prefix in getattr(profile, "bang_prefixes", []) if prefix and prefix.strip()]
    if not prefixes:
        return query
    return f"{' '.join(prefixes)} {query}".strip()


def _is_backend_degraded(snapshot: SearchSnapshot) -> bool:
    return not snapshot.results and bool(snapshot.unresponsive_engines)


def _simplify_fallback_query(query: str) -> str:
    text = re.sub(r"[\"“”]", " ", query)
    text = re.sub(r"\b(?:OR|AND)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[?]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or query


def _fallback_profiles(profile) -> list:
    from profiles import get_profile

    chain: list[str] = []
    if getattr(profile, "name", "") != "reference":
        chain.append("reference")
    if getattr(profile, "name", "") != "web":
        chain.append("web")
    if getattr(profile, "name", "") != "deep":
        chain.append("deep")
    return [get_profile(name) for name in chain]


def search_searxng(query: str, profile, log=None) -> SearchSnapshot:
    log = log or (lambda msg: None)
    routed_query = _build_routed_query(query, profile)

    params = {
        "q": routed_query,
        "format": "json",
        "pageno": 1,
        "categories": ",".join(profile.categories),
        "language": profile.language,
    }
    if profile.engines:
        params["engines"] = ",".join(profile.engines)
    if profile.time_range:
        params["time_range"] = profile.time_range

    log(
        f"  [cyan]SearXNG[/cyan]  [italic]\"{routed_query}\"[/italic]  "
        f"[dim]({'+'.join(profile.categories)}, lang={profile.language})[/dim]"
    )
    try:
        resp = requests.get(f"{SEARXNG_URL}/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to SearXNG at {SEARXNG_URL}. Run: docker compose up -d"
        )

    suggestions: list[str] = data.get("suggestions", [])
    unresponsive_engines = [
        f"{engine}: {reason}"
        for engine, reason in data.get("unresponsive_engines", [])
        if engine
    ]
    raw_results = [row for row in data.get("results", []) if row.get("url")]
    raw_results = sorted(raw_results, key=lambda row: row.get("score", 0), reverse=True)
    raw_results = raw_results[: profile.max_results]

    log(
        f"  [dim]→ {len(raw_results)} results"
        + (f", suggestions: {suggestions[:3]}" if suggestions else "")
        + (f", unresponsive: {unresponsive_engines[:3]}" if unresponsive_engines else "")
        + "[/dim]"
    )

    results: list[SerpResult] = []
    for idx, raw_result in enumerate(raw_results, 1):
        url = raw_result.get("url", "")
        if not url:
            continue
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        results.append(
            SerpResult(
                result_id=f"serp:{idx}",
                query_variant_id="legacy",
                title=raw_result.get("title", ""),
                url=url,
                snippet=(raw_result.get("content") or "").strip(),
                canonical_url=_canonicalize_url(url),
                host=host,
                position=idx,
                score=raw_result.get("score"),
                engine=raw_result.get("engine"),
                published_at=_extract_published_at(raw_result),
                raw=raw_result,
            )
        )

    return SearchSnapshot(
        query=routed_query,
        suggestions=suggestions,
        results=results,
        retrieved_at=datetime.now(UTC).isoformat(),
        profile_name=getattr(profile, "name", None),
        unresponsive_engines=unresponsive_engines,
    )


def search_searxng_with_fallback(query: str, profile, log=None) -> list[SearchSnapshot]:
    log = log or (lambda msg: None)
    attempts: list[SearchSnapshot] = []
    attempted: set[tuple[str | None, str]] = set()

    def _execute(query_text: str, active_profile):
        key = (getattr(active_profile, "name", None), query_text)
        if key in attempted:
            return None
        attempted.add(key)
        snapshot = search_searxng(query_text, active_profile, log=log)
        attempts.append(snapshot)
        return snapshot

    snapshot = _execute(query, profile)
    if snapshot is None or not _is_backend_degraded(snapshot):
        return attempts

    log("  [yellow]search backend degraded; starting fallback chain[/yellow]")
    fallback_query = _simplify_fallback_query(query)
    plan: list[tuple[str, object]] = []
    if fallback_query != query:
        plan.append((fallback_query, profile))
    for fallback_profile in _fallback_profiles(profile):
        plan.append((fallback_query, fallback_profile))

    for idx, (candidate_query, candidate_profile) in enumerate(plan, 1):
        if SEARCH_BACKEND_FALLBACK_DELAY_SEC > 0:
            time.sleep(SEARCH_BACKEND_FALLBACK_DELAY_SEC)
        log(
            f"  [yellow]fallback {idx}[/yellow] "
            f"[dim]profile={getattr(candidate_profile, 'name', 'unknown')}[/dim]"
        )
        fallback_snapshot = _execute(candidate_query, candidate_profile)
        if fallback_snapshot is None:
            continue
        if fallback_snapshot.results:
            log(
                f"  [green]fallback recovered search results[/green] "
                f"[dim]profile={getattr(candidate_profile, 'name', 'unknown')}[/dim]"
            )
            break
        if not _is_backend_degraded(fallback_snapshot):
            break

    return attempts


def _fetch_with_budget(candidates, fetch_top_n, log):
    """
    Fetch candidates with budget. Budget is consumed only on success.
    """
    from extractor import fetch_and_extract

    results = []
    budget = fetch_top_n

    for result in candidates:
        url = result["url"]
        snippet = result["snippet"]

        if budget > 0 and url:
            full_text = fetch_and_extract(url, log=log)
            if full_text and len(full_text) > len(snippet):
                snippet = full_text

            if snippet:
                budget -= 1
            else:
                log(f"    [dim]· no content, skipping (budget={budget})[/dim]")
                continue

        if snippet:
            results.append({"title": result["title"], "url": url, "snippet": snippet})

    return results


def search_web(
    query: str,
    profile,
    client=None,
    log=None,
) -> tuple[list[SearchResult], list[str]]:
    """
    Legacy flow: SearXNG -> score filter -> snippet rerank -> fetch.

    Kept for compatibility while the new agentic runtime uses `search_searxng()`.
    """
    from reranker import filter_by_score, llm_rerank

    log = log or (lambda msg: None)
    snapshot = search_searxng(query, profile, log=log)
    suggestions = snapshot.suggestions

    candidates: list[SearchResult] = []
    for result in snapshot.results:
        candidates.append({
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "score": result.score,
        })

    candidates = filter_by_score(candidates, log=log)
    if client and len(candidates) > 1:
        candidates = llm_rerank(query, candidates, client, log=log)

    if profile.fetch_top_n > 0:
        results = _fetch_with_budget(candidates, profile.fetch_top_n, log)
    else:
        results = [
            {"title": row["title"], "url": row["url"], "snippet": row["snippet"]}
            for row in candidates
            if row.get("snippet")
        ]

    return results, suggestions


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"


def _parse_arxiv(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        title_el = entry.find(f"{{{ARXIV_NS}}}title")
        summary_el = entry.find(f"{{{ARXIV_NS}}}summary")
        id_el = entry.find(f"{{{ARXIV_NS}}}id")
        authors = [
            author.find(f"{{{ARXIV_NS}}}name").text
            for author in entry.findall(f"{{{ARXIV_NS}}}author")
            if author.find(f"{{{ARXIV_NS}}}name") is not None
        ]
        if title_el is None or summary_el is None:
            continue
        papers.append({
            "title": re.sub(r"\s+", " ", title_el.text or "").strip(),
            "url": (id_el.text or "").strip(),
            "abstract": re.sub(r"\s+", " ", summary_el.text or "").strip(),
            "authors": authors[:3],
        })
    return papers


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    try:
        resp = requests.get(
            ARXIV_API,
            params={
                "search_query": f"all:{query}",
                "max_results": max_results,
                "sortBy": "relevance",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return _parse_arxiv(resp.text)
    except Exception as exc:
        print(f"[arxiv error] {exc}")
        return []


_RAG_QUERIES = [
    "RAG hallucination reduction faithfulness grounded generation",
    "retrieval augmented generation evaluation citation verification",
    "open source web search LLM agent free pipeline",
    "multi-query retrieval reranking evidence verification",
]


def fetch_rag_research(max_per_query: int = 3) -> list[dict]:
    seen: set[str] = set()
    papers = []
    for query in _RAG_QUERIES:
        for paper in search_arxiv(query, max_results=max_per_query):
            if paper["url"] not in seen:
                seen.add(paper["url"])
                papers.append(paper)
        time.sleep(1)
    return papers
