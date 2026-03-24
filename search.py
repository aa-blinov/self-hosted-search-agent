"""
Search: SearXNG → ReaderLM-v2 full-text extraction.

Uses SearchProfile to configure categories, engines, language, time_range, fetch_top_n.
SearXNG must be running: docker compose up -d
"""

import os
import re
import time
import xml.etree.ElementTree as ET

import requests

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")

SearchResult = dict  # {title: str, url: str, snippet: str}


def search_web(
    query: str,
    profile,  # SearchProfile
    log=None,
) -> tuple[list[SearchResult], list[str]]:
    """
    SearXNG search with profile settings → full-text extraction via ReaderLM-v2.

    Returns:
        (results, suggestions)
    """
    from extractor import fetch_and_extract

    _noop = lambda msg: None
    log = log or _noop

    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "categories": ",".join(profile.categories),
        "language": profile.language,
    }
    if profile.engines:
        params["engines"] = ",".join(profile.engines)
    if profile.time_range:
        params["time_range"] = profile.time_range

    log(f"  [cyan]SearXNG[/cyan]  [italic]\"{query}\"[/italic]  "
        f"[dim]({'+'.join(profile.categories)}, lang={profile.language})[/dim]")
    try:
        resp = requests.get(f"{SEARXNG_URL}/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to SearXNG at {SEARXNG_URL}. Run: docker compose up -d"
        )

    suggestions: list[str] = data.get("suggestions", [])

    raw = [r for r in data.get("results", []) if r.get("url")]
    raw_sorted = sorted(raw, key=lambda r: r.get("score", 0), reverse=True)
    raw_sorted = raw_sorted[:profile.max_results]

    log(f"  [dim]→ {len(raw_sorted)} results"
        + (f", suggestions: {suggestions[:3]}" if suggestions else "") + "[/dim]")

    results = []
    for i, r in enumerate(raw_sorted):
        url = r.get("url", "")
        snippet = r.get("content", "").strip()
        if i < profile.fetch_top_n and url:
            full_text = fetch_and_extract(url, log=log)
            # используем полный текст только если он длиннее сниппета SearXNG
            if full_text and len(full_text) > len(snippet):
                snippet = full_text
        if not snippet:
            continue
        results.append({"title": r.get("title", ""), "url": url, "snippet": snippet})

    return results, suggestions


# --------------------------------------------------------------------------- #
#  arXiv — pipeline research only                                             #
# --------------------------------------------------------------------------- #

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
            a.find(f"{{{ARXIV_NS}}}name").text
            for a in entry.findall(f"{{{ARXIV_NS}}}author")
            if a.find(f"{{{ARXIV_NS}}}name") is not None
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
            params={"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"},
            timeout=15,
        )
        resp.raise_for_status()
        return _parse_arxiv(resp.text)
    except Exception as e:
        print(f"[arxiv error] {e}")
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
    for q in _RAG_QUERIES:
        for p in search_arxiv(q, max_results=max_per_query):
            if p["url"] not in seen:
                seen.add(p["url"])
                papers.append(p)
        time.sleep(1)
    return papers
