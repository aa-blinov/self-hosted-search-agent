"""arXiv helpers for /research CLI (not tied to a specific search backend)."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"

_RAG_QUERIES = [
    "RAG hallucination reduction faithfulness grounded generation",
    "retrieval augmented generation evaluation citation verification",
    "open source web search LLM agent free pipeline",
    "multi-query retrieval reranking evidence verification",
]


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


def fetch_arxiv_paper_by_id(arxiv_id: str, timeout: float = 15) -> dict | None:
    """
    Load a single paper by arXiv id (e.g. ``2312.12345`` or ``cs/9901001``) via the Atom API.
    Returns the same shape as entries from :func:`search_arxiv`, or ``None``.
    """
    arxiv_id = arxiv_id.strip().rstrip(".pdf")
    if not arxiv_id:
        return None
    try:
        resp = requests.get(
            ARXIV_API,
            params={"id_list": arxiv_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        papers = _parse_arxiv(resp.text)
        return papers[0] if papers else None
    except Exception:
        return None


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
