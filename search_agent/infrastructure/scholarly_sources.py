"""
Structured APIs for academic / code hosts (URL-detected): arXiv, Crossref, Semantic Scholar, GitHub.

Used by :mod:`search_agent.infrastructure.source_handlers` before generic HTTP scraping.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from urllib.parse import quote, unquote, urlparse

import requests
from rich.markup import escape as _rich_escape

from search_agent.infrastructure.arxiv_research import fetch_arxiv_paper_by_id
from search_agent.infrastructure.log_preview import preview_snippet

LogFn = Callable[[str], None]

_API_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 "
    "(search-agent; +https://github.com/local/search-agent)"
)

_GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/([^/]+)/([^/?#]+)",
    re.I,
)
_GITHUB_RESERVED = frozenset(
    {
        "topics",
        "features",
        "explore",
        "marketplace",
        "pricing",
        "sponsors",
        "settings",
        "login",
        "signup",
        "collections",
        "trending",
    },
)

_ARXIV_PATH_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([^?#]+)",
    re.I,
)


_S2_HEX_RE = re.compile(r"/([a-f0-9]{40})(?:/|$|\?)", re.I)
_S2_ARXIV_IN_PATH_RE = re.compile(r"arxiv[:\s]+([\d.]+)(?:v\d+)?", re.I)


def parse_github_repo_url(url: str) -> tuple[str, str] | None:
    """``https://github.com/owner/repo`` or ``.../tree/...`` — owner/repo only."""
    m = _GITHUB_REPO_RE.match(url.strip())
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    if owner.lower() in _GITHUB_RESERVED or repo.lower() in _GITHUB_RESERVED:
        return None
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def parse_arxiv_id_from_url(url: str) -> str | None:
    m = _ARXIV_PATH_RE.search(url)
    if not m:
        return None
    aid = unquote(m.group(1).strip()).rstrip("/")
    if aid.lower().endswith(".pdf"):
        aid = aid[:-4]
    return aid or None


def parse_doi_from_url(url: str) -> str | None:
    u = url.strip()
    m = re.search(r"doi\.org/([^?\s#]+)", u, re.I)
    if not m:
        return None
    raw = unquote(m.group(1).strip().rstrip("/"))
    if "?" in raw:
        raw = raw.split("?")[0]
    if raw.lower().startswith("10."):
        return raw
    return None


def parse_semanticscholar_paper_id(url: str) -> str | None:
    """Corpus id (40 hex) or ``ARXIV:…`` / ``DOI:…`` for the Graph API."""
    try:
        path = unquote(urlparse(url.strip()).path or "")
    except Exception:
        return None
    m = _S2_HEX_RE.search(path)
    if m:
        return m.group(1)
    marx = _S2_ARXIV_IN_PATH_RE.search(path)
    if marx:
        return "ARXIV:" + marx.group(1)
    mdoi = re.search(r"DOI[:\s]+(10\.[^\s/]+/[^\s?#]+)", path, re.I)
    if mdoi:
        return "DOI:" + mdoi.group(1)
    return None


def _strip_xmlish(text: str) -> str:
    t = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", t).strip()


def _payload(
    *,
    final_url: str,
    title: str,
    body: str,
    max_chars: int,
    meta_description: str | None = None,
) -> dict:
    text = (body or "").strip()[:max_chars]
    first = text.split("\n\n")[0].strip() if text else ""
    if not first and text:
        first = text[:800]
    return {
        "final_url": final_url,
        "title": title[:500] or final_url,
        "meta_description": meta_description,
        "headings": [],
        "first_paragraphs": [first] if first else [],
        "author": None,
        "published_at": None,
        "schema_org": {},
        "content": text,
    }


def fetch_github_readme_and_release(
    owner: str,
    repo: str,
    *,
    token: str | None,
    timeout: float,
) -> tuple[str, str, str] | None:
    """
    Returns ``(title, html_url, combined_text)`` or ``None`` if nothing usable.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _API_UA,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token and token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"

    parts: list[str] = []
    title = f"{owner}/{repo}"
    page_url = f"https://github.com/{owner}/{repo}"

    r_readme = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/readme",
        headers=headers,
        timeout=timeout,
    )
    if r_readme.status_code == 200:
        data = r_readme.json()
        raw_b64 = data.get("content") or ""
        try:
            body = base64.b64decode(raw_b64).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if body.strip():
            parts.append("# README\n\n" + body.strip())

    r_rel = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/releases/latest",
        headers=headers,
        timeout=timeout,
    )
    if r_rel.status_code == 200:
        rj = r_rel.json()
        tag = (rj.get("tag_name") or "").strip()
        body_rel = (rj.get("body") or "").strip()
        if tag or body_rel:
            parts.append(f"# Latest release ({tag})\n\n{body_rel[:12000]}")

    if not parts:
        return None
    combined = "\n\n---\n\n".join(parts)
    return title, page_url, combined


def fetch_crossref_work(doi: str, *, timeout: float) -> tuple[str, str, str] | None:
    """Returns ``(title, landing_url, text)``."""
    enc = quote(doi, safe="")
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{enc}",
            headers={"User-Agent": _API_UA, "Accept": "application/json"},
            timeout=timeout,
        )
        r.raise_for_status()
        msg = (r.json() or {}).get("message") or {}
    except Exception:
        return None

    titles = msg.get("title")
    title = titles[0] if isinstance(titles, list) and titles else (titles or "Crossref work")
    abstract = _strip_xmlish(str(msg.get("abstract") or ""))
    if not abstract:
        parts = []
        if msg.get("container-title"):
            ct = msg["container-title"]
            parts.append(str(ct[0] if isinstance(ct, list) else ct))
        if msg.get("issued", {}).get("date-parts"):
            parts.append(str(msg["issued"]["date-parts"][0]))
        abstract = " — ".join(parts) if parts else ""

    url = f"https://doi.org/{doi}"
    authors = msg.get("author") or []
    names: list[str] = []
    for a in authors[:8]:
        if isinstance(a, dict):
            fam = a.get("family", "")
            giv = a.get("given", "")
            names.append(f"{giv} {fam}".strip() or fam or str(a))
    head = ", ".join(names) if names else ""
    text = f"# {title}\n\n"
    if head:
        text += f"Authors: {head}\n\n"
    text += abstract
    return title.strip(), url, text.strip()


def fetch_semantic_scholar_paper(paper_id: str, *, timeout: float) -> tuple[str, str, str] | None:
    """Graph API: corpus id, ``ARXIV:…``, ``DOI:…``, etc."""
    enc = quote(paper_id, safe="")
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{enc}",
            params={"fields": "title,abstract,year,authors,url"},
            headers={"User-Agent": _API_UA},
            timeout=timeout,
        )
        r.raise_for_status()
        p = r.json() or {}
    except Exception:
        return None

    title = (p.get("title") or "Semantic Scholar paper").strip()
    abstract = (p.get("abstract") or "").strip()
    year = p.get("year")
    authors = p.get("authors") or []
    names = []
    for a in authors[:12]:
        if isinstance(a, dict) and a.get("name"):
            names.append(str(a["name"]))
    url = (p.get("url") or "").strip() or f"https://www.semanticscholar.org/paper/{paper_id}"
    text = f"# {title}\n\n"
    if year:
        text += f"Year: {year}\n\n"
    if names:
        text += "Authors: " + ", ".join(names) + "\n\n"
    text += abstract
    return title, url, text.strip()


def github_shallow_payload(
    url: str,
    *,
    max_chars: int,
    timeout: float,
    log: LogFn,
    github_token: str | None,
) -> dict | None:
    parsed = parse_github_repo_url(url)
    if parsed is None:
        return None
    owner, repo = parsed
    got = fetch_github_readme_and_release(owner, repo, token=github_token, timeout=timeout)
    if got is None:
        return None
    title, page_url, body = got
    log(
        "    [dim]  [green]extract[/green] github API · README + latest release notes · "
        f"[cyan]{len(body)}[/] chars · [italic]{_rich_escape(preview_snippet(body))}[/][/dim]"
    )
    return _payload(final_url=page_url, title=title, body=body, max_chars=max_chars)


def arxiv_shallow_payload(url: str, *, max_chars: int, timeout: float, log: LogFn) -> dict | None:
    aid = parse_arxiv_id_from_url(url)
    if aid is None:
        return None
    paper = fetch_arxiv_paper_by_id(aid, timeout=timeout)
    if not paper:
        return None
    title = paper.get("title") or aid
    abstract = paper.get("abstract") or ""
    authors = paper.get("authors") or []
    head = ", ".join(authors[:12]) if authors else ""
    body = f"# {title}\n\n"
    if head:
        body += f"Authors: {head}\n\n"
    body += abstract
    log(
        "    [dim]  [green]extract[/green] arxiv API · title + authors + abstract · "
        f"[cyan]{len(body)}[/] chars · [italic]{_rich_escape(preview_snippet(body))}[/][/dim]"
    )
    return _payload(
        final_url=paper.get("url") or url,
        title=title,
        body=body,
        max_chars=max_chars,
        meta_description=abstract[:280] + "…" if len(abstract) > 280 else abstract or None,
    )


def crossref_shallow_payload(url: str, *, max_chars: int, timeout: float, log: LogFn) -> dict | None:
    doi = parse_doi_from_url(url)
    if doi is None:
        return None
    got = fetch_crossref_work(doi, timeout=timeout)
    if got is None:
        return None
    title, landing, body = got
    log(
        "    [dim]  [green]extract[/green] crossref API · work metadata + abstract · "
        f"[cyan]{len(body)}[/] chars · [italic]{_rich_escape(preview_snippet(body))}[/][/dim]"
    )
    return _payload(final_url=landing, title=title, body=body, max_chars=max_chars)


def semantic_scholar_shallow_payload(url: str, *, max_chars: int, timeout: float, log: LogFn) -> dict | None:
    if "semanticscholar.org" not in url.lower():
        return None
    pid = parse_semanticscholar_paper_id(url)
    if pid is None:
        return None
    got = fetch_semantic_scholar_paper(pid, timeout=timeout)
    if got is None:
        return None
    title, landing, body = got
    log(
        "    [dim]  [green]extract[/green] semantic scholar API · paper metadata + abstract · "
        f"[cyan]{len(body)}[/] chars · [italic]{_rich_escape(preview_snippet(body))}[/][/dim]"
    )
    return _payload(final_url=landing, title=title, body=body, max_chars=max_chars)


def scholarly_plaintext(url: str, *, timeout: float, github_token: str | None) -> str | None:
    """Plain text for :func:`extractor._http_article_text` (APIs only, no HTML)."""
    u = url.strip()
    aid = parse_arxiv_id_from_url(u)
    if aid:
        paper = fetch_arxiv_paper_by_id(aid, timeout=timeout)
        if paper:
            title = paper.get("title") or ""
            abstract = paper.get("abstract") or ""
            authors = paper.get("authors") or []
            head = ", ".join(authors[:12]) if authors else ""
            body = f"# {title}\n\n"
            if head:
                body += f"Authors: {head}\n\n"
            body += abstract
            return body.strip() or None

    doi = parse_doi_from_url(u)
    if doi:
        got = fetch_crossref_work(doi, timeout=timeout)
        if got:
            return got[2].strip() or None

    if "semanticscholar.org" in u.lower():
        pid = parse_semanticscholar_paper_id(u)
        if pid:
            got = fetch_semantic_scholar_paper(pid, timeout=timeout)
            if got:
                return got[2].strip() or None

    gh = parse_github_repo_url(u)
    if gh:
        got = fetch_github_readme_and_release(gh[0], gh[1], token=github_token, timeout=timeout)
        if got:
            return got[2].strip() or None
    return None
