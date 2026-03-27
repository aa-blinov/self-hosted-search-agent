"""
Structured APIs for academic / code hosts (URL-detected): arXiv, Crossref, Semantic Scholar, GitHub.

Used by :mod:`search_agent.infrastructure.source_handlers` before generic HTTP scraping.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from html.parser import HTMLParser
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


class _MarkupStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join("".join(self._parts).split())


def _normalized_host(url: str) -> str:
    try:
        host = (urlparse(url.strip()).netloc or "").lower().split("@")[-1].split(":")[0]
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _path_segments(url: str) -> list[str]:
    try:
        path = unquote(urlparse(url.strip()).path or "")
    except Exception:
        return []
    return [segment for segment in path.split("/") if segment]


def _is_hex_token(value: str) -> bool:
    token = (value or "").strip()
    if len(token) != 40:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in token)


def _strip_arxiv_version(value: str) -> str:
    token = (value or "").strip()
    marker = token.rfind("v")
    if marker > 0 and token[marker + 1 :].isdigit():
        return token[:marker]
    return token


def parse_github_repo_url(url: str) -> tuple[str, str] | None:
    """``https://github.com/owner/repo`` or ``.../tree/...`` -> owner/repo only."""
    if _normalized_host(url) != "github.com":
        return None
    parts = _path_segments(url)
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if owner.lower() in _GITHUB_RESERVED or repo.lower() in _GITHUB_RESERVED:
        return None
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def parse_arxiv_id_from_url(url: str) -> str | None:
    if _normalized_host(url) != "arxiv.org":
        return None
    parts = _path_segments(url)
    if len(parts) < 2 or parts[0].lower() not in {"abs", "pdf"}:
        return None
    aid = "/".join(parts[1:]).rstrip("/")
    if aid.lower().endswith(".pdf"):
        aid = aid[:-4]
    return aid or None


def parse_doi_from_url(url: str) -> str | None:
    if not _normalized_host(url).endswith("doi.org"):
        return None
    raw = "/".join(_path_segments(url)).strip().rstrip("/")
    if raw.lower().startswith("10."):
        return raw
    return None


def parse_semanticscholar_paper_id(url: str) -> str | None:
    """Corpus id (40 hex) or ``ARXIV:...`` / ``DOI:...`` for the Graph API."""
    if "semanticscholar.org" not in _normalized_host(url):
        return None
    parts = _path_segments(url)
    for part in parts:
        if _is_hex_token(part):
            return part

    for index, part in enumerate(parts):
        lower = part.casefold()
        if lower.startswith("arxiv:"):
            candidate = _strip_arxiv_version(part.split(":", 1)[1])
            if candidate and all(char.isdigit() or char == "." for char in candidate):
                return "ARXIV:" + candidate
        if lower == "arxiv" and index + 1 < len(parts):
            candidate = _strip_arxiv_version(parts[index + 1])
            if candidate and all(char.isdigit() or char == "." for char in candidate):
                return "ARXIV:" + candidate
        if lower.startswith("doi:10."):
            return "DOI:" + part.split(":", 1)[1]
        if lower == "doi" and index + 1 < len(parts):
            candidate = "/".join(parts[index + 1 :]).strip()
            if candidate.lower().startswith("10."):
                return "DOI:" + candidate
    return None


def _strip_xmlish(text: str) -> str:
    parser = _MarkupStripper()
    parser.feed(text or "")
    parser.close()
    return parser.text()


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
        abstract = " - ".join(parts) if parts else ""

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
    """Graph API: corpus id, ``ARXIV:...``, ``DOI:...``, etc."""
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
