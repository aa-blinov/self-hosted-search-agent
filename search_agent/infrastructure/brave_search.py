"""
Brave Search API helpers (web + news) with the same fallback chain as legacy backends.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from urllib.parse import urlparse

import requests

from search_agent.domain.models import SearchSnapshot, SerpResult
from search_agent.infrastructure.serp_query import build_routed_query
from search_agent.infrastructure.url_utils import canonicalize_url
from search_agent.settings import AppSettings


def _freshness_param(time_range: str | None) -> str | None:
    if not time_range:
        return None
    mapping = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}
    return mapping.get(time_range)


def _is_backend_degraded(snapshot: SearchSnapshot) -> bool:
    return not snapshot.results and bool(snapshot.unresponsive_engines)


def _simplify_fallback_query(query: str) -> str:
    text = re.sub(r"[\"“”]", " ", query)
    text = re.sub(r"\b(?:OR|AND)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[?]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or query


def _fallback_profiles(profile) -> list:
    from search_agent.config.profiles import get_profile

    chain: list[str] = []
    if getattr(profile, "name", "") != "reference":
        chain.append("reference")
    if getattr(profile, "name", "") != "web":
        chain.append("web")
    if getattr(profile, "name", "") != "deep":
        chain.append("deep")
    return [get_profile(name) for name in chain]


def _merge_goggles(profile, settings: AppSettings) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for source in (*settings.resolved_brave_goggles(), *(getattr(profile, "goggles", None) or [])):
        g = (source or "").strip()
        if not g or g in seen:
            continue
        seen.add(g)
        out.append(g)
    return out


def _news_mode(profile) -> bool:
    cats = getattr(profile, "categories", []) or []
    return bool(cats) and cats[0] == "news" and "general" not in cats


def _parse_web_results(data: dict, profile, max_results: int) -> tuple[list[SerpResult], list[str]]:
    raw_list = (data.get("web") or {}).get("results") or []
    unresponsive: list[str] = []
    results: list[SerpResult] = []
    for idx, row in enumerate(raw_list[:max_results], 1):
        url = (row.get("url") or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        snippet = (row.get("description") or row.get("snippet") or "").strip()
        results.append(
            SerpResult(
                result_id=f"brave:{idx}",
                query_variant_id="legacy",
                title=(row.get("title") or "").strip(),
                url=url,
                snippet=snippet,
                canonical_url=canonicalize_url(url),
                host=host,
                position=idx,
                engine="brave-web",
                published_at=None,
                raw=row,
            )
        )
    if not results and not raw_list:
        unresponsive.append("brave-web: empty")
    return results, unresponsive


def _parse_news_results(data: dict, profile, max_results: int) -> tuple[list[SerpResult], list[str]]:
    # Brave News API returns a top-level "results" list; older docs also mention news.results
    raw_list = data.get("results")
    if raw_list is None:
        raw_list = (data.get("news") or {}).get("results")
    raw_list = raw_list or []
    unresponsive: list[str] = []
    results: list[SerpResult] = []
    for idx, row in enumerate(raw_list[:max_results], 1):
        url = (row.get("url") or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        snippet = (row.get("description") or "").strip()
        published_at = (row.get("page_age") or row.get("age") or None)
        if published_at is not None:
            published_at = str(published_at).strip() or None
        results.append(
            SerpResult(
                result_id=f"brave-news:{idx}",
                query_variant_id="legacy",
                title=(row.get("title") or "").strip(),
                url=url,
                snippet=snippet,
                canonical_url=canonicalize_url(url),
                host=host,
                position=idx,
                engine="brave-news",
                published_at=published_at,
                raw=row,
            )
        )
    if not results and not raw_list:
        unresponsive.append("brave-news: empty")
    return results, unresponsive


def search_brave(query: str, profile, settings: AppSettings, log=None) -> SearchSnapshot:
    log = log or (lambda msg: None)
    api_key = (settings.brave_api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "BRAVE_API_KEY is not set. Get a key at https://brave.com/search/api/ and add it to .env"
        )

    routed_query = build_routed_query(query, profile)
    base = (settings.brave_base_url or "https://api.search.brave.com").rstrip("/")
    news = _news_mode(profile)
    path = "/res/v1/news/search" if news else "/res/v1/web/search"
    url = f"{base}{path}"

    count = min(int(profile.max_results), 50 if news else 20)
    goggles = _merge_goggles(profile, settings)

    params_pairs: list[tuple[str, str | int]] = [
        ("q", routed_query),
        ("count", count),
        ("offset", 0),
        ("extra_snippets", "true"),
    ]
    freshness = _freshness_param(profile.time_range)
    if freshness:
        params_pairs.append(("freshness", freshness))

    lang = (profile.language or "auto").strip()
    if lang and lang != "auto":
        params_pairs.append(("search_lang", lang))

    country = (settings.brave_country or "US").strip().upper()
    if len(country) == 2:
        params_pairs.append(("country", country))

    ss = (settings.brave_safesearch or "moderate").strip().lower()
    if ss in {"off", "moderate", "strict"}:
        params_pairs.append(("safesearch", ss))

    for g in goggles:
        params_pairs.append(("goggles", g))

    mode = "news" if news else "web"
    gog_note = f", goggles={len(goggles)}" if goggles else ""
    log(
        f"  [cyan]Brave[/cyan]  [italic]\"{routed_query}\"[/italic]  "
        f"[dim]({mode}, lang={profile.language}{gog_note})[/dim]"
    )

    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, params=params_pairs, headers=headers, timeout=20)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach Brave Search API at {base}: {exc}") from exc

    if resp.status_code == 401:
        raise RuntimeError("Brave Search API rejected the key (401). Check BRAVE_API_KEY.")
    if resp.status_code == 429:
        raise RuntimeError("Brave Search API rate limit (429). Retry later or upgrade plan.")

    try:
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Brave Search API error: HTTP {resp.status_code} — {exc}") from exc

    if news:
        results, unresponsive = _parse_news_results(data, profile, count)
    else:
        results, unresponsive = _parse_web_results(data, profile, count)

    log(
        f"  [dim]-> {len(results)} results"
        + (f", issues: {unresponsive[:3]}" if unresponsive else "")
        + "[/dim]"
    )

    return SearchSnapshot(
        query=routed_query,
        suggestions=[],
        results=results,
        retrieved_at=datetime.now(UTC).isoformat(),
        profile_name=getattr(profile, "name", None),
        unresponsive_engines=unresponsive,
    )


def search_brave_with_fallback(query: str, profile, settings: AppSettings, log=None) -> list[SearchSnapshot]:
    log = log or (lambda msg: None)
    attempts: list[SearchSnapshot] = []
    attempted: set[tuple[str | None, str]] = set()
    delay = settings.search_backend_fallback_delay_sec

    def _execute(query_text: str, active_profile):
        key = (getattr(active_profile, "name", None), query_text)
        if key in attempted:
            return None
        attempted.add(key)
        snapshot = search_brave(query_text, active_profile, settings, log=log)
        attempts.append(snapshot)
        return snapshot

    snapshot = _execute(query, profile)
    if snapshot is None or not _is_backend_degraded(snapshot):
        return attempts

    if not settings.brave_search_fallback:
        return attempts

    log("  [yellow]search backend degraded; starting fallback chain[/yellow]")
    fallback_query = _simplify_fallback_query(query)
    plan: list[tuple[str, object]] = []
    if fallback_query != query:
        plan.append((fallback_query, profile))
    for fallback_profile in _fallback_profiles(profile):
        plan.append((fallback_query, fallback_profile))

    for idx, (candidate_query, candidate_profile) in enumerate(plan, 1):
        if delay > 0:
            time.sleep(delay)
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
