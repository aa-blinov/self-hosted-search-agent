from __future__ import annotations

import time
from datetime import UTC, datetime
from urllib.parse import urlparse

import logfire
from ddgs import DDGS

from search_agent.domain.models import SearchSnapshot, SerpResult
from search_agent.infrastructure.serp_query import build_routed_query
from search_agent.infrastructure.url_utils import canonicalize_url
from search_agent.settings import AppSettings

_DDGS_TIMELIMIT_FROM_TIME_RANGE: dict[str, str] = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}

# Retry policy for transient DDGS failures (e.g. Yahoo backend 5xx, flaky
# DNS, empty results).  We keep retries tight — the agent parallelises many
# SERP calls and the caller has its own budget, so we don't want to amplify
# a slow backend into minutes of waiting.
_DDGS_RETRY_ATTEMPTS = 2  # retries on top of the initial attempt (3 total)
_DDGS_RETRY_BACKOFF_S = (0.5, 1.0)  # sleep before retry 1, retry 2


def _ddgs_timelimit(profile) -> str | None:
    override = getattr(profile, "ddgs_timelimit", None)
    if override is not None:
        return override.strip() or None
    tr = getattr(profile, "time_range", None)
    if not tr:
        return None
    return _DDGS_TIMELIMIT_FROM_TIME_RANGE.get(tr)


def _ddgs_region(profile, settings: AppSettings) -> str:
    r = getattr(profile, "ddgs_region", None)
    return r if r is not None else settings.ddgs_region


def _ddgs_safesearch(profile, settings: AppSettings) -> str:
    s = getattr(profile, "ddgs_safesearch", None)
    return s if s is not None else settings.ddgs_safesearch


class DDGSSearchGateway:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def search_variant(self, query: str, profile, log=None):
        log = log or (lambda msg: None)
        with logfire.span(
            "search_gateway.ddgs.search_variant",
            query=query,
            profile=getattr(profile, "name", None),
        ):
            routed = build_routed_query(query, profile)
            region = _ddgs_region(profile, self._settings)
            safesearch = _ddgs_safesearch(profile, self._settings)
            timelimit = _ddgs_timelimit(profile)
            log(
                f"  [cyan]DDGS[/cyan]  [italic]\"{routed}\"[/italic]  "
                f"[dim](region={region}, safesearch={safesearch}, timelimit={timelimit})[/dim]"
            )
            ddgs_failed = False
            results: list = []
            total_attempts = _DDGS_RETRY_ATTEMPTS + 1
            for attempt in range(total_attempts):
                try:
                    results = DDGS(timeout=self._settings.ddgs_timeout).text(
                        routed,
                        region=region,
                        safesearch=safesearch,
                        max_results=profile.max_results,
                        timelimit=timelimit,
                        timeout=self._settings.ddgs_timeout,
                    )
                except Exception as exc:
                    exc_str = str(exc)
                    # DDGS internally queries Wikipedia OpenSearch using the region prefix
                    # (e.g. "wt" from "wt-wt") which produces invalid subdomains like
                    # wt.wikipedia.org. This is a library-level limitation — suppress the
                    # noise and treat as empty rather than a real backend failure, no retry.
                    is_wiki_dns = "wikipedia.org" in exc_str and "ConnectError" in exc_str
                    if is_wiki_dns:
                        log("  [dim]DDGS Wikipedia instant-answer skipped (DNS not reachable)[/dim]")
                        results = []
                        break
                    # Real backend failure (Yahoo 5xx, DNS hiccup, timeout).
                    if attempt < total_attempts - 1:
                        backoff = _DDGS_RETRY_BACKOFF_S[attempt]
                        log(
                            f"  [yellow]DDGS error (attempt {attempt + 1}/{total_attempts}):[/yellow] "
                            f"{exc} [dim]— retrying in {backoff}s[/dim]"
                        )
                        time.sleep(backoff)
                        continue
                    ddgs_failed = True
                    logfire.warning(
                        "search_gateway.ddgs.text_failed",
                        error=exc_str,
                        query=routed[:200],
                        attempts=total_attempts,
                    )
                    log(f"  [yellow]DDGS error (empty results after {total_attempts} attempts):[/yellow] {exc}")
                    results = []
                    break
                # No exception.  If the backend returned an empty list, retry —
                # DDGS sometimes shrugs off a transient Yahoo failure and comes
                # back with real results on a second try.
                if not results and attempt < total_attempts - 1:
                    backoff = _DDGS_RETRY_BACKOFF_S[attempt]
                    log(
                        f"  [yellow]DDGS empty result (attempt {attempt + 1}/{total_attempts}):[/yellow] "
                        f"[dim]retrying in {backoff}s[/dim]"
                    )
                    time.sleep(backoff)
                    continue
                break

            serp_results: list[SerpResult] = []
            for idx, row in enumerate(results[: profile.max_results], 1):
                url = str(row.get("href", "")).strip()
                if not url:
                    continue
                parsed = urlparse(url)
                host = parsed.netloc.lower()
                if host.startswith("www."):
                    host = host[4:]
                serp_results.append(
                    SerpResult(
                        result_id=f"ddgs:{idx}",
                        query_variant_id="legacy",
                        title=str(row.get("title", "")),
                        url=url,
                        snippet=str(row.get("body", "")).strip(),
                        canonical_url=canonicalize_url(url),
                        host=host,
                        position=idx,
                        raw=row,
                    )
                )

            unresponsive: list[str] = ["ddgs"] if ddgs_failed else []
            return [
                SearchSnapshot(
                    query=routed,
                    suggestions=[],
                    results=serp_results,
                    retrieved_at=datetime.now(UTC).isoformat(),
                    profile_name=f"ddgs:{getattr(profile, 'name', 'default')}",
                    unresponsive_engines=unresponsive,
                )
            ]
