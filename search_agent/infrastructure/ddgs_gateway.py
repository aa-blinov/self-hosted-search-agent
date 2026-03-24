from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlparse

import logfire
from ddgs import DDGS

from search_agent.domain.models import SearchSnapshot, SerpResult
from search_agent.infrastructure.searxng import _canonicalize_url
from search_agent.settings import AppSettings


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
            log(
                f"  [cyan]DDGS[/cyan]  [italic]\"{query}\"[/italic]  "
                f"[dim](region={self._settings.ddgs_region}, safesearch={self._settings.ddgs_safesearch})[/dim]"
            )
            results = DDGS().text(
                query,
                region=self._settings.ddgs_region,
                safesearch=self._settings.ddgs_safesearch,
                max_results=profile.max_results,
                timelimit=profile.time_range,
                timeout=self._settings.ddgs_timeout,
            )

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
                        canonical_url=_canonicalize_url(url),
                        host=host,
                        position=idx,
                        raw=row,
                    )
                )

            return [
                SearchSnapshot(
                    query=query,
                    suggestions=[],
                    results=serp_results,
                    retrieved_at=datetime.now(UTC).isoformat(),
                    profile_name=f"ddgs:{getattr(profile, 'name', 'default')}",
                    unresponsive_engines=[],
                )
            ]
