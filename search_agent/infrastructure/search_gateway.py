from __future__ import annotations

import logfire

from search_agent.infrastructure.brave_search import search_brave_with_fallback
from search_agent.settings import AppSettings


class BraveSearchGateway:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def search_variant(self, query: str, profile, log=None):
        with logfire.span(
            "search_gateway.search_variant",
            query=query,
            profile=getattr(profile, "name", None),
        ):
            return search_brave_with_fallback(query, profile, self._settings, log=log)
