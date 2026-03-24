from __future__ import annotations

import logfire

from search_agent.infrastructure.searxng import search_searxng_with_fallback


class SearxngSearchGateway:
    def search_variant(self, query: str, profile, log=None):
        with logfire.span(
            "search_gateway.search_variant",
            query=query,
            profile=getattr(profile, "name", None),
        ):
            return search_searxng_with_fallback(query, profile, log=log)
