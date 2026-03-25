from __future__ import annotations

from search_agent.infrastructure.ddgs_gateway import DDGSSearchGateway
from search_agent.infrastructure.search_gateway import BraveSearchGateway
from search_agent.settings import AppSettings


def build_search_gateway(settings: AppSettings):
    provider = settings.resolved_search_provider()
    if provider == "ddgs":
        return DDGSSearchGateway(settings)
    return BraveSearchGateway(settings)
