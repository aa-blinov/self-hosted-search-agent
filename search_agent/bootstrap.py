from __future__ import annotations

from functools import lru_cache

from search_agent.application.step_library import AgentStepLibrary
from search_agent.application.unified_runner import UnifiedSearchAgentUseCase
from search_agent.application.use_cases import SearchAgentUseCase
from search_agent.infrastructure.fetch_gateway import AgentFetchGateway
from search_agent.infrastructure.gateway_factory import build_search_gateway
from search_agent.infrastructure.intelligence import PydanticAIQueryIntelligence
from search_agent.infrastructure.receipt_gateway import JsonReceiptWriter
from search_agent.infrastructure.telemetry import configure_logfire
from search_agent.settings import get_settings


@lru_cache(maxsize=1)
def build_search_agent_use_case() -> SearchAgentUseCase:
    settings = get_settings()
    configure_logfire(settings)
    return SearchAgentUseCase(
        intelligence=PydanticAIQueryIntelligence(settings),
        search_gateway=build_search_gateway(settings),
        fetch_gateway=AgentFetchGateway(),
        receipt_writer=JsonReceiptWriter(),
        steps=AgentStepLibrary(),
    )


@lru_cache(maxsize=1)
def build_unified_search_agent_use_case() -> UnifiedSearchAgentUseCase:
    settings = get_settings()
    configure_logfire(settings)
    return UnifiedSearchAgentUseCase(
        intelligence=PydanticAIQueryIntelligence(settings),
        search_gateway=build_search_gateway(settings),
        fetch_gateway=AgentFetchGateway(),
        receipt_writer=JsonReceiptWriter(),
        steps=AgentStepLibrary(),
    )
