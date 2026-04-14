from __future__ import annotations

from typing import Protocol

from search_agent.domain.assessment import Assessment
from search_agent.domain.models import (
    AgentRunResult,
    Claim,
    FetchedDocument,
    FetchPlan,
    GatedSerpResult,
    Passage,
    QueryClassification,
    QueryVariant,
    RoutingDecision,
    SearchSnapshot,
)


class QueryIntelligencePort(Protocol):
    def classify_query(self, query: str, log=None) -> QueryClassification:
        ...

    def generate_queries_unified(
        self,
        *,
        user_query: str,
        normalized_query: str,
        iteration: int,
        prior_assessment: Assessment | None,
        used_queries: set[str],
        log=None,
    ) -> list[str]:
        ...

    def assess_and_answer(
        self,
        user_query: str,
        passages: list[Passage],
        log=None,
    ) -> Assessment:
        ...


class SearchGatewayPort(Protocol):
    def search_variant(self, query: str, profile, log=None) -> list[SearchSnapshot]:
        ...


class FetchGatewayPort(Protocol):
    def fetch_claim_documents(
        self,
        claim: Claim,
        gated_results: list[GatedSerpResult],
        profile,
        routing_decision: RoutingDecision,
        *,
        seen_urls: set[str],
        log=None,
        iteration: int = 1,
        page_cache=None,
        page_cache_lock=None,
        intent: str = "factual",
    ) -> tuple[list[FetchPlan], list[FetchedDocument]]:
        ...


class ReceiptWriterPort(Protocol):
    def write(self, report: AgentRunResult, output_dir: str) -> str:
        ...


class StepLibraryPort(Protocol):
    """Thin surface of helper steps the unified runner calls.

    Lots of helper functions still live in ``agent_steps`` / ``agent_scoring``
    (they are invoked directly by component evals and ``bench/routing_probe``),
    but the unified orchestrator only needs this small subset.
    """

    def build_run_id(self, query: str, started_at) -> str:
        ...

    def retag_snapshot(self, snapshot: SearchSnapshot, variant: QueryVariant) -> SearchSnapshot:
        ...

    def gate_serp_results(
        self,
        claim: Claim,
        snapshots: list[SearchSnapshot],
        limit: int,
    ) -> list[GatedSerpResult]:
        ...

    def documents_for_passage_extraction(
        self,
        documents: list[FetchedDocument],
    ) -> list[FetchedDocument]:
        ...

    def split_into_passages(self, document: FetchedDocument) -> list[Passage]:
        ...

    def estimate_search_cost(self, claim_runs) -> float:
        ...
