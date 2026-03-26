from __future__ import annotations

from typing import Protocol

from search_agent.domain.models import (
    AgentRunResult,
    Claim,
    EvidenceBundle,
    FetchedDocument,
    FetchPlan,
    GatedSerpResult,
    Passage,
    QueryClassification,
    QueryVariant,
    RoutingDecision,
    SearchSnapshot,
    VerificationResult,
)


class QueryIntelligencePort(Protocol):
    def classify_query(self, query: str, log=None) -> QueryClassification:
        ...

    def decompose_claims(self, classification: QueryClassification, log=None) -> list[Claim]:
        ...

    def verify_claim(self, claim: Claim, passages: list[Passage], log=None) -> VerificationResult:
        ...

    def synthesize_answer(self, query: str, passages: list[Passage], log=None, intent: str = "synthesis") -> str:
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
    def build_run_id(self, query: str, started_at) -> str:
        ...

    def build_query_variants(
        self,
        claim: Claim,
        classification: QueryClassification,
    ) -> list[QueryVariant]:
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

    def route_claim_retrieval(
        self,
        claim: Claim,
        gated_results: list[GatedSerpResult],
    ) -> RoutingDecision:
        ...

    def build_snippet_passages(
        self,
        gated_results: list[GatedSerpResult],
    ) -> list[Passage]:
        ...

    def documents_for_passage_extraction(
        self,
        documents: list[FetchedDocument],
    ) -> list[FetchedDocument]:
        ...

    def split_into_passages(self, document: FetchedDocument) -> list[Passage]:
        ...

    def cheap_passage_filter(self, claim: Claim, passages: list[Passage]) -> list[Passage]:
        ...

    def utility_rerank_passages(self, claim: Claim, passages: list[Passage]) -> list[Passage]:
        ...

    def build_evidence_bundle(
        self,
        claim: Claim,
        passages: list[Passage],
        verification: VerificationResult,
        gated_results: list[GatedSerpResult],
    ) -> EvidenceBundle:
        ...

    def refine_query_variants(
        self,
        claim: Claim,
        classification: QueryClassification,
        verification: VerificationResult,
        gated_results: list[GatedSerpResult],
        bundle: EvidenceBundle,
        next_iteration: int,
        existing_queries: set[str],
    ) -> list[QueryVariant]:
        ...

    def should_stop_claim_loop(
        self,
        claim: Claim,
        bundle: EvidenceBundle,
        iteration: int,
    ) -> bool:
        ...

    def compose_answer(self, report: AgentRunResult) -> str:
        ...

    def estimate_search_cost(self, claim_runs) -> float:
        ...
