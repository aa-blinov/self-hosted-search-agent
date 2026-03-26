from __future__ import annotations

from search_agent.application.agent_steps import (
    _build_run_id,
    _documents_for_passage_extraction,
    _estimate_search_cost,
    _retag_snapshot,
    _split_into_passages,
    build_evidence_bundle,
    build_query_variants,
    build_snippet_passages,
    cheap_passage_filter,
    compose_answer,
    gate_serp_results,
    refine_query_variants,
    route_claim_retrieval,
    should_stop_claim_loop,
    utility_rerank_passages,
)


class LegacyAgentStepLibrary:
    def build_run_id(self, query: str, started_at) -> str:
        return _build_run_id(query, started_at)

    def build_query_variants(self, claim, classification):
        return build_query_variants(claim, classification)

    def retag_snapshot(self, snapshot, variant):
        return _retag_snapshot(snapshot, variant)

    def gate_serp_results(self, claim, snapshots, limit):
        return gate_serp_results(claim, snapshots, limit)

    def route_claim_retrieval(self, claim, gated_results):
        return route_claim_retrieval(claim, gated_results)

    def documents_for_passage_extraction(self, documents):
        return _documents_for_passage_extraction(documents)

    def split_into_passages(self, document):
        return _split_into_passages(document)

    def cheap_passage_filter(self, claim, passages):
        return cheap_passage_filter(claim, passages)

    def utility_rerank_passages(self, claim, passages):
        return utility_rerank_passages(claim, passages)

    def build_evidence_bundle(self, claim, passages, verification, gated_results):
        return build_evidence_bundle(claim, passages, verification, gated_results)

    def build_snippet_passages(self, gated_results):
        return build_snippet_passages(gated_results)

    def refine_query_variants(
        self,
        claim,
        classification,
        verification,
        gated_results,
        bundle,
        next_iteration,
        existing_queries,
    ):
        return refine_query_variants(
            claim,
            classification,
            verification,
            gated_results,
            bundle,
            next_iteration,
            existing_queries,
        )

    def should_stop_claim_loop(self, claim, bundle, iteration):
        return should_stop_claim_loop(claim, bundle, iteration)

    def compose_answer(self, report):
        return compose_answer(report)

    def estimate_search_cost(self, claim_runs):
        return _estimate_search_cost(claim_runs)
