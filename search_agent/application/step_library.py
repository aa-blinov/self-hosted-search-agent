from __future__ import annotations

from search_agent import tuning
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
    infer_claim_profile,
    route_claim_retrieval,
    should_stop_claim_loop,
    utility_rerank_passages,
)


class AgentStepLibrary:
    def build_run_id(self, query: str, started_at) -> str:
        return _build_run_id(query, started_at)

    def infer_claim_profile(self, claim, classification):
        return infer_claim_profile(claim, classification)

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

    def cheap_passage_filter(self, claim, passages, limit=tuning.CHEAP_PASSAGE_LIMIT):
        return cheap_passage_filter(claim, passages, limit=limit)

    def utility_rerank_passages(self, claim, passages, prior_passage_ids=None):
        return utility_rerank_passages(claim, passages, prior_passage_ids=prior_passage_ids)

    def build_evidence_bundle(self, claim, passages, verification, gated_results):
        return build_evidence_bundle(claim, passages, verification, gated_results)

    def build_snippet_passages(self, gated_results):
        return build_snippet_passages(gated_results)

    def should_stop_claim_loop(self, claim, bundle, iteration):
        return should_stop_claim_loop(claim, bundle, iteration)

    def compose_answer(self, report):
        return compose_answer(report)

    def estimate_search_cost(self, claim_runs):
        return _estimate_search_cost(claim_runs)
