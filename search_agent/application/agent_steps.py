from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime

from search_agent import tuning
from search_agent.application import policy_tuning
from search_agent.application.agent_compose import (
    _compose_ui_labels,
    _estimate_search_cost,
    compose_answer,
)
from search_agent.application.agent_scoring import (
    _answer_type,
    _documents_for_passage_extraction,
    _split_into_passages,
    build_evidence_bundle,
    build_snippet_passages,
    cheap_passage_filter,
    fetch_claim_documents,
    gate_serp_results,
    route_claim_retrieval,
    should_stop_claim_loop,
    utility_rerank_passages,
)
from search_agent.application.text_heuristics import normalized_text as _shared_normalized_text
from search_agent.domain.models import (
    AgentRunResult,
    Claim,
    ClaimProfile,
    QueryClassification,
    QueryVariant,
    SearchSnapshot,
)


def _build_run_id(query: str, started_at: datetime) -> str:
    digest = hashlib.sha1(f"{query}|{started_at.isoformat()}".encode("utf-8")).hexdigest()[:8]
    return f"{started_at.strftime('%Y%m%dT%H%M%S')}-{digest}"


def _normalized_text(text: str) -> str:
    return _shared_normalized_text(text)


def infer_claim_profile(claim: Claim, classification: QueryClassification) -> ClaimProfile:
    if claim.claim_profile is not None:
        return claim.claim_profile
    if classification.intent == "news_digest":
        return ClaimProfile(
            answer_shape="news_digest",
            min_independent_sources=policy_tuning.DEFAULT_NEWS_DIGEST_MIN_INDEPENDENT_SOURCES,
            routing_bias="iterative_loop",
        )
    if classification.intent == "synthesis":
        return ClaimProfile(
            answer_shape="overview",
            min_independent_sources=policy_tuning.DEFAULT_OVERVIEW_MIN_INDEPENDENT_SOURCES,
            routing_bias="iterative_loop",
        )
    return ClaimProfile(
        answer_shape="fact",
        min_independent_sources=policy_tuning.DEFAULT_FACT_MIN_INDEPENDENT_SOURCES,
    )


def build_query_variants(claim: Claim, classification: QueryClassification) -> list[QueryVariant]:
    queries = [_normalized_text(query) for query in (claim.search_queries or []) if _normalized_text(query)]
    if not queries:
        fallback = _normalized_text(claim.claim_text)
        queries = [fallback] if fallback else []

    seen: set[str] = set()
    variants: list[QueryVariant] = []
    for idx, query_text in enumerate(queries, 1):
        key = query_text.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        variants.append(
            QueryVariant(
                variant_id=f"{claim.claim_id}-q{idx}",
                claim_id=claim.claim_id,
                query_text=query_text,
                strategy=f"llm_{idx}" if claim.search_queries else "claim_text",
                rationale="LLM-planned search query." if claim.search_queries else "Fallback to the raw claim text.",
                source_restriction=None,
                freshness_hint=claim.time_scope,
            )
        )
        if len(variants) >= tuning.AGENT_MAX_QUERY_VARIANTS:
            break
    return variants


def _retag_snapshot(snapshot: SearchSnapshot, variant: QueryVariant) -> SearchSnapshot:
    results = [
        replace(result, result_id=f"{variant.variant_id}:{result.position}", query_variant_id=variant.variant_id)
        for result in snapshot.results
    ]
    return SearchSnapshot(
        query=snapshot.query,
        suggestions=snapshot.suggestions,
        results=results,
        retrieved_at=snapshot.retrieved_at,
        profile_name=snapshot.profile_name,
        unresponsive_engines=list(snapshot.unresponsive_engines),
    )


def refine_query_variants(
    claim: Claim,
    classification: QueryClassification,
    verification,
    gated_results,
    bundle,
    iteration: int,
    existing_queries: set[str],
) -> list[QueryVariant]:
    return []


def run_search_agent(
    query: str,
    profile,
    client=None,
    receipts_dir: str | None = None,
    log=None,
) -> AgentRunResult:
    from search_agent import build_search_agent_use_case

    return build_search_agent_use_case().run(
        query,
        profile,
        receipts_dir=receipts_dir,
        log=log,
    )
