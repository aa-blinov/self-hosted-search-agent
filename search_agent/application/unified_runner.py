"""Unified iterative runner — single procedure for all query types.

Replaces the classify -> decompose -> N×(fetch -> verify) -> synthesize pipeline
with one iterative loop whose stop condition is:

- ``contradicts_query == True`` (counterfactual user query, immediate stop), or
- every ``assessment.key_claims`` entry has >= 2 independent-domain citations and
  ``assessment.confidence >= STOP_CONFIDENCE_THRESHOLD``, or
- iteration == ``AGENT_MAX_CLAIM_ITERATIONS`` hit.

There are no intents, no decomposition, no per-claim verification. One LLM call
per iteration (``assess_and_answer``) does synthesis + self-evaluation in one go.

The runner still returns an ``AgentRunResult`` compatible with the classic eval
harness, so metrics like ``claim_support_rate`` and ``contradiction_detection_rate``
can score it side-by-side.  A single synthetic ``ClaimRun`` is fabricated so that
``evaluation.py`` can pattern-match its verdict / primary-source / source-count
fields without modification.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import urlparse

import logfire

from search_agent import tuning
from search_agent.application.contracts import (
    FetchGatewayPort,
    QueryIntelligencePort,
    ReceiptWriterPort,
    SearchGatewayPort,
    StepLibraryPort,
)
from search_agent.application.text_heuristics import extract_entities, extract_time_scope
from search_agent.domain.assessment import Assessment
from search_agent.domain.models import (
    AgentRunResult,
    AuditTrail,
    Claim,
    ClaimProfile,
    ClaimRun,
    EvidenceBundle,
    EvidenceSpan,
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
from search_agent.infrastructure.caching_search_gateway import CachingBudgetSearchGateway
from search_agent.settings import get_settings


# Stop condition: confidence threshold beneath which we never stop early.
# The full stop check additionally requires every key claim to have >=2
# independent domains in its supporting citations.
STOP_CONFIDENCE_THRESHOLD = 0.75
# Min key_claims the model must return for the structural check to fire.
# If the model returns 0 key_claims we fall back to confidence-only (rare).
STOP_MIN_KEY_CLAIMS = 1
# Top-K passages taken off the ranked pool before URL/domain capping.
# Oversamples so the caps have room to enforce diversity.
UNIFIED_PASSAGE_TOP_K = 25
# URL/domain caps applied to build the final prompt slate.  These numbers match
# the classic synthesize_answer path's effective budget, but the capping now
# lives in the runner (not in intelligence.assess_and_answer) so that the
# caller-controlled passage order is the same order the LLM sees as [1..N].
UNIFIED_PROMPT_MAX_PER_URL = 2
UNIFIED_PROMPT_MAX_PER_DOMAIN = 4
UNIFIED_PROMPT_MAX_TOTAL = 14


def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return url or ""


def _verdict_from_assessment(assessment: Assessment | None) -> str:
    """Map assessment → legacy verdict string for eval compatibility.

    - contradicts_query=True      → "contradicted"
    - confidence >= STOP_THRESHOLD → "supported"
    - otherwise                    → "insufficient_evidence"
    """
    if assessment is None:
        return "insufficient_evidence"
    if assessment.contradicts_query:
        return "contradicted"
    if assessment.confidence >= STOP_CONFIDENCE_THRESHOLD:
        return "supported"
    return "insufficient_evidence"


def _independent_domains_for_key_claim(
    citation_numbers: list[int],
    passage_index: dict[int, Passage],
) -> set[str]:
    domains: set[str] = set()
    for n in citation_numbers:
        passage = passage_index.get(n)
        if passage is None:
            continue
        host = (passage.host or "").lower() or _domain_of(passage.url)
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.add(host)
    return domains


def _stop_check(
    assessment: Assessment,
    passage_index: dict[int, Passage],
    iteration: int,
    max_iterations: int,
) -> bool:
    """Return True when the unified loop should terminate."""
    if iteration >= max_iterations:
        return True
    if assessment.contradicts_query:
        return True
    if assessment.confidence < STOP_CONFIDENCE_THRESHOLD:
        return False
    if len(assessment.key_claims) < STOP_MIN_KEY_CLAIMS:
        # Model confident but returned no structured claims — accept confidence as stop.
        return True
    for kc in assessment.key_claims:
        domains = _independent_domains_for_key_claim(kc.supporting_citation_numbers, passage_index)
        if len(domains) < 2:
            return False
    return True


def _build_synthetic_claim(query: str) -> Claim:
    """Single synthetic claim used to drive gate/fetch/rerank helpers that still take a Claim.

    min_independent_sources=2 mirrors the unified stop threshold so that the
    eval harness can check ``source_count_ok`` against the same bar.
    """
    entities = extract_entities(query)
    time_scope = extract_time_scope(query)
    return Claim(
        claim_id="unified-1",
        claim_text=query,
        priority=1,
        needs_freshness=False,
        entity_set=list(entities),
        time_scope=time_scope,
        search_queries=[],
        claim_profile=ClaimProfile(
            answer_shape="overview",
            primary_source_required=False,
            min_independent_sources=2,
            preferred_domain_types=["official", "academic", "vendor", "major_media"],
            required_dimensions=[],
            focus_terms=[],
            allow_synthesis_without_primary=True,
            strict_contract=False,
        ),
    )


def _build_synthetic_classification(query: str) -> QueryClassification:
    """Fake QueryClassification for the report shell.

    intent='synthesis' is chosen so existing eval post-processing that inspects
    this field (e.g. the ``claim_run_allows_synthesis`` gate in legacy code) stays
    out of the way.  The unified runner ignores intent otherwise.
    """
    return QueryClassification(
        query=query,
        normalized_query=query,
        intent="synthesis",
        complexity="single_hop",
        needs_freshness=False,
        time_scope=None,
        region_hint=None,
        entity_disambiguation=False,
    )


def _build_query_variants(claim: Claim, queries: list[str]) -> list[QueryVariant]:
    variants: list[QueryVariant] = []
    for idx, text in enumerate(queries, 1):
        if not text:
            continue
        variants.append(
            QueryVariant(
                variant_id=f"{claim.claim_id}-u{idx}",
                claim_id=claim.claim_id,
                query_text=text,
                strategy=f"unified_{idx}",
                rationale="Unified runner query",
                source_restriction=None,
                freshness_hint=None,
            )
        )
    return variants


def _rank_passages_for_prompt(
    passages: list[Passage],
    *,
    top_k: int | None = None,
) -> list[Passage]:
    """Rank passages for the prompt slate, with URL-level diversification.

    Passages are first sorted by ``(source_score, utility_score)`` desc — the
    base quality order.  Then a round-robin pass reorders them so chunks
    from different URLs are interleaved, preventing one long article from
    monopolizing the top-K window.

    Motivation: without round-robin, a query that extracts 20+ chunks from a
    single authority page would push every other URL out of the top-K.  The
    downstream URL/domain caps in ``_build_prompt_passages`` then starve the
    slate — e.g. "latest Node.js LTS" produced only 4 prompt passages
    because top-25 held just 3 distinct URLs.  With interleave, top-25 gets
    one passage from each URL before any URL gets two, so the caps see a
    diverse pool to pack from.

    Passages must already be deduped by passage_id before calling this.
    """
    if not passages:
        return []
    sorted_passages = sorted(
        passages,
        key=lambda p: (p.source_score, p.utility_score),
        reverse=True,
    )
    # Bucket by URL, preserving each bucket's internal quality order.
    buckets: dict[str, list[Passage]] = {}
    url_order: list[str] = []
    for p in sorted_passages:
        url = p.url or ""
        if url not in buckets:
            buckets[url] = []
            url_order.append(url)
        buckets[url].append(p)
    # Round-robin: round r picks the r-th best passage from each URL bucket in
    # first-appearance order.  First-appearance order == base quality order of
    # each URL's best chunk, so the highest-quality URL still leads the list.
    interleaved: list[Passage] = []
    round_idx = 0
    while True:
        added = False
        for url in url_order:
            bucket = buckets[url]
            if round_idx < len(bucket):
                interleaved.append(bucket[round_idx])
                added = True
                if top_k is not None and len(interleaved) >= top_k:
                    return interleaved
        if not added:
            break
        round_idx += 1
    return interleaved


def _build_prompt_passages(
    ranked: list[Passage],
    *,
    max_per_url: int = UNIFIED_PROMPT_MAX_PER_URL,
    max_per_domain: int = UNIFIED_PROMPT_MAX_PER_DOMAIN,
    max_total: int = UNIFIED_PROMPT_MAX_TOTAL,
) -> list[Passage]:
    """Apply URL/domain caps on an already-ranked passage list.

    Walks ``ranked`` in order and greedily keeps up to ``max_per_url`` passages
    per exact URL, ``max_per_domain`` per host, and ``max_total`` overall.  The
    returned list is the exact slate the LLM will see, in the exact order that
    citation indices ``[1..N]`` will reference.  This is what the runner then
    hands to ``assess_and_answer`` AND uses to rebuild its ``passage_index`` so
    the indices align 1-to-1.

    Strict greedy pack with per-URL and per-domain caps.
    """
    kept: list[Passage] = []
    per_url: dict[str, int] = {}
    per_domain: dict[str, int] = {}
    for p in ranked:
        if len(kept) >= max_total:
            break
        url = p.url or ""
        host = (p.host or "").lower() or _domain_of(url)
        if host.startswith("www."):
            host = host[4:]
        if per_url.get(url, 0) >= max_per_url:
            continue
        if host and per_domain.get(host, 0) >= max_per_domain:
            continue
        kept.append(p)
        per_url[url] = per_url.get(url, 0) + 1
        if host:
            per_domain[host] = per_domain.get(host, 0) + 1
    return kept


class UnifiedSearchAgentUseCase:
    def __init__(
        self,
        *,
        intelligence: QueryIntelligencePort,
        search_gateway: SearchGatewayPort,
        fetch_gateway: FetchGatewayPort,
        receipt_writer: ReceiptWriterPort,
        steps: StepLibraryPort,
    ) -> None:
        self._intelligence = intelligence
        self._search_gateway = search_gateway
        self._fetch_gateway = fetch_gateway
        self._receipt_writer = receipt_writer
        self._steps = steps

    def run(
        self,
        query: str,
        profile,
        *,
        receipts_dir: str | None = None,
        log=None,
    ) -> AgentRunResult:
        log = log or (lambda msg: None)
        started_at = datetime.now(UTC)

        with logfire.span(
            "search_agent.unified_run",
            query=query,
            profile=getattr(profile, "name", None),
        ):
            provider = get_settings().resolved_search_provider()
            search_gateway = CachingBudgetSearchGateway(
                self._search_gateway,
                provider_label=provider,
            )

            classification = _build_synthetic_classification(query)
            claim = _build_synthetic_claim(query)
            audit = AuditTrail(
                run_id=self._steps.build_run_id(query, started_at),
                profile_name=getattr(profile, "name", None),
                started_at=started_at.isoformat(),
            )

            # Normalize the query through the standard time-reference pipeline so
            # cached intent/queries from classify_query align with the classic runner.
            classic_classification = self._intelligence.classify_query(query, log=log)
            normalized_query = classic_classification.normalized_query

            log(f"\n[bold]Agent Search[/bold] [dim]unified[/dim]")

            page_cache: dict[str, dict] = {}
            page_cache_lock = threading.Lock()

            all_variants: list[QueryVariant] = []
            all_snapshots: list[SearchSnapshot] = []
            all_gated: list[GatedSerpResult] = []
            all_plans: list[FetchPlan] = []
            all_documents: list[FetchedDocument] = []
            seen_urls: set[str] = set()
            seen_documents: set[tuple[str, str, str]] = set()
            used_queries: set[str] = set()
            all_passages: list[Passage] = []
            seen_passage_ids: set[str] = set()

            assessment: Assessment | None = None
            iterations_used = 0
            max_iter = max(1, tuning.AGENT_MAX_CLAIM_ITERATIONS)

            for iteration in range(1, max_iter + 1):
                iterations_used = iteration
                log(f"  [bold]Iteration {iteration}/{max_iter}[/bold]")

                queries = self._intelligence.generate_queries_unified(
                    user_query=query,
                    normalized_query=normalized_query,
                    iteration=iteration,
                    prior_assessment=assessment,
                    used_queries=used_queries,
                    log=log,
                )
                if not queries:
                    log("  [dim yellow]-> no new queries, stopping[/dim yellow]")
                    break
                for q in queries:
                    used_queries.add(q.casefold())

                variants = _build_query_variants(claim, queries)
                all_variants.extend(variants)

                # Parallel SERP
                new_snapshots = self._parallel_search(variants, profile, search_gateway, log)
                all_snapshots.extend(new_snapshots)

                # Gate results (over all snapshots so far, so older hits can rank too)
                gated_limit = min(
                    tuning.SERP_GATE_MAX_URLS,
                    max(tuning.SERP_GATE_MIN_URLS, profile.max_results),
                )
                gated = self._steps.gate_serp_results(claim, all_snapshots, gated_limit)
                all_gated = _merge_gated(all_gated, gated)
                log(f"  [dim]SERP gate kept {len(gated)} URLs[/dim]")

                # Fetch — unified always uses "full" routing (no fast/full split)
                routing = RoutingDecision(
                    mode="full",
                    certainty=0.0,
                    consistency=0.0,
                    evidence_sufficiency=0.0,
                    rationale="Unified runner (no routing).",
                )
                plans, documents = self._fetch_gateway.fetch_claim_documents(
                    claim,
                    gated,
                    profile,
                    routing,
                    seen_urls=seen_urls,
                    log=log,
                    iteration=iteration,
                    page_cache=page_cache,
                    page_cache_lock=page_cache_lock,
                    intent="synthesis",  # unified uses synthesis fetch limits (highest ceiling)
                )
                all_plans.extend(plans)
                for doc in documents:
                    key = (doc.url, doc.fetch_depth, doc.content_hash)
                    if key in seen_documents:
                        continue
                    seen_documents.add(key)
                    seen_urls.add(doc.url)
                    all_documents.append(doc)

                # Split fresh documents into passages
                passage_docs = self._steps.documents_for_passage_extraction(all_documents)
                fresh_passages: list[Passage] = []
                for doc in passage_docs:
                    for passage in self._steps.split_into_passages(doc):
                        if passage.passage_id in seen_passage_ids:
                            continue
                        seen_passage_ids.add(passage.passage_id)
                        fresh_passages.append(passage)
                all_passages.extend(fresh_passages)
                log(f"  [dim]passages: {len(all_passages)} total (+{len(fresh_passages)} new)[/dim]")

                # Rank, then cap per URL/domain → this is the exact slate the
                # LLM will cite as [1..N].  passage_index uses the same order.
                ranked = _rank_passages_for_prompt(all_passages, top_k=UNIFIED_PASSAGE_TOP_K)
                prompt_passages = _build_prompt_passages(ranked)

                # One LLM call: answer + key_claims + confidence + gaps + contradicts
                assessment = self._intelligence.assess_and_answer(query, prompt_passages, log=log)
                log(
                    f"  [dim]assess: prompt_passages={len(prompt_passages)} | "
                    f"confidence={assessment.confidence:.2f} | "
                    f"key_claims={len(assessment.key_claims)} | "
                    f"contradicts_query={assessment.contradicts_query} | "
                    f"gaps={len(assessment.gaps)}[/dim]"
                )

                # Passage index uses the same [N] numbering the model saw
                passage_index = {i + 1: p for i, p in enumerate(prompt_passages)}

                if _stop_check(assessment, passage_index, iteration, max_iter):
                    log("  [dim green]-> stop: sufficiency met[/dim green]")
                    break

            # Build a compatible AgentRunResult shell so evaluation.py can score it.
            # Rebuild the same prompt slate the last iteration saw so that
            # passage_index lookups align with the key_claim citation numbers.
            final_assessment = assessment or Assessment(answer="", confidence=0.0)
            ranked = _rank_passages_for_prompt(all_passages, top_k=UNIFIED_PASSAGE_TOP_K)
            prompt_passages = _build_prompt_passages(ranked)
            passage_index = {i + 1: p for i, p in enumerate(prompt_passages)}

            verdict = _verdict_from_assessment(final_assessment)

            # Independent-source count = union of domains across all key_claims
            all_kc_domains: set[str] = set()
            for kc in final_assessment.key_claims:
                all_kc_domains |= _independent_domains_for_key_claim(
                    kc.supporting_citation_numbers,
                    passage_index,
                )
            if not all_kc_domains:
                # Fall back to unique hosts of the passages actually cited in the answer text
                from search_agent.evaluation import _answer_source_urls as _extract_cited_urls

                for url in _extract_cited_urls(final_assessment.answer):
                    host = _domain_of(url)
                    if host:
                        all_kc_domains.add(host)

            # Primary-source heuristic: at least one gated result marked as primary
            has_primary_source = any(
                gr.assessment.primary_source_likelihood >= 0.7 for gr in all_gated
            )

            supporting_spans: list[EvidenceSpan] = []
            contradicting_spans: list[EvidenceSpan] = []
            for kc in final_assessment.key_claims:
                for n in kc.supporting_citation_numbers:
                    passage = passage_index.get(n)
                    if passage is None:
                        continue
                    span = EvidenceSpan(
                        passage_id=passage.passage_id,
                        url=passage.url,
                        title=passage.title,
                        section=passage.section,
                        text=(passage.text or "")[:220],
                    )
                    if final_assessment.contradicts_query:
                        contradicting_spans.append(span)
                    else:
                        supporting_spans.append(span)

            synthetic_verification = VerificationResult(
                verdict=verdict,
                confidence=final_assessment.confidence,
                supporting_spans=supporting_spans,
                contradicting_spans=contradicting_spans,
                missing_dimensions=list(final_assessment.gaps),
                rationale=(
                    "unified runner: " + ("contradicts_query" if final_assessment.contradicts_query else f"confidence={final_assessment.confidence:.2f}")
                ),
            )

            synthetic_bundle = EvidenceBundle(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                supporting_passages=[passage_index[n] for kc in final_assessment.key_claims for n in kc.supporting_citation_numbers if n in passage_index and not final_assessment.contradicts_query],
                contradicting_passages=[passage_index[n] for kc in final_assessment.key_claims for n in kc.supporting_citation_numbers if n in passage_index and final_assessment.contradicts_query],
                considered_passages=prompt_passages,
                independent_source_count=len(all_kc_domains),
                has_primary_source=has_primary_source,
                freshness_ok=True,
                verification=synthetic_verification,
                contract_satisfied=verdict == "supported",
                contract_gaps=list(final_assessment.gaps),
            )

            routing_decision = RoutingDecision(
                mode="full",
                certainty=final_assessment.confidence,
                consistency=1.0 if verdict == "supported" else 0.0,
                evidence_sufficiency=1.0 if verdict == "supported" else 0.0,
                rationale="Unified runner",
            )

            claim_run = ClaimRun(
                claim=claim,
                query_variants=all_variants,
                search_snapshots=all_snapshots,
                gated_results=all_gated,
                fetch_plans=all_plans,
                fetched_documents=all_documents,
                passages=prompt_passages,
                evidence_bundle=synthetic_bundle,
                routing_decision=routing_decision,
            )

            audit.query_variants = list(all_variants)
            audit.serp_snapshots = list(all_snapshots)
            audit.selected_urls = [p.url for p in all_plans]
            audit.passage_ids = [p.passage_id for p in prompt_passages]
            audit.claim_to_passages = {claim.claim_id: [p.passage_id for p in prompt_passages]}
            audit.claim_iterations = {claim.claim_id: iterations_used}
            audit.verification_results = {claim.claim_id: synthetic_verification}
            audit.final_verdicts = {claim.claim_id: verdict}

            report = AgentRunResult(
                user_query=query,
                classification=classification,
                claims=[claim_run],
                answer=final_assessment.answer or "",
                audit_trail=audit,
            )

            completed_at = datetime.now(UTC)
            report.audit_trail.completed_at = completed_at.isoformat()
            report.audit_trail.latency_ms = int((completed_at - started_at).total_seconds() * 1000)
            report.audit_trail.estimated_search_cost = self._steps.estimate_search_cost([claim_run])

            resolved_receipts_dir = receipts_dir or (get_settings().agent_receipts_dir or "").strip() or None
            if resolved_receipts_dir:
                report.audit_trail.receipt_path = self._receipt_writer.write(
                    report,
                    resolved_receipts_dir,
                )
            return report

    def _parallel_search(
        self,
        variants: list[QueryVariant],
        profile,
        search_gateway,
        log,
    ) -> list[SearchSnapshot]:
        if not variants:
            return []
        new_snapshots: list[SearchSnapshot] = []
        with ThreadPoolExecutor(max_workers=min(len(variants), 6)) as pool:
            futures = {
                pool.submit(search_gateway.search_variant, v.query_text, profile, log): v
                for v in variants
            }
            for future in as_completed(futures):
                variant = futures[future]
                variant_snapshots = future.result()
                for snapshot in variant_snapshots:
                    new_snapshots.append(self._steps.retag_snapshot(snapshot, variant))
        return new_snapshots


def _merge_gated(
    existing: list[GatedSerpResult],
    new: list[GatedSerpResult],
) -> list[GatedSerpResult]:
    """Union by canonical_url, preserving the higher source_score entry."""
    seen: dict[str, GatedSerpResult] = {}
    for item in existing + new:
        key = item.serp.canonical_url or item.serp.url
        prior = seen.get(key)
        if prior is None or item.assessment.source_score > prior.assessment.source_score:
            seen[key] = item
    return list(seen.values())
