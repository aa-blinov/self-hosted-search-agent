from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime

import logfire

from search_agent import tuning
from search_agent.infrastructure.caching_search_gateway import CachingBudgetSearchGateway
from search_agent.settings import get_settings
from search_agent.domain.models import AgentRunResult, AuditTrail, ClaimRun, EvidenceBundle, GatedSerpResult, QueryVariant, RoutingDecision

from search_agent.application.contracts import (
    FetchGatewayPort,
    QueryIntelligencePort,
    ReceiptWriterPort,
    SearchGatewayPort,
    StepLibraryPort,
)
from search_agent.application.use_case_support import (
    claim_run_allows_synthesis as _claim_run_allows_synthesis,
    extend_audit as _extend_audit,
    merge_gated_results as _merge_gated_results,
    reconcile_classification_with_claims as _reconcile_classification_with_claims,
    select_synthesis_passages as _select_synthesis_passages,
)


class SearchAgentUseCase:
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
            "search_agent.run",
            query=query,
            profile=getattr(profile, "name", None),
        ):
            classification = self._intelligence.classify_query(query, log=log)
            claims = self._intelligence.decompose_claims(classification, log=log)
            if tuning.DECOMPOSE_MAX_CLAIMS > 0:
                claims = claims[:tuning.DECOMPOSE_MAX_CLAIMS]
            classification = _reconcile_classification_with_claims(classification, claims)
            log(f"\n[bold]Agent Search[/bold] [dim]{len(claims)} claim(s)[/dim]")

            provider = get_settings().resolved_search_provider()
            search_gateway = CachingBudgetSearchGateway(
                self._search_gateway,
                provider_label=provider,
            )

            claim_runs: list[ClaimRun] = []
            audit = AuditTrail(
                run_id=self._steps.build_run_id(query, started_at),
                profile_name=getattr(profile, "name", None),
                started_at=started_at.isoformat(),
            )

            # Shared page-content cache: prevents the same URL being HTTP-fetched
            # multiple times across claims within the same run.
            page_cache: dict[str, dict] = {}
            page_cache_lock = threading.Lock()

            workers = max(1, min(tuning.AGENT_MAX_PARALLEL_CLAIMS, len(claims)))
            log_lock = threading.Lock()

            def safe_log(msg: str) -> None:
                if workers > 1:
                    with log_lock:
                        log(msg)
                else:
                    log(msg)

            if workers == 1 or len(claims) <= 1:
                for claim in claims:
                    with logfire.span(
                        "search_agent.claim",
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                    ):
                        claim_run, iterations_used = self._run_claim(
                            claim,
                            classification,
                            profile,
                            search_gateway=search_gateway,
                            log=safe_log,
                            page_cache=page_cache,
                            page_cache_lock=page_cache_lock,
                        )
                    claim_runs.append(claim_run)
                    _extend_audit(audit, claim_run, iterations_used)
            else:
                future_to_idx: dict[Future, int] = {}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for idx, claim in enumerate(claims):
                        fut = pool.submit(
                            self._run_claim_with_span,
                            claim,
                            classification,
                            profile,
                            search_gateway,
                            safe_log,
                            page_cache,
                            page_cache_lock,
                        )
                        future_to_idx[fut] = idx
                    tmp: list[tuple[ClaimRun, int] | None] = [None] * len(claims)
                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        tmp[idx] = fut.result()
                for row in tmp:
                    assert row is not None
                    claim_run, iterations_used = row
                    claim_runs.append(claim_run)
                    _extend_audit(audit, claim_run, iterations_used)

            report = AgentRunResult(
                user_query=query,
                classification=classification,
                claims=claim_runs,
                answer="",
                audit_trail=audit,
            )
            report.answer = self._steps.compose_answer(report)

            # For synthesis and news_digest queries, synthesize a direct answer from
            # all collected passages — verify_claim reliably returns
            # insufficient_evidence for open-ended questions, so compose_answer
            # would produce a nearly empty response.
            if classification.intent in ("synthesis", "news_digest"):
                synth_passages: list = []
                for cr in claim_runs:
                    if not _claim_run_allows_synthesis(cr):
                        continue
                    # Re-extract all split passages from fetched documents.
                    # For synthesis we skip TF-IDF threshold filtering: the query may be
                    # in a different language from the content (e.g. Russian query vs
                    # English docs.python.org), so keyword overlap scores are unreliable.
                    # Instead rank by source_score (authoritative pages first) and take
                    # the top SYNTHESIS_PASSAGE_LIMIT passages in document order within
                    # each source.  This ensures diverse section coverage (f-strings,
                    # TypeVar, per-interpreter GIL, etc.) rather than only intro paragraphs
                    # that happen to repeat both version numbers.
                    raw: list = []
                    for doc in (cr.fetched_documents or []):
                        raw.extend(self._steps.split_into_passages(doc))
                    raw.sort(key=lambda p: p.source_score, reverse=True)
                    synth_passages.extend(
                        _select_synthesis_passages(
                            raw,
                            intent=classification.intent,
                            limit=tuning.SYNTHESIS_PASSAGE_LIMIT,
                        )
                    )
                if synth_passages:
                    synthesis = self._intelligence.synthesize_answer(query, synth_passages, log=log, intent=classification.intent)
                    if synthesis:
                        report = replace(report, answer=synthesis)
            completed_at = datetime.now(UTC)
            report.audit_trail.completed_at = completed_at.isoformat()
            report.audit_trail.latency_ms = int((completed_at - started_at).total_seconds() * 1000)
            report.audit_trail.estimated_search_cost = self._steps.estimate_search_cost(claim_runs)

            resolved_receipts_dir = receipts_dir or (get_settings().agent_receipts_dir or "").strip() or None
            if resolved_receipts_dir:
                report.audit_trail.receipt_path = self._receipt_writer.write(
                    report,
                    resolved_receipts_dir,
                )
            return report

    def _run_claim_with_span(
        self,
        claim,
        classification,
        profile,
        search_gateway,
        log,
        page_cache=None,
        page_cache_lock=None,
    ) -> tuple[ClaimRun, int]:
        with logfire.span(
            "search_agent.claim",
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
        ):
            return self._run_claim(
                claim,
                classification,
                profile,
                search_gateway=search_gateway,
                log=log,
                page_cache=page_cache,
                page_cache_lock=page_cache_lock,
            )

    def _search_iteration_snapshots(
        self,
        next_variants,
        profile,
        search_gateway,
        log,
    ) -> list:
        new_snapshots = []
        with ThreadPoolExecutor(max_workers=min(len(next_variants), 6)) as serp_pool:
            future_to_variant = {
                serp_pool.submit(search_gateway.search_variant, variant.query_text, profile, log): variant
                for variant in next_variants
            }
            for future in as_completed(future_to_variant):
                variant = future_to_variant[future]
                variant_snapshots = future.result()
                new_snapshots.extend(
                    self._steps.retag_snapshot(snapshot, variant) for snapshot in variant_snapshots
                )
        return new_snapshots

    def _route_iteration(
        self,
        claim,
        classification,
        gated_results,
        bundle: EvidenceBundle | None,
        iteration: int,
        reported_routing_decision: RoutingDecision,
    ) -> tuple[RoutingDecision, RoutingDecision]:
        base_routing_decision = self._steps.route_claim_retrieval(claim, gated_results)
        # iter1 = fast, iter2+ = full (escalated by should_stop_claim_loop)
        mode = "fast" if iteration == 1 else "full"
        routing_decision = replace(base_routing_decision, mode=mode)
        return routing_decision, routing_decision

    def _fetch_iteration_documents(
        self,
        claim,
        classification,
        profile,
        routing_decision,
        gated_results,
        *,
        seen_urls,
        seen_documents,
        documents,
        log,
        iteration: int,
        page_cache=None,
        page_cache_lock=None,
    ) -> list:
        new_fetch_plans, new_documents = self._fetch_gateway.fetch_claim_documents(
            claim,
            gated_results,
            profile,
            routing_decision,
            seen_urls=seen_urls,
            log=log,
            iteration=iteration,
            page_cache=page_cache,
            page_cache_lock=page_cache_lock,
            intent=classification.intent,
        )
        for document in new_documents:
            key = (document.url, document.fetch_depth, document.content_hash)
            if key in seen_documents:
                continue
            seen_documents.add(key)
            seen_urls.add(document.url)
            documents.append(document)
        return new_fetch_plans

    def _collect_iteration_passages(self, claim, documents, log, prior_passage_ids: set[str] | None = None) -> list:
        passage_documents = self._steps.documents_for_passage_extraction(documents)
        passages = []
        for document in passage_documents:
            passages.extend(self._steps.split_into_passages(document))

        cheap_filtered = self._steps.cheap_passage_filter(claim, passages)
        final_passages = self._steps.utility_rerank_passages(claim, cheap_filtered, prior_passage_ids=prior_passage_ids)
        log(
            f"  [dim]passages: {len(passages)} total | "
            f"{len(cheap_filtered)} after cheap filter | "
            f"{len(final_passages)} after utility rerank[/dim]"
        )
        return final_passages

    def _verify_iteration_bundle(
        self,
        claim,
        final_passages,
        all_gated_results,
        log,
    ) -> EvidenceBundle:
        verification = self._intelligence.verify_claim(claim, final_passages, log=log)
        bundle = self._steps.build_evidence_bundle(claim, final_passages, verification, all_gated_results)
        log(
            f"  [dim]verdict={verification.verdict} | "
            f"confidence={verification.confidence:.2f} | "
            f"independent_sources={bundle.independent_source_count}[/dim]"
        )
        return bundle

    def _refine_iteration_variants(
        self,
        claim,
        classification,
        bundle: EvidenceBundle,
        all_gated_results,
        existing_queries,
        iteration: int,
        log,
    ) -> list[QueryVariant]:
        verification = bundle.verification
        assert verification is not None
        next_queries = self._intelligence.refine_search_queries(
            claim,
            classification,
            verification,
            all_gated_results,
            bundle,
            iteration + 1,
            existing_queries,
            log=log,
        )
        if not next_queries:
            return []
        return self._steps.build_query_variants(
            replace(claim, search_queries=next_queries),
            classification,
        )

    def _run_claim(
        self,
        claim,
        classification,
        profile,
        *,
        search_gateway,
        log=None,
        page_cache=None,
        page_cache_lock=None,
    ) -> tuple[ClaimRun, int]:
        log = log or (lambda msg: None)
        log(f"\n[bold]  Claim[/bold] [italic]{claim.claim_text}[/italic]")

        all_variants = []
        snapshots = []
        all_gated_results: list[GatedSerpResult] = []
        fetch_plans = []
        documents = []
        final_passages = []
        routing_decision = RoutingDecision(
            mode="fast",
            certainty=0.0,
            consistency=0.0,
            evidence_sufficiency=0.0,
            rationale="Not evaluated yet.",
        )
        bundle: EvidenceBundle | None = None

        existing_queries: set[str] = set()
        seen_urls: set[str] = set()
        seen_documents: set[tuple[str, str, str]] = set()
        prior_passage_ids: set[str] = set()
        next_variants = self._steps.build_query_variants(claim, classification)[:tuning.AGENT_MAX_QUERY_VARIANTS_ITER1]

        iterations_used = 0
        claim_cap = tuning.AGENT_MAX_CLAIM_ITERATIONS
        for iteration in range(1, claim_cap + 1):
            if not next_variants:
                break
            iterations_used = iteration

            log(f"  [bold]Iteration {iteration}/{claim_cap}[/bold]")
            for variant in next_variants:
                log(f"  [dim]-> {variant.strategy}: {variant.query_text}[/dim]")
                existing_queries.add(variant.query_text.casefold())
            all_variants.extend(next_variants)

            new_snapshots = self._search_iteration_snapshots(
                next_variants,
                profile,
                search_gateway,
                log,
            )
            snapshots.extend(new_snapshots)

            gated_limit = min(tuning.SERP_GATE_MAX_URLS, max(tuning.SERP_GATE_MIN_URLS, profile.max_results))
            gated_results = self._steps.gate_serp_results(claim, snapshots, gated_limit)
            all_gated_results = _merge_gated_results(all_gated_results, gated_results)
            routing_decision, _ = self._route_iteration(
                claim,
                classification,
                gated_results,
                bundle,
                iteration,
                routing_decision,
            )

            log(
                f"  [dim]route={routing_decision.mode} | "
                f"certainty={routing_decision.certainty:.2f} | "
                f"consistency={routing_decision.consistency:.2f} | "
                f"sufficiency={routing_decision.evidence_sufficiency:.2f}[/dim]"
            )
            log(f"  [dim]SERP gate kept {len(gated_results)} URLs[/dim]")

            fetch_plans.extend(
                self._fetch_iteration_documents(
                    claim,
                    classification,
                    profile,
                    routing_decision,
                    gated_results,
                    seen_urls=seen_urls,
                    seen_documents=seen_documents,
                    documents=documents,
                    log=log,
                    iteration=iteration,
                    page_cache=page_cache,
                    page_cache_lock=page_cache_lock,
                )
            )

            final_passages = self._collect_iteration_passages(
                claim, documents, log,
                prior_passage_ids=prior_passage_ids if iteration > 1 else None,
            )
            prior_passage_ids.update(p.passage_id for p in final_passages)
            bundle = self._verify_iteration_bundle(
                claim,
                final_passages,
                all_gated_results,
                log,
            )
            verification = bundle.verification
            assert verification is not None

            if (
                verification.verdict == "contradicted"
                and verification.confidence >= 0.9
                and bundle.independent_source_count >= 2
            ):
                break

            if self._steps.should_stop_claim_loop(claim, bundle, iteration):
                break

            next_variants = self._refine_iteration_variants(
                claim,
                classification,
                bundle,
                all_gated_results,
                existing_queries,
                iteration,
                log,
            )

        claim_run = ClaimRun(
            claim=claim,
            query_variants=all_variants,
            search_snapshots=snapshots,
            gated_results=all_gated_results,
            fetch_plans=fetch_plans,
            fetched_documents=documents,
            passages=final_passages,
            evidence_bundle=bundle,
            routing_decision=routing_decision,
        )
        return claim_run, iterations_used
