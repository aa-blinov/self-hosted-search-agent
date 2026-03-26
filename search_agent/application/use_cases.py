from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime

import logfire

from search_agent import tuning
from search_agent.infrastructure.caching_search_gateway import CachingBudgetSearchGateway
from search_agent.settings import get_settings
from search_agent.domain.models import AgentRunResult, AuditTrail, ClaimRun, EvidenceBundle, RoutingDecision

from search_agent.application.contracts import (
    FetchGatewayPort,
    QueryIntelligencePort,
    ReceiptWriterPort,
    SearchGatewayPort,
    StepLibraryPort,
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
                    self._extend_audit(audit, claim_run, iterations_used)
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
                    self._extend_audit(audit, claim_run, iterations_used)

            report = AgentRunResult(
                user_query=query,
                classification=classification,
                claims=claim_runs,
                answer="",
                audit_trail=audit,
            )
            report.answer = self._steps.compose_answer(report)

            # For comparison/synthesis queries, synthesize a direct answer from all
            # collected passages — verify_claim reliably returns insufficient_evidence
            # for open-ended comparison questions even with rich evidence, so compose_answer
            # would produce a nearly empty response.
            if classification.intent == "comparison":
                synth_passages: list = []
                for cr in claim_runs:
                    # Re-extract all split passages from fetched documents for richer context.
                    # Pass ALL cheap-filtered passages (not deduplicated by URL) so that
                    # the synthesis LLM receives multiple informative chunks from the same
                    # document (e.g. different sections of docs.python.org/whatsnew).
                    raw: list = []
                    for doc in (cr.fetched_documents or []):
                        raw.extend(self._steps.split_into_passages(doc))
                    cheap = self._steps.cheap_passage_filter(cr.claim, raw)
                    synth_passages.extend(cheap)
                if synth_passages:
                    synthesis = self._intelligence.synthesize_answer(query, synth_passages, log=log)
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
        gated_results = []
        fetch_plans = []
        documents = []
        final_passages = []
        routing_decision = RoutingDecision(
            mode="iterative_loop",
            certainty=0.0,
            consistency=0.0,
            evidence_sufficiency=0.0,
            rationale="Not evaluated yet.",
        )
        bundle: EvidenceBundle | None = None

        existing_queries: set[str] = set()
        seen_urls: set[str] = set()
        seen_documents: set[tuple[str, str, str]] = set()
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

            new_snapshots = []
            with ThreadPoolExecutor(max_workers=min(len(next_variants), 6)) as serp_pool:
                future_to_variant = {
                    serp_pool.submit(search_gateway.search_variant, v.query_text, profile, log): v
                    for v in next_variants
                }
                for future in as_completed(future_to_variant):
                    variant = future_to_variant[future]
                    variant_snapshots = future.result()
                    new_snapshots.extend(
                        self._steps.retag_snapshot(snapshot, variant) for snapshot in variant_snapshots
                    )
            snapshots.extend(new_snapshots)

            gated_limit = min(tuning.SERP_GATE_MAX_URLS, max(tuning.SERP_GATE_MIN_URLS, profile.max_results))
            gated_results = self._steps.gate_serp_results(claim, snapshots, gated_limit)
            routing_decision = self._steps.route_claim_retrieval(claim, gated_results)
            if bundle and bundle.verification and bundle.verification.verdict != "supported":
                if routing_decision.mode == "short_path":
                    routing_decision = replace(
                        routing_decision,
                        mode="targeted_retrieval",
                        rationale=routing_decision.rationale + " | escalated after weak verification",
                    )
                elif iteration > 1:
                    routing_decision = replace(
                        routing_decision,
                        mode="iterative_loop",
                        rationale=routing_decision.rationale + " | iterative escalation",
                    )

            log(
                f"  [dim]route={routing_decision.mode} | "
                f"certainty={routing_decision.certainty:.2f} | "
                f"consistency={routing_decision.consistency:.2f} | "
                f"sufficiency={routing_decision.evidence_sufficiency:.2f}[/dim]"
            )
            log(f"  [dim]SERP gate kept {len(gated_results)} URLs[/dim]")

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
            )
            fetch_plans.extend(new_fetch_plans)
            for document in new_documents:
                key = (document.url, document.fetch_depth, document.content_hash)
                if key in seen_documents:
                    continue
                seen_documents.add(key)
                seen_urls.add(document.url)
                documents.append(document)

            passage_documents = self._steps.documents_for_passage_extraction(documents)
            passages = []
            for document in passage_documents:
                passages.extend(self._steps.split_into_passages(document))

            cheap_filtered = self._steps.cheap_passage_filter(claim, passages)
            final_passages = self._steps.utility_rerank_passages(claim, cheap_filtered)
            log(
                f"  [dim]passages: {len(passages)} total | "
                f"{len(cheap_filtered)} after cheap filter | "
                f"{len(final_passages)} after utility rerank[/dim]"
            )

            verification = self._intelligence.verify_claim(claim, final_passages, log=log)
            bundle = self._steps.build_evidence_bundle(claim, final_passages, verification, gated_results)
            log(
                f"  [dim]verdict={verification.verdict} | "
                f"confidence={verification.confidence:.2f} | "
                f"independent_sources={bundle.independent_source_count}[/dim]"
            )

            if self._steps.should_stop_claim_loop(claim, bundle, iteration):
                break

            next_variants = self._steps.refine_query_variants(
                claim,
                classification,
                verification,
                gated_results,
                bundle,
                iteration + 1,
                existing_queries,
            )

        claim_run = ClaimRun(
            claim=claim,
            query_variants=all_variants,
            search_snapshots=snapshots,
            gated_results=gated_results,
            fetch_plans=fetch_plans,
            fetched_documents=documents,
            passages=final_passages,
            evidence_bundle=bundle,
            routing_decision=routing_decision,
        )
        return claim_run, iterations_used

    @staticmethod
    def _extend_audit(audit: AuditTrail, claim_run: ClaimRun, iterations_used: int) -> None:
        claim = claim_run.claim
        bundle = claim_run.evidence_bundle
        audit.query_variants.extend(claim_run.query_variants)
        audit.serp_snapshots.extend(claim_run.search_snapshots)
        audit.selected_urls.extend([result.serp.url for result in claim_run.gated_results])
        audit.crawl_events.extend(
            {
                "claim_id": claim.claim_id,
                "url": document.url,
                "fetched_at": document.extracted_at,
                "content_hash": document.content_hash,
                "fetch_depth": document.fetch_depth,
            }
            for document in claim_run.fetched_documents
        )
        audit.passage_ids.extend([passage.passage_id for passage in claim_run.passages])
        audit.claim_to_passages[claim.claim_id] = [passage.passage_id for passage in claim_run.passages]
        audit.claim_iterations[claim.claim_id] = iterations_used
        if bundle and bundle.verification:
            audit.verification_results[claim.claim_id] = bundle.verification
            audit.final_verdicts[claim.claim_id] = bundle.verification.verdict
