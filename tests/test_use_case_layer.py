import unittest

from search_agent.config.profiles import get_profile
from search_agent.domain.models import (
    Claim,
    ClaimProfile,
    ClaimRun,
    EvidenceBundle,
    FetchedDocument,
    FetchPlan,
    GatedSerpResult,
    Passage,
    QueryClassification,
    QueryVariant,
    RoutingDecision,
    SearchSnapshot,
    SerpResult,
    SourceAssessment,
    VerificationResult,
)
from search_agent.application.use_cases import SearchAgentUseCase, _claim_run_allows_synthesis, _select_synthesis_passages


class _FakeIntelligence:
    def __init__(self):
        self.verify_calls = 0
        self.synthesize_calls = 0

    def classify_query(self, query: str, log=None):
        return QueryClassification(
            query=query,
            normalized_query=query,
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )

    def decompose_claims(self, classification, log=None):
        return [
            Claim(
                claim_id="claim-1",
                claim_text=classification.normalized_query,
                priority=1,
                needs_freshness=False,
                entity_set=["Microsoft"],
            )
        ]

    def verify_claim(self, claim, passages, log=None):
        self.verify_calls += 1
        return VerificationResult(
            verdict="supported",
            confidence=0.98,
            supporting_spans=[],
            rationale="Enough evidence.",
        )

    def synthesize_answer(self, query: str, passages, log=None, intent: str = "synthesis"):
        self.synthesize_calls += 1
        return "Synthesized answer"

    def refine_search_queries(
        self,
        claim,
        classification,
        verification,
        gated_results,
        bundle,
        next_iteration,
        existing_queries,
        log=None,
    ):
        return []


class _FakeSearchGateway:
    def __init__(self):
        self.queries = []

    def search_variant(self, query, profile, log=None):
        self.queries.append((query, profile.name))
        return [
            SearchSnapshot(
                query=query,
                suggestions=[],
                retrieved_at="2026-03-24T00:00:00+00:00",
                results=[
                    SerpResult(
                        result_id="r1",
                        query_variant_id="v1",
                        title="Satya Nadella leadership profile",
                        url="https://news.microsoft.com/source/exec/satya-nadella/",
                        snippet="Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
                        canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
                        host="news.microsoft.com",
                        position=1,
                    )
                ],
                profile_name=profile.name,
            )
        ]


class _FakeFetchGateway:
    def fetch_claim_documents(
        self,
        claim,
        gated_results,
        profile,
        routing_decision,
        *,
        seen_urls,
        log=None,
        iteration=1,
        page_cache=None,
        page_cache_lock=None,
        intent="factual",
    ):
        return (
            [
                FetchPlan(
                    depth="deep",
                    url="https://news.microsoft.com/source/exec/satya-nadella/",
                    reason="top source",
                    source_score=0.95,
                )
            ],
            [
                FetchedDocument(
                    doc_id="doc-1",
                    url="https://news.microsoft.com/source/exec/satya-nadella/",
                    canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
                    host="news.microsoft.com",
                    title="Satya Nadella",
                    author=None,
                    published_at="2026-03-24T00:00:00+00:00",
                    extracted_at="2026-03-24T00:00:00+00:00",
                    content_hash="hash",
                    content="Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
                    fetch_depth="deep",
                    source_score=0.95,
                )
            ],
        )


class _FakeReceiptWriter:
    def __init__(self):
        self.calls = []

    def write(self, report, output_dir: str):
        self.calls.append((report.user_query, output_dir))
        return f"{output_dir}/receipt.json"


class _FakeSteps:
    def build_run_id(self, query, started_at):
        return "test-run"

    def infer_claim_profile(self, claim, classification):
        return claim.claim_profile or ClaimProfile(answer_shape="fact", min_independent_sources=2)

    def build_query_variants(self, claim, classification):
        return [
            QueryVariant(
                variant_id="v1",
                claim_id=claim.claim_id,
                query_text=claim.claim_text,
                strategy="broad",
                rationale="initial",
            )
        ]

    def retag_snapshot(self, snapshot, variant):
        return snapshot

    def gate_serp_results(self, claim, snapshots, limit):
        serp = snapshots[0].results[0]
        return [
            GatedSerpResult(
                serp=serp,
                assessment=SourceAssessment(
                    domain_type="official",
                    source_prior=0.95,
                    primary_source_likelihood=0.95,
                    freshness_score=0.9,
                    seo_spam_risk=0.0,
                    entity_match_score=1.0,
                    semantic_match_score=0.95,
                    source_score=0.97,
                ),
            )
        ]

    def route_claim_retrieval(self, claim, gated_results):
        return RoutingDecision(
            mode="short_path",
            certainty=0.9,
            consistency=0.9,
            evidence_sufficiency=0.9,
            rationale="easy claim",
        )

    def documents_for_passage_extraction(self, documents):
        return documents

    def split_into_passages(self, document):
        return [
            Passage(
                passage_id="p1",
                url=document.url,
                canonical_url=document.canonical_url,
                host=document.host,
                title=document.title,
                section="Leadership",
                published_at=document.published_at,
                author=document.author,
                extracted_at=document.extracted_at,
                chunk_id="chunk-1",
                text=document.content,
                source_score=document.source_score,
                utility_score=0.9,
            )
        ]

    def cheap_passage_filter(self, claim, passages):
        return passages

    def utility_rerank_passages(self, claim, passages):
        return passages

    def build_evidence_bundle(self, claim, passages, verification, gated_results):
        return EvidenceBundle(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            supporting_passages=passages,
            considered_passages=passages,
            independent_source_count=1,
            has_primary_source=True,
            freshness_ok=True,
            verification=verification,
        )

    def refine_query_variants(self, claim, classification, verification, gated_results, bundle, next_iteration, existing_queries):
        return []

    def should_stop_claim_loop(self, claim, bundle, iteration):
        return True

    def compose_answer(self, report):
        return "Supported answer"

    def estimate_search_cost(self, claim_runs):
        return 1.5


class _FakeIntelligenceThreeSameQuery(_FakeIntelligence):
    def decompose_claims(self, classification, log=None):
        return [
            Claim(
                claim_id="c1",
                claim_text="Same query text",
                priority=1,
                needs_freshness=False,
                entity_set=[],
            ),
            Claim(
                claim_id="c2",
                claim_text="Same query text",
                priority=1,
                needs_freshness=False,
                entity_set=[],
            ),
            Claim(
                claim_id="c3",
                claim_text="Same query text",
                priority=1,
                needs_freshness=False,
                entity_set=[],
            ),
        ]


class _FakeContradictionIntelligence(_FakeIntelligence):
    def decompose_claims(self, classification, log=None):
        return [
            Claim(
                claim_id="claim-1",
                claim_text=classification.normalized_query,
                priority=1,
                needs_freshness=False,
                entity_set=["Python"],
            )
        ]

    def verify_claim(self, claim, passages, log=None):
        self.verify_calls += 1
        return VerificationResult(
            verdict="contradicted",
            confidence=0.99,
            contradicting_spans=[],
            rationale="Contradiction is already clear.",
        )


class _FakeContradictionSteps(_FakeSteps):
    def route_claim_retrieval(self, claim, gated_results):
        return RoutingDecision(
            mode="targeted_retrieval",
            certainty=0.9,
            consistency=0.8,
            evidence_sufficiency=0.9,
            rationale="strong initial contradiction evidence",
        )

    def build_evidence_bundle(self, claim, passages, verification, gated_results):
        return EvidenceBundle(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            contradicting_passages=passages,
            considered_passages=passages,
            independent_source_count=2,
            has_primary_source=True,
            freshness_ok=True,
            verification=verification,
        )

    def refine_query_variants(self, claim, classification, verification, gated_results, bundle, next_iteration, existing_queries):
        return [
            QueryVariant(
                variant_id="v2",
                claim_id=claim.claim_id,
                query_text=f'"{claim.claim_text}"',
                strategy="refined_exact",
                rationale="second iteration",
            )
        ]

    def should_stop_claim_loop(self, claim, bundle, iteration):
        return False


class _FakeIterativeEscalationIntelligence(_FakeIntelligence):
    def verify_claim(self, claim, passages, log=None):
        self.verify_calls += 1
        if self.verify_calls == 1:
            return VerificationResult(
                verdict="insufficient_evidence",
                confidence=0.45,
                missing_dimensions=["coverage"],
                rationale="Need another corroborating source.",
            )
        return VerificationResult(
            verdict="supported",
            confidence=0.95,
            supporting_spans=[],
            rationale="Enough corroboration after the second pass.",
        )

    def refine_search_queries(
        self,
        claim,
        classification,
        verification,
        gated_results,
        bundle,
        next_iteration,
        existing_queries,
        log=None,
    ):
        if verification.verdict != "supported":
            return [f"{claim.claim_text} official"]
        return []


class _FakeIterativeEscalationSteps(_FakeSteps):
    def route_claim_retrieval(self, claim, gated_results):
        return RoutingDecision(
            mode="targeted_retrieval",
            certainty=0.88,
            consistency=0.72,
            evidence_sufficiency=0.84,
            rationale="focused retrieval remains sufficient",
        )

    def should_stop_claim_loop(self, claim, bundle, iteration):
        return bundle.verification is not None and bundle.verification.verdict == "supported"


class _FakeSynthesisIntelligence(_FakeIntelligence):
    def classify_query(self, query: str, log=None):
        return QueryClassification(
            query=query,
            normalized_query=query,
            intent="synthesis",
            complexity="single_hop",
            needs_freshness=True,
        )

    def decompose_claims(self, classification, log=None):
        return [
            Claim(
                claim_id="claim-1",
                claim_text=classification.normalized_query,
                priority=1,
                needs_freshness=True,
                entity_set=["MacBook Neo"],
                claim_profile=ClaimProfile(
                    answer_shape="product_specs",
                    primary_source_required=True,
                    min_independent_sources=2,
                    preferred_domain_types=["official", "vendor", "major_media"],
                    routing_bias="iterative_loop",
                    required_dimensions=["source", "specs"],
                    allow_synthesis_without_primary=False,
                    strict_contract=True,
                ),
            )
        ]

    def synthesize_answer(self, query: str, passages, log=None, intent: str = "synthesis"):
        self.synthesize_calls += 1
        return "Synthesized product answer"


class _FakeSynthesisInsufficientIntelligence(_FakeSynthesisIntelligence):
    def verify_claim(self, claim, passages, log=None):
        self.verify_calls += 1
        return VerificationResult(
            verdict="insufficient_evidence",
            confidence=0.2,
            missing_dimensions=["coverage"],
            rationale="Open-ended product overview.",
        )


class _FakeStrictExactNumberIntelligence(_FakeIntelligence):
    def decompose_claims(self, classification, log=None):
        return [
            Claim(
                claim_id="claim-1",
                claim_text="What was the exact room temperature when Satya Nadella was named CEO of Microsoft?",
                priority=1,
                needs_freshness=False,
                entity_set=["Satya Nadella", "Microsoft"],
                claim_profile=ClaimProfile(
                    answer_shape="exact_number",
                    min_independent_sources=2,
                    routing_bias="iterative_loop",
                    required_dimensions=["time", "number", "source"],
                    strict_contract=True,
                ),
            )
        ]

    def verify_claim(self, claim, passages, log=None):
        self.verify_calls += 1
        return VerificationResult(
            verdict="insufficient_evidence",
            confidence=0.95,
            missing_dimensions=["temperature", "number"],
            rationale="Missing measured value.",
        )


class _FakeSynthesisSteps(_FakeSteps):
    def __init__(self, *, contract_satisfied: bool):
        self.contract_satisfied = contract_satisfied

    def infer_claim_profile(self, claim, classification):
        return ClaimProfile(
            answer_shape="product_specs",
            primary_source_required=True,
            min_independent_sources=2,
            preferred_domain_types=["official", "vendor", "major_media"],
            routing_bias="iterative_loop",
            required_dimensions=["source", "specs"],
            allow_synthesis_without_primary=False,
            strict_contract=True,
        )

    def build_evidence_bundle(self, claim, passages, verification, gated_results):
        return EvidenceBundle(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            supporting_passages=passages,
            considered_passages=passages,
            independent_source_count=2 if self.contract_satisfied else 1,
            has_primary_source=self.contract_satisfied,
            freshness_ok=True,
            verification=verification,
            contract_satisfied=self.contract_satisfied,
            contract_gaps=[] if self.contract_satisfied else ["primary_source", "independent_sources"],
        )

    def compose_answer(self, report):
        return "Fallback product answer"


class UseCaseLayerTests(unittest.TestCase):
    def test_select_synthesis_passages_diversifies_news_digest_domains(self):
        passages = [
            Passage(
                passage_id="p1",
                url="https://news.example.com/story-1",
                canonical_url="https://news.example.com/story-1",
                host="news.example.com",
                title="Story 1",
                section="News",
                published_at=None,
                author=None,
                extracted_at="2026-03-27T00:00:00+00:00",
                chunk_id="p1",
                text="Story 1",
                source_score=0.95,
                utility_score=0.7,
            ),
            Passage(
                passage_id="p2",
                url="https://news.example.com/story-2",
                canonical_url="https://news.example.com/story-2",
                host="news.example.com",
                title="Story 2",
                section="News",
                published_at=None,
                author=None,
                extracted_at="2026-03-27T00:00:00+00:00",
                chunk_id="p2",
                text="Story 2",
                source_score=0.94,
                utility_score=0.7,
            ),
            Passage(
                passage_id="p3",
                url="https://wire.example.net/story-3",
                canonical_url="https://wire.example.net/story-3",
                host="wire.example.net",
                title="Story 3",
                section="News",
                published_at=None,
                author=None,
                extracted_at="2026-03-27T00:00:00+00:00",
                chunk_id="p3",
                text="Story 3",
                source_score=0.8,
                utility_score=0.7,
            ),
            Passage(
                passage_id="p4",
                url="https://local.example.org/story-4",
                canonical_url="https://local.example.org/story-4",
                host="local.example.org",
                title="Story 4",
                section="News",
                published_at=None,
                author=None,
                extracted_at="2026-03-27T00:00:00+00:00",
                chunk_id="p4",
                text="Story 4",
                source_score=0.79,
                utility_score=0.7,
            ),
        ]

        selected = _select_synthesis_passages(passages, intent="news_digest", limit=3)

        self.assertEqual([passage.host for passage in selected], ["news.example.com", "wire.example.net", "local.example.org"])

    def test_select_synthesis_passages_diversifies_urls_before_reusing_same_source(self):
        passages = [
            Passage(
                passage_id="p1",
                url="https://docs.python.org/3/whatsnew/3.13.html",
                canonical_url="https://docs.python.org/3/whatsnew/3.13.html",
                host="docs.python.org",
                title="What's New 3.13",
                section="A",
                published_at=None,
                author=None,
                extracted_at="2026-03-27T00:00:00+00:00",
                chunk_id="p1",
                text="Feature A",
                source_score=0.95,
                utility_score=0.9,
            ),
            Passage(
                passage_id="p2",
                url="https://docs.python.org/3/whatsnew/3.13.html",
                canonical_url="https://docs.python.org/3/whatsnew/3.13.html",
                host="docs.python.org",
                title="What's New 3.13",
                section="B",
                published_at=None,
                author=None,
                extracted_at="2026-03-27T00:00:00+00:00",
                chunk_id="p2",
                text="Feature B",
                source_score=0.94,
                utility_score=0.88,
            ),
            Passage(
                passage_id="p3",
                url="https://realpython.com/python313-new-features/",
                canonical_url="https://realpython.com/python313-new-features/",
                host="realpython.com",
                title="Real Python 3.13",
                section="Main",
                published_at=None,
                author=None,
                extracted_at="2026-03-27T00:00:00+00:00",
                chunk_id="p3",
                text="Feature C",
                source_score=0.85,
                utility_score=0.8,
            ),
        ]

        selected = _select_synthesis_passages(passages, intent="synthesis", limit=3)

        self.assertEqual(
            [passage.url for passage in selected[:2]],
            [
                "https://docs.python.org/3/whatsnew/3.13.html",
                "https://realpython.com/python313-new-features/",
            ],
        )

    def test_news_digest_allows_synthesis_from_multiple_fetched_documents(self):
        claim = Claim(
            claim_id="claim-1",
            claim_text="Latest developments in the Iran conflict",
            priority=1,
            needs_freshness=True,
            entity_set=["Iran"],
            claim_profile=ClaimProfile(
                answer_shape="news_digest",
                min_independent_sources=3,
                routing_bias="iterative_loop",
                required_dimensions=["time", "source", "event"],
            ),
        )
        passages = [
            Passage(
                passage_id="p1",
                url="https://apnews.com/hub/iran",
                canonical_url="https://apnews.com/hub/iran",
                host="apnews.com",
                title="Iran live updates",
                section="News",
                published_at="2026-03-27T00:00:00+00:00",
                author=None,
                extracted_at="2026-03-27T00:00:00+00:00",
                chunk_id="p1",
                text="Update",
                source_score=0.9,
                utility_score=0.8,
            )
        ]
        bundle = EvidenceBundle(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            considered_passages=passages,
            independent_source_count=1,
            has_primary_source=False,
            freshness_ok=True,
            verification=VerificationResult(
                verdict="insufficient_evidence",
                confidence=0.9,
                missing_dimensions=["coverage"],
                rationale="Need more corroboration.",
            ),
        )
        claim_run = ClaimRun(
            claim=claim,
            fetched_documents=[
                FetchedDocument(
                    doc_id="doc-1",
                    url="https://apnews.com/hub/iran",
                    canonical_url="https://apnews.com/hub/iran",
                    host="apnews.com",
                    title="Iran live updates",
                    author=None,
                    published_at="2026-03-27T00:00:00+00:00",
                    extracted_at="2026-03-27T00:00:00+00:00",
                    content_hash="h1",
                    content="Update",
                    fetch_depth="deep",
                    source_score=0.9,
                ),
                FetchedDocument(
                    doc_id="doc-2",
                    url="https://www.npr.org/2026/03/25/example",
                    canonical_url="https://www.npr.org/2026/03/25/example",
                    host="www.npr.org",
                    title="NPR update",
                    author=None,
                    published_at="2026-03-27T00:00:00+00:00",
                    extracted_at="2026-03-27T00:00:00+00:00",
                    content_hash="h2",
                    content="Update",
                    fetch_depth="deep",
                    source_score=0.85,
                ),
            ],
            evidence_bundle=bundle,
        )

        self.assertTrue(_claim_run_allows_synthesis(claim_run))

    def test_use_case_keeps_iterative_route_for_strict_exact_number_contract(self):
        use_case = SearchAgentUseCase(
            intelligence=_FakeStrictExactNumberIntelligence(),
            search_gateway=_FakeSearchGateway(),
            fetch_gateway=_FakeFetchGateway(),
            receipt_writer=_FakeReceiptWriter(),
            steps=_FakeSteps(),
        )

        report = use_case.run("room temperature claim", get_profile("web"))

        self.assertEqual(report.claims[0].routing_decision.mode, "iterative_loop")

    def test_use_case_dedupes_search_across_parallel_claims(self):
        search_gateway = _FakeSearchGateway()
        intelligence = _FakeIntelligenceThreeSameQuery()
        use_case = SearchAgentUseCase(
            intelligence=intelligence,
            search_gateway=search_gateway,
            fetch_gateway=_FakeFetchGateway(),
            receipt_writer=_FakeReceiptWriter(),
            steps=_FakeSteps(),
        )
        report = use_case.run("Q", get_profile("web"), receipts_dir=None)
        self.assertEqual(len(search_gateway.queries), 1)
        self.assertEqual(intelligence.verify_calls, 3)
        ids = [run.claim.claim_id for run in report.claims]
        self.assertEqual(ids, ["c1", "c2", "c3"])

    def test_use_case_runs_single_claim_flow_and_writes_receipt(self):
        receipt_writer = _FakeReceiptWriter()
        search_gateway = _FakeSearchGateway()
        intelligence = _FakeIntelligence()
        use_case = SearchAgentUseCase(
            intelligence=intelligence,
            search_gateway=search_gateway,
            fetch_gateway=_FakeFetchGateway(),
            receipt_writer=receipt_writer,
            steps=_FakeSteps(),
        )

        report = use_case.run(
            "Who is the CEO of Microsoft?",
            get_profile("web"),
            receipts_dir="receipts",
        )

        self.assertEqual(report.answer, "Supported answer")
        self.assertEqual(report.audit_trail.run_id, "test-run")
        self.assertEqual(report.audit_trail.receipt_path, "receipts/receipt.json")
        self.assertEqual(report.audit_trail.final_verdicts["claim-1"], "supported")
        self.assertEqual(search_gateway.queries[0], ("Who is the CEO of Microsoft?", "web"))
        self.assertEqual(intelligence.verify_calls, 1)
        self.assertEqual(receipt_writer.calls[0][1], "receipts")

    def test_use_case_reports_best_non_iterative_route_after_successful_escalation(self):
        use_case = SearchAgentUseCase(
            intelligence=_FakeIterativeEscalationIntelligence(),
            search_gateway=_FakeSearchGateway(),
            fetch_gateway=_FakeFetchGateway(),
            receipt_writer=_FakeReceiptWriter(),
            steps=_FakeIterativeEscalationSteps(),
        )

        report = use_case.run(
            "Who is the CEO of Microsoft?",
            get_profile("web"),
            receipts_dir=None,
        )

        self.assertEqual(report.claims[0].routing_decision.mode, "targeted_retrieval")

    def test_use_case_stops_after_strong_contradiction_without_second_iteration(self):
        receipt_writer = _FakeReceiptWriter()
        search_gateway = _FakeSearchGateway()
        intelligence = _FakeContradictionIntelligence()
        use_case = SearchAgentUseCase(
            intelligence=intelligence,
            search_gateway=search_gateway,
            fetch_gateway=_FakeFetchGateway(),
            receipt_writer=receipt_writer,
            steps=_FakeContradictionSteps(),
        )

        report = use_case.run(
            "Was Python 3.13.0 released on October 1, 2024?",
            get_profile("web"),
            receipts_dir=None,
        )

        self.assertEqual(len(search_gateway.queries), 1)
        self.assertEqual(intelligence.verify_calls, 1)
        self.assertEqual(report.claims[0].routing_decision.mode, "iterative_loop")

    def test_use_case_skips_synthesis_when_strict_claim_contract_is_not_satisfied(self):
        receipt_writer = _FakeReceiptWriter()
        search_gateway = _FakeSearchGateway()
        intelligence = _FakeSynthesisIntelligence()
        use_case = SearchAgentUseCase(
            intelligence=intelligence,
            search_gateway=search_gateway,
            fetch_gateway=_FakeFetchGateway(),
            receipt_writer=receipt_writer,
            steps=_FakeSynthesisSteps(contract_satisfied=False),
        )

        report = use_case.run(
            "Какие характеристики нового MacBook Neo?",
            get_profile("web"),
            receipts_dir=None,
        )

        self.assertEqual(intelligence.synthesize_calls, 0)
        self.assertEqual(report.answer, "Fallback product answer")
        self.assertEqual(report.claims[0].claim.claim_profile.answer_shape, "product_specs")

    def test_use_case_allows_synthesis_when_strict_claim_contract_is_satisfied(self):
        receipt_writer = _FakeReceiptWriter()
        search_gateway = _FakeSearchGateway()
        intelligence = _FakeSynthesisIntelligence()
        use_case = SearchAgentUseCase(
            intelligence=intelligence,
            search_gateway=search_gateway,
            fetch_gateway=_FakeFetchGateway(),
            receipt_writer=receipt_writer,
            steps=_FakeSynthesisSteps(contract_satisfied=True),
        )

        report = use_case.run(
            "Какие характеристики нового MacBook Neo?",
            get_profile("web"),
            receipts_dir=None,
        )

        self.assertEqual(intelligence.synthesize_calls, 1)
        self.assertEqual(report.answer, "Synthesized product answer")
        self.assertTrue(report.claims[0].evidence_bundle.contract_satisfied)

    def test_use_case_allows_synthesis_when_contract_is_satisfied_but_verdict_is_insufficient(self):
        receipt_writer = _FakeReceiptWriter()
        search_gateway = _FakeSearchGateway()
        intelligence = _FakeSynthesisInsufficientIntelligence()
        use_case = SearchAgentUseCase(
            intelligence=intelligence,
            search_gateway=search_gateway,
            fetch_gateway=_FakeFetchGateway(),
            receipt_writer=receipt_writer,
            steps=_FakeSynthesisSteps(contract_satisfied=True),
        )

        report = use_case.run(
            "Какие характеристики нового MacBook Neo?",
            get_profile("web"),
            receipts_dir=None,
        )

        self.assertEqual(intelligence.verify_calls, 1)
        self.assertEqual(intelligence.synthesize_calls, 1)
        self.assertEqual(report.answer, "Synthesized product answer")
        self.assertTrue(report.claims[0].evidence_bundle.contract_satisfied)


if __name__ == "__main__":
    unittest.main()
