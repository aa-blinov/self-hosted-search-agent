import unittest

from search_agent.config.profiles import get_profile
from search_agent.domain.models import (
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
    SerpResult,
    SourceAssessment,
    VerificationResult,
)
from search_agent.application.use_cases import SearchAgentUseCase


class _FakeIntelligence:
    def __init__(self):
        self.verify_calls = 0

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


class UseCaseLayerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
