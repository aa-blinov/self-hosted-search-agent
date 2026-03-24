import json
import tempfile
import unittest
from pathlib import Path

from search_agent.domain.models import (
    AgentRunResult,
    AuditTrail,
    Claim,
    ClaimRun,
    EvidenceBundle,
    Passage,
    QueryClassification,
    VerificationResult,
)
from search_agent.evaluation import EvaluationCase, ExpectedClaim, load_evaluation_cases, score_reports
from search_agent.infrastructure.receipts import build_receipt_payload, write_receipt


def _make_supported_report() -> AgentRunResult:
    passage = Passage(
        passage_id="p1",
        url="https://news.microsoft.com/announcement/satya-nadella-named-ceo/",
        canonical_url="https://news.microsoft.com/announcement/satya-nadella-named-ceo/",
        host="news.microsoft.com",
        title="Satya Nadella named CEO",
        section="Intro",
        published_at="2024-02-04T00:00:00+00:00",
        author=None,
        extracted_at="2026-03-24T00:00:00+00:00",
        chunk_id="p1",
        text="Longtime Microsoft executive Satya Nadella was named CEO.",
        source_score=0.95,
        utility_score=0.9,
    )
    bundle = EvidenceBundle(
        claim_id="claim-1",
        claim_text="Who is the CEO of Microsoft?",
        supporting_passages=[passage],
        considered_passages=[passage],
        independent_source_count=2,
        has_primary_source=True,
        freshness_ok=True,
        verification=VerificationResult(verdict="supported", confidence=0.98),
    )
    return AgentRunResult(
        user_query="Who is the CEO of Microsoft?",
        classification=QueryClassification(
            query="Who is the CEO of Microsoft?",
            normalized_query="Who is the CEO of Microsoft?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        ),
        claims=[
            ClaimRun(
                claim=Claim(
                    claim_id="claim-1",
                    claim_text="Who is the CEO of Microsoft?",
                    priority=1,
                    needs_freshness=False,
                    entity_set=["Microsoft"],
                ),
                passages=[passage],
                evidence_bundle=bundle,
            )
        ],
        answer=(
            "Ответ\n"
            "- Longtime Microsoft executive Satya Nadella was named CEO [1]\n\n"
            "Источники\n"
            "[1] Satya Nadella named CEO — https://news.microsoft.com/announcement/satya-nadella-named-ceo/\n"
        ),
        audit_trail=AuditTrail(
            run_id="test-run",
            profile_name="web",
            started_at="2026-03-24T00:00:00+00:00",
            completed_at="2026-03-24T00:00:01+00:00",
            latency_ms=1000,
            estimated_search_cost=4.5,
            final_verdicts={"claim-1": "supported"},
        ),
    )


class Phase3Tests(unittest.TestCase):
    def test_write_receipt_persists_json(self):
        report = _make_supported_report()
        payload = build_receipt_payload(report)
        self.assertEqual(payload["summary"]["supported_claims"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_receipt(report, output_dir=tmpdir)
            saved = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(saved["run_id"], "test-run")
        self.assertEqual(saved["summary"]["supported_claims"], 1)
        self.assertEqual(saved["audit_trail"]["profile_name"], "web")

    def test_evaluation_score_reports_computes_metrics(self):
        report = _make_supported_report()
        cases = [
            EvaluationCase(
                case_id="case-1",
                split="factual_single-hop",
                query=report.user_query,
                expected_claims=[
                    ExpectedClaim(
                        match="CEO of Microsoft",
                        expected_verdict="supported",
                        requires_primary_source=True,
                        expected_routes=["targeted_retrieval"],
                        min_independent_sources=2,
                    )
                ],
            )
        ]
        report.claims[0].routing_decision = type("Route", (), {"mode": "targeted_retrieval"})()
        summary = score_reports(cases, {"case-1": report}, {"case-1": 1000})

        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["metrics"]["claim_support_rate"], 1.0)
        self.assertEqual(summary["metrics"]["citation_validity_rate"], 1.0)
        self.assertEqual(summary["metrics"]["unsupported_statement_rate"], 0.0)
        self.assertEqual(summary["metrics"]["primary_source_coverage"], 1.0)
        self.assertEqual(summary["metrics"]["route_match_rate"], 1.0)
        self.assertEqual(summary["metrics"]["primary_requirement_rate"], 1.0)
        self.assertEqual(summary["metrics"]["source_requirement_rate"], 1.0)
        self.assertEqual(summary["metrics"]["median_answer_latency"], 1000)
        self.assertEqual(summary["cases"][0]["claims"][0]["actual_route"], "targeted_retrieval")

    def test_load_evaluation_cases_reads_split_dataset(self):
        cases = load_evaluation_cases("eval_data/sample_cases.jsonl")

        self.assertGreaterEqual(len(cases), 5)
        self.assertTrue(any(case.split == "conflicting-web-sources" for case in cases))
        self.assertTrue(any(case.split == "entity-disambiguation" for case in cases))

    def test_load_control_dataset_reads_route_expectations(self):
        cases = load_evaluation_cases("eval_data/control_dataset.jsonl")

        self.assertGreaterEqual(len(cases), 8)
        microsoft_case = next(case for case in cases if case.case_id == "microsoft-ceo")
        self.assertIn("targeted_retrieval", microsoft_case.expected_claims[0].expected_routes)
        self.assertEqual(microsoft_case.expected_claims[0].min_independent_sources, 2)


if __name__ == "__main__":
    unittest.main()
