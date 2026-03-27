import unittest
from unittest.mock import patch

from search_agent.application.text_heuristics import normalize_relative_time_references
from search_agent.domain.models import Claim, Passage, QueryClassification
from search_agent.infrastructure.intelligence import (
    PydanticAIQueryIntelligence,
    _ClaimDecompositionOutput,
    _ClaimDraft,
    _ClaimProfileOutput,
    _IntentOutput,
    _NormalizedQueryOutput,
    _VerificationOutput,
    _EvidenceQuote,
)
from search_agent.settings import AppSettings


class IntelligenceLayerTests(unittest.TestCase):
    def test_classify_query_uses_deterministic_relative_date_normalization(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))

        with patch("search_agent.infrastructure.intelligence.normalize_relative_time_references", return_value="OpenAI revenue 2026-03-25"):
            with patch.object(service._normalize_agent, "run_sync") as normalize_call:
                with patch.object(service, "_classify_intent_llm", return_value=_IntentOutput(intent="factual", complexity="single_hop")):
                    classification = service.classify_query("OpenAI revenue today")

        self.assertEqual(classification.normalized_query, "OpenAI revenue 2026-03-25")
        self.assertEqual(classification.time_scope, "2026-03-25")
        normalize_call.assert_not_called()

    def test_classify_query_marks_news_digest_intent(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))

        with patch("search_agent.infrastructure.intelligence.normalize_relative_time_references", return_value="что 2026-03-25 было в Астане"):
            with patch.object(service, "_classify_intent_llm", return_value=_IntentOutput(intent="news_digest", complexity="single_hop")):
                classification = service.classify_query("что сегодня было в Астане")

        self.assertEqual(classification.intent, "news_digest")
        self.assertEqual(classification.time_scope, "2026-03-25")
        self.assertEqual(classification.region_hint, "Астане")

    def test_normalize_relative_time_references_supports_explicit_now(self):
        normalized = normalize_relative_time_references(
            "что сегодня было в Астане",
            now=__import__("datetime").datetime(2026, 3, 25),
        )
        self.assertEqual(normalized, "что 2026-03-25 было в Астане")

    def test_decompose_claims_maps_structured_output_to_claims(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        classification = QueryClassification(
            query="Compare OpenAI revenue and Microsoft revenue in 2025",
            normalized_query="Compare OpenAI revenue and Microsoft revenue in 2025",
            intent="comparison",
            complexity="multi_hop",
            needs_freshness=False,
            time_scope="2025",
        )
        result = type(
            "Result",
            (),
            {
                "output": _ClaimDecompositionOutput(
                    claims=[
                        _ClaimDraft(
                            claim_text="OpenAI revenue in 2025",
                            priority=1,
                            needs_freshness=False,
                            entity_set=["OpenAI"],
                            time_scope="2025",
                            claim_profile=_ClaimProfileOutput(
                                answer_shape="exact_number",
                                required_dimensions=["number"],
                                focus_terms=["revenue", "2025"],
                            ),
                        ),
                        _ClaimDraft(
                            claim_text="Microsoft revenue in 2025",
                            priority=2,
                            needs_freshness=False,
                            entity_set=["Microsoft"],
                            time_scope="2025",
                            claim_profile=_ClaimProfileOutput(
                                answer_shape="exact_number",
                                required_dimensions=["number"],
                                focus_terms=["revenue", "2025"],
                            ),
                        ),
                    ]
                )
            },
        )()

        with patch.object(service._claim_agent, "run_sync", return_value=result):
            claims = service.decompose_claims(classification)

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].entity_set, ["OpenAI"])
        self.assertEqual(claims[1].entity_set, ["Microsoft"])
        self.assertEqual(claims[0].claim_profile.focus_terms, ["revenue", "2025"])

    def test_verify_claim_maps_structured_quotes_to_spans(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="Satya Nadella is the CEO of Microsoft.",
            priority=1,
            needs_freshness=False,
            entity_set=["Satya Nadella", "Microsoft"],
        )
        passage = Passage(
            passage_id="p1",
            url="https://news.microsoft.com/source/exec/satya-nadella/",
            canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
            host="news.microsoft.com",
            title="Satya Nadella",
            section="Leadership",
            published_at="2026-03-24T00:00:00+00:00",
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p1",
            text="Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
            source_score=0.9,
            utility_score=0.9,
        )
        result = type(
            "Result",
            (),
            {
                "output": _VerificationOutput(
                    verdict="supported",
                    confidence=0.95,
                    supporting_passages=[_EvidenceQuote(passage_id="p1", quote="Chief Executive Officer of Microsoft")],
                    rationale="Direct official leadership page.",
                )
            },
        )()

        with patch.object(service._verifier_agent, "run_sync", return_value=result):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "supported")
        self.assertEqual(len(verification.supporting_spans), 1)
        self.assertIn("Chief Executive Officer", verification.supporting_spans[0].text)

    def test_verify_claim_floors_supported_confidence_when_near_zero(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="Satya Nadella leads Microsoft.",
            priority=1,
            needs_freshness=False,
            entity_set=["Microsoft"],
        )
        passage = Passage(
            passage_id="p1",
            url="https://news.microsoft.com/source/exec/satya-nadella/",
            canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
            host="news.microsoft.com",
            title="Satya Nadella",
            section="Leadership",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p1",
            text="Satya Nadella is Chairman and CEO of Microsoft.",
            source_score=0.9,
            utility_score=0.9,
        )
        result = type(
            "Result",
            (),
            {
                "output": _VerificationOutput(
                    verdict="supported",
                    confidence=0.0,
                    supporting_passages=[_EvidenceQuote(passage_id="p1", quote="CEO")],
                    rationale="",
                )
            },
        )()

        with patch.object(service._verifier_agent, "run_sync", return_value=result):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "supported")
        self.assertGreaterEqual(verification.confidence, 0.38)

    def test_verify_claim_preserves_llm_insufficient_without_template_boost(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="Python 3.8 introduced assignment expressions (walrus operator).",
            priority=1,
            needs_freshness=False,
            entity_set=["Python"],
        )
        long_body = "Introductory prose about assignment expressions. " * 20
        passage = Passage(
            passage_id="p-doc",
            url="https://docs.python.org/3/whatsnew/3.8.html",
            canonical_url="https://docs.python.org/3/whatsnew/3.8.html",
            host="docs.python.org",
            title="What is new in Python 3.8",
            section="",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p-doc",
            text=long_body,
            source_score=0.9,
            utility_score=0.3,
        )
        result = type(
            "Result",
            (),
            {
                "output": _VerificationOutput(
                    verdict="insufficient_evidence",
                    confidence=0.15,
                    supporting_passages=[],
                    rationale="Need clearer quote.",
                )
            },
        )()

        with patch.object(service._verifier_agent, "run_sync", return_value=result):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "insufficient_evidence")
        self.assertEqual(verification.supporting_spans, [])

    def test_verify_claim_uses_rescue_llm_pass_when_primary_fails(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="Who is the CEO of Microsoft?",
            priority=1,
            needs_freshness=False,
            entity_set=["Microsoft"],
        )
        passage = Passage(
            passage_id="p-role",
            url="https://news.microsoft.com/source/exec/satya-nadella/",
            canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
            host="news.microsoft.com",
            title="Satya Nadella",
            section="Leadership",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p-role",
            text="Satya Nadella is Chairman and Chief Executive Officer of Microsoft. Before being named CEO, he held leadership roles across the company.",
            source_score=0.9,
            utility_score=0.4,
        )
        result = type(
            "Result",
            (),
            {
                "output": _VerificationOutput(
                    verdict="supported",
                    confidence=0.72,
                    supporting_passages=[_EvidenceQuote(passage_id="p-role", quote="Chief Executive Officer of Microsoft")],
                    rationale="Official leadership page directly names the CEO.",
                )
            },
        )()

        with patch.object(service._verifier_agent, "run_sync", side_effect=[RuntimeError("provider error"), result]):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "supported")
        self.assertGreaterEqual(verification.confidence, 0.72)
        self.assertTrue(verification.supporting_spans)
        self.assertIn("Chief Executive Officer of Microsoft", verification.supporting_spans[0].text)

    def test_verify_claim_returns_default_insufficient_after_double_failure(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="When was Python 3.13.0 released?",
            priority=1,
            needs_freshness=False,
            entity_set=["Python"],
        )
        passage = Passage(
            passage_id="p-release",
            url="https://www.python.org/downloads/release/python-3130/",
            canonical_url="https://www.python.org/downloads/release/python-3130/",
            host="www.python.org",
            title="Python 3.13.0",
            section="Release",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p-release",
            text="Python 3.13.0, released on October 7, 2024, includes new features and bug fixes. " * 8,
            source_score=0.9,
            utility_score=0.3,
        )
        with patch.object(service._verifier_agent, "run_sync", side_effect=[RuntimeError("provider error"), RuntimeError("provider error")]):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "insufficient_evidence")
        self.assertIn("after retry", verification.rationale)

    def test_synthesize_answer_news_digest_keeps_four_source_footer_entries(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        passages = [
            Passage(
                passage_id=f"p{i}",
                url=f"https://source{i}.example/story",
                canonical_url=f"https://source{i}.example/story",
                host=f"source{i}.example",
                title=f"Story {i}",
                section="News",
                published_at="2026-03-24T00:00:00+00:00",
                author=None,
                extracted_at="2026-03-24T00:00:00+00:00",
                chunk_id=f"p{i}",
                text=f"Important development {i}.",
                source_score=0.8,
                utility_score=0.8,
            )
            for i in range(1, 5)
        ]
        result = type("Result", (), {"output": "Summary line [1]"})()

        with patch.object(service._synth_agent, "run_sync", return_value=result):
            answer = service.synthesize_answer(
                "What are the latest news on the Iran conflict?",
                passages,
                intent="news_digest",
            )

        self.assertIn("[1] Story 1", answer)
        self.assertIn("[2] Story 2", answer)
        self.assertIn("[3] Story 3", answer)
        self.assertIn("[4] Story 4", answer)


if __name__ == "__main__":
    unittest.main()
