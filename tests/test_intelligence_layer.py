import unittest
from unittest.mock import patch

from search_agent.application.text_heuristics import normalize_relative_time_references
from search_agent.domain.models import Claim, Passage, QueryClassification
from search_agent.infrastructure.intelligence import (
    PydanticAIQueryIntelligence,
    _ClaimDecompositionOutput,
    _ClaimDraft,
    _NormalizedQueryOutput,
    _VerificationOutput,
    _EvidenceQuote,
)
from search_agent.settings import AppSettings


class IntelligenceLayerTests(unittest.TestCase):
    def test_classify_query_uses_deterministic_relative_date_normalization(self):
        service = PydanticAIQueryIntelligence(
            AppSettings(
                llm_api_key="test-key",
                logfire_send_to_logfire="false",
            )
        )

        with patch("search_agent.infrastructure.intelligence.normalize_relative_time_references", return_value="OpenAI revenue 2026-03-25"):
            with patch.object(service._normalize_agent, "run_sync") as normalize_call:
                classification = service.classify_query("OpenAI revenue today")

        self.assertEqual(classification.normalized_query, "OpenAI revenue 2026-03-25")
        self.assertEqual(classification.time_scope, "2026-03-25")
        normalize_call.assert_not_called()

    def test_classify_query_marks_news_digest_intent(self):
        service = PydanticAIQueryIntelligence(
            AppSettings(
                llm_api_key="test-key",
                logfire_send_to_logfire="false",
            )
        )

        with patch("search_agent.infrastructure.intelligence.normalize_relative_time_references", return_value="что 2026-03-25 было в Астане"):
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
        service = PydanticAIQueryIntelligence(
            AppSettings(
                llm_api_key="test-key",
                logfire_send_to_logfire="false",
            )
        )
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
                        ),
                        _ClaimDraft(
                            claim_text="Microsoft revenue in 2025",
                            priority=2,
                            needs_freshness=False,
                            entity_set=["Microsoft"],
                            time_scope="2025",
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

    def test_verify_claim_maps_structured_quotes_to_spans(self):
        service = PydanticAIQueryIntelligence(
            AppSettings(
                llm_api_key="test-key",
                logfire_send_to_logfire="false",
            )
        )
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


if __name__ == "__main__":
    unittest.main()
